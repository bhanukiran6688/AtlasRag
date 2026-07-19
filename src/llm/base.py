import hashlib
import json
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from src.cache.in_memory_cache import InMemoryTTLCache
from src.config.settings import settings
from src.utils.exceptions import GatewayError, RAGConfigurationError
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None


@dataclass(slots=True)
class LLMResponse:
    content: str
    model: str
    latency_ms: float
    usage: LLMUsage = field(default_factory=LLMUsage)
    raw_response: Any | None = None
    cache_hit: bool = False
    route: str = "general"
    error: str | None = None
    request_id: str | None = None


class LLMGateway(ABC):
    """
    Provider-neutral boundary for text generation.

    RAGService should depend on this contract, not on Gemini, Groq, Cerebras,
    or LiteLLM directly.
    """

    @abstractmethod
    def validate_connection(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        routing_text: str | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        raise NotImplementedError


class LiteLLMGateway(LLMGateway):
    """
    Production LLM gateway backed by LiteLLM Router.

    The router owns provider fallback and Gemini key load balancing. This keeps
    provider mechanics isolated from the retrieval and prompt-building layers.
    """

    # Validate that at least one configured LLM provider can serve RAG answers.
    def validate_connection(self) -> None:
        if (
            not settings.gemini_api_keys
            and not settings.groq_api_key
            and not settings.cerebras_api_key
        ):
            raise RAGConfigurationError(
                "No LLM API keys are configured for the gateway."
            )

    # Initialize LiteLLM routing, response caching, and local rate-limit state.
    def __init__(self) -> None:
        self._router = self._build_router()
        self._cache: InMemoryTTLCache[LLMResponse] = InMemoryTTLCache()
        self._request_timestamps: deque[float] = deque()
        logger.info(
            "LiteLLM gateway initialized with primary model '%s' and %d fallback model(s).",
            settings.llm_primary_model,
            len(settings.fallback_model_names),
        )

    async def generate(
        self,
        prompt: str,
        *,
        routing_text: str | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        route = self._classify_route(routing_text or prompt)
        model = self._model_for_route(route)
        request_id = hashlib.sha256(
            f"{time.time_ns()}:{prompt[:120]}".encode("utf-8")
        ).hexdigest()[:12]
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_output_tokens,
        }
        if response_format:
            payload["response_format"] = dict(response_format)

        cache_key = self._build_cache_key(payload)
        # Production note: a bounded in-memory cache reduces repeated provider cost for
        # common prompts, but it should stay lightweight and replaceable.
        cached_response = self._cache.get(cache_key)
        if cached_response:
            logger.info("LLM cache hit. model=%s", cached_response.model)
            return LLMResponse(
                content=cached_response.content,
                model=cached_response.model,
                latency_ms=0.0,
                usage=cached_response.usage,
                cache_hit=True,
                route=cached_response.route,
                request_id=request_id,
            )

        budget_error = self._budget_error(prompt)
        if budget_error:
            logger.warning(budget_error)
            return LLMResponse(
                content="",
                model=model,
                latency_ms=0.0,
                route=route,
                error=budget_error,
                request_id=request_id,
            )

        rate_limit_error = self._rate_limit_error()
        if rate_limit_error:
            logger.warning(rate_limit_error)
            return LLMResponse(
                content="",
                model=model,
                latency_ms=0.0,
                route=route,
                error=rate_limit_error,
                request_id=request_id,
            )

        started_at = time.perf_counter()
        response = None
        last_error: Exception | None = None
        for candidate_model in self._fallback_chain(model):
            payload["model"] = candidate_model
            try:
                response = await self._router.acompletion(**payload)
                break
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LLM model failed. model=%s error=%s",
                    candidate_model,
                    type(exc).__name__,
                )

        if response is None:
            latency_ms = (time.perf_counter() - started_at) * 1000
            error_type = type(last_error).__name__ if last_error else "UnknownError"
            return LLMResponse(
                content="",
                model=model,
                latency_ms=latency_ms,
                route=route,
                error=f"LLM request failed after fallback attempts: {error_type}",
                request_id=request_id,
            )

        latency_ms = (time.perf_counter() - started_at) * 1000

        content = response.choices[0].message.content or ""
        response_model = getattr(response, "model", model)
        usage = self._parse_usage(response)

        logger.info(
            "LLM request completed. request_id=%s route=%s model=%s latency_ms=%.2f tokens=%d cost_usd=%s",
            request_id,
            route,
            response_model,
            latency_ms,
            usage.total_tokens,
            usage.cost_usd,
        )

        llm_response = LLMResponse(
            content=content,
            model=response_model,
            latency_ms=latency_ms,
            usage=usage,
            raw_response=response,
            route=route,
            request_id=request_id,
        )
        self._cache.set(cache_key, llm_response)
        return llm_response

    # Build the LiteLLM Router with provider deployments and fallback settings.
    def _build_router(self) -> Any:
        try:
            from litellm import Router
        except ImportError as exc:
            raise RuntimeError(
                "LiteLLM is required for the production LLM gateway. "
                "Install it with `pip install litellm`."
            ) from exc

        model_list = self._build_model_list()
        fallbacks = (
            [{settings.llm_primary_model: settings.fallback_model_names}]
            if settings.fallback_model_names
            else []
        )

        return Router(
            model_list=model_list,
            fallbacks=fallbacks,
            num_retries=settings.llm_max_retries,
            retry_after=settings.llm_retry_backoff_seconds,
            timeout=settings.llm_timeout_seconds,
            routing_strategy="simple-shuffle",
            enable_weighted_failover=True,
        )

    # Convert configured provider keys into LiteLLM Router deployment records.
    def _build_model_list(self) -> list[dict[str, Any]]:
        model_list: list[dict[str, Any]] = []

        for index, api_key in enumerate(settings.gemini_api_keys, start=1):
            model_list.append(
                {
                    "model_name": settings.llm_primary_model,
                    "litellm_params": {
                        "model": settings.llm_primary_model,
                        "api_key": api_key,
                    },
                    "model_info": {"id": f"gemini-key-{index}"},
                }
            )

        if settings.groq_api_key:
            model_list.append(
                {
                    "model_name": "groq/llama-3.3-70b-versatile",
                    "litellm_params": {
                        "model": "groq/llama-3.3-70b-versatile",
                        "api_key": settings.groq_api_key,
                    },
                }
            )

        if settings.cerebras_api_key:
            model_list.append(
                {
                    "model_name": "cerebras/llama3.1-8b",
                    "litellm_params": {
                        "model": "cerebras/llama3.1-8b",
                        "api_key": settings.cerebras_api_key,
                    },
                }
            )

        if not model_list:
            raise GatewayError("No LLM API keys are configured for LiteLLM gateway.")

        return model_list

    # Extract token and cost metadata from provider responses when available.
    def _parse_usage(self, response: Any) -> LLMUsage:
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
        cost_usd = getattr(response, "_hidden_params", {}).get("response_cost")

        return LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd if settings.llm_enable_cost_tracking else None,
        )

    # Choose a lightweight model route based on the user task type.
    def _classify_route(self, text: str) -> str:
        text_lower = text.lower()

        code_markers = (
            "code",
            "python",
            "function",
            "class ",
            "debug",
            "error",
            "stack trace",
            "api",
            "sql",
        )
        summary_markers = (
            "summarize",
            "summary",
            "brief",
            "short version",
            "tl;dr",
            "key points",
        )

        if any(marker in text_lower for marker in code_markers):
            return "code"
        if any(marker in text_lower for marker in summary_markers):
            return "summary"
        return "general"

    # Resolve the configured model for a route while honoring provider availability.
    def _model_for_route(self, route: str) -> str:
        if route == "code":
            return self._configured_or_primary(settings.llm_code_model)
        if route == "summary":
            return self._configured_or_primary(settings.llm_summary_model)
        return self._configured_or_primary(settings.llm_general_model)

    # Build an ordered model fallback chain that skips unconfigured providers.
    def _fallback_chain(self, model: str) -> list[str]:
        candidates = [model]
        candidates.extend(settings.fallback_model_names)

        configured_candidates: list[str] = []
        for candidate in candidates:
            if candidate in configured_candidates:
                continue
            if self._is_model_configured(candidate):
                configured_candidates.append(candidate)

        return configured_candidates or [settings.llm_primary_model]

    # Use a route model only when its provider credentials are configured.
    def _configured_or_primary(self, model: str) -> str:
        if self._is_model_configured(model):
            return model

        logger.warning(
            "Configured route model '%s' is not available. Falling back to '%s'.",
            model,
            settings.llm_primary_model,
        )
        return settings.llm_primary_model

    # Check whether a model's provider has usable credentials.
    def _is_model_configured(self, model: str) -> bool:
        if model.startswith("gemini/"):
            return bool(settings.gemini_api_keys)
        if model.startswith("groq/"):
            return bool(settings.groq_api_key)
        if model.startswith("cerebras/"):
            return bool(settings.cerebras_api_key)
        return model == settings.llm_primary_model

    # Enforce a process-local LLM request rate limit before provider calls.
    def _rate_limit_error(self) -> str | None:
        now = time.time()
        window_start = now - settings.llm_rate_limit_window_seconds

        while self._request_timestamps and self._request_timestamps[0] <= window_start:
            self._request_timestamps.popleft()

        if len(self._request_timestamps) >= settings.llm_rate_limit_requests:
            return (
                "LLM rate limit exceeded. "
                f"Try again in {settings.llm_rate_limit_window_seconds} seconds."
            )

        self._request_timestamps.append(now)
        return None

    # Estimate prompt cost before making a provider call.
    def _budget_error(self, prompt: str) -> str | None:
        estimated_tokens = self._estimate_tokens(prompt)
        if (
            settings.llm_max_estimated_prompt_tokens > 0
            and estimated_tokens > settings.llm_max_estimated_prompt_tokens
        ):
            return (
                "LLM prompt budget exceeded. "
                f"estimated_tokens={estimated_tokens} limit={settings.llm_max_estimated_prompt_tokens}"
            )

        if (
            settings.llm_max_estimated_cost_usd_per_request > 0
            and settings.llm_estimated_cost_per_1k_tokens > 0
        ):
            estimated_cost = (
                estimated_tokens / 1000
            ) * settings.llm_estimated_cost_per_1k_tokens
            if estimated_cost > settings.llm_max_estimated_cost_usd_per_request:
                return (
                    "LLM cost budget exceeded. "
                    f"estimated_cost_usd={estimated_cost:.6f} limit={settings.llm_max_estimated_cost_usd_per_request:.6f}"
                )

        return None

    # Use a fast local token approximation for pre-call budget checks.
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, int(len(text.split()) * 1.3))

    # Build a stable cache key from prompt and generation settings.
    def _build_cache_key(self, payload: Mapping[str, Any]) -> str:
        cache_payload = {
            "model": payload.get("model"),
            "messages": payload.get("messages"),
            "temperature": payload.get("temperature"),
            "max_tokens": payload.get("max_tokens"),
            "response_format": payload.get("response_format"),
        }
        serialized_payload = json.dumps(cache_payload, sort_keys=True)
        return hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()


class LLMGatewayFactory:
    # Create the configured LLM gateway after validating required settings.
    @staticmethod
    def create() -> LLMGateway:
        settings.validate_llm_configuration()
        return LiteLLMGateway()
