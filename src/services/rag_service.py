import asyncio
import json
from dataclasses import dataclass
from typing import Any

from src.config.settings import settings
from src.utils.exceptions import GatewayError, RetrievalError
from src.utils.logger import get_logger
from src.llm.base import LLMGateway, LLMResponse
from src.guardrails.input_guardrails import InputGuardrails, OutputGuardrails
from src.retrievers.retriever import RetrievalResult, Retriever
from src.prompts.prompt_builder import PromptBuilder, ContextBuilder
from src.services.query_planner import QueryPlanner
from src.utils.metadata_filters import build_metadata_filter, parse_metadata_filter
from src.schemas.structured_output import StructuredRAGOutput

logger = get_logger(__name__)


@dataclass(slots=True)
class RAGResult:
    """
    Represents the output of the retrieval pipeline.

    Later this class will also contain:
    - answer
    - citations
    - token usage
    - latency
    """

    question: str
    retrieved_chunks: list[RetrievalResult]
    context: str
    prompt: str
    answer: str
    llm_model: str | None = None
    llm_latency_ms: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    sanitized_question: str | None = None
    is_blocked: bool = False
    blocked_reason: str | None = None
    pii_detected: list[str] | None = None
    cache_hit: bool = False
    route: str | None = None
    structured_output: dict[str, Any] | None = None
    error: str | None = None
    retrieval_queries: list[str] | None = None
    is_grounded: bool = True
    grounding_reason: str | None = None
    output_blocked: bool = False
    grounding_confidence: float | None = None
    request_id: str | None = None


class RAGService:
    """
    Orchestrates the Retrieval-Augmented Generation pipeline.

    Current pipeline:

        Question
            │
            ▼
        Retriever
            │
            ▼
        ContextBuilder
            │
            ▼
        PromptBuilder
            |
            v
        LLMGateway
    """

    def __init__(
        self,
        retriever: Retriever,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        llm_gateway: LLMGateway,
        input_guardrails: InputGuardrails | None = None,
        output_guardrails: OutputGuardrails | None = None,
    ) -> None:

        self._retriever = retriever
        self._context_builder = context_builder
        self._prompt_builder = prompt_builder
        self._llm_gateway = llm_gateway
        self._query_planner = QueryPlanner(llm_gateway)
        self._input_guardrails = input_guardrails or InputGuardrails()
        self._output_guardrails = output_guardrails or OutputGuardrails()

        logger.info("RAGService initialized.")

    @property
    def retriever(self) -> Retriever:
        return self._retriever

    def process(
        self,
        question: str,
        *,
        use_query_expansion: bool = False,
        use_query_decomposition: bool = False,
        metadata_filter: dict[str, Any] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> RAGResult:
        """
        Synchronous wrapper for Streamlit and simple scripts.
        """

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.aprocess(
                    question,
                    use_query_expansion=use_query_expansion,
                    use_query_decomposition=use_query_decomposition,
                    metadata_filter=metadata_filter,
                    conversation_history=conversation_history,
                )
            )

        raise RuntimeError(
            "Use `await aprocess(...)` when already inside an async event loop."
        )

    async def aprocess(
        self,
        question: str,
        *,
        use_query_expansion: bool = False,
        use_query_decomposition: bool = False,
        metadata_filter: dict[str, Any] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> RAGResult:
        """
        Process a user question through retrieval, prompt building, and generation.
        """

        logger.info("Processing question.")
        try:
            guardrail_result = self._input_guardrails.validate(question)
            if guardrail_result.is_blocked:
                return RAGResult(
                    question=question,
                    retrieved_chunks=[],
                    context="",
                    prompt="",
                    answer=guardrail_result.blocked_reason
                    or "Request blocked by input guardrails.",
                    sanitized_question=guardrail_result.sanitized_text,
                    is_blocked=True,
                    blocked_reason=guardrail_result.blocked_reason,
                    pii_detected=guardrail_result.pii_detected,
                )

            sanitized_question = guardrail_result.sanitized_text
            effective_filter = self._build_metadata_filter(metadata_filter)
            initial_retrieved_chunks: list[RetrievalResult] | None = None
            initial_grounding_confidence: float | None = None
            if settings.query_planning_adaptive_enabled and (
                use_query_expansion or use_query_decomposition
            ):
                initial_retrieved_chunks = self._retrieve_for_queries(
                    queries=[sanitized_question],
                    metadata_filter=effective_filter,
                )
                initial_grounding_confidence = self._compute_grounding_confidence(
                    initial_retrieved_chunks
                )

            retrieval_queries = await self._query_planner.build_queries(
                sanitized_question,
                use_query_expansion=use_query_expansion,
                use_query_decomposition=use_query_decomposition,
                retrieval_confidence=initial_grounding_confidence,
            )
            if initial_retrieved_chunks is not None and retrieval_queries == [
                sanitized_question
            ]:
                retrieved_chunks = initial_retrieved_chunks
            else:
                retrieved_chunks = self._retrieve_for_queries(
                    queries=retrieval_queries,
                    metadata_filter=effective_filter,
                )
            context = self._context_builder.build(retrieved_chunks)
            if not context:
                logger.warning("No context could be built.")
                return RAGResult(
                    question=question,
                    retrieved_chunks=retrieved_chunks,
                    context="",
                    prompt="",
                    answer="",
                    sanitized_question=sanitized_question,
                    pii_detected=guardrail_result.pii_detected,
                    retrieval_queries=retrieval_queries,
                )

            # FEATURE: Conversation memory support
            memory_history = self._normalize_conversation_history(conversation_history)
            # FEATURE: Cost optimization / prompt simplification
            prompt = self._prompt_builder.build(
                question=sanitized_question,
                context=context,
                conversation_history=memory_history,
                max_context_chunks=settings.cost_optimization_max_context_chunks,
                simplify_prompt=settings.cost_optimization_enable_prompt_simplification,
            )
            structured_prompt = self._with_structured_output_instructions(prompt)
            llm_response = await self._llm_gateway.generate(
                structured_prompt,
                routing_text=sanitized_question,
                response_format={"type": "json_object"},
            )

            if llm_response.error:
                return RAGResult(
                    question=question,
                    retrieved_chunks=retrieved_chunks,
                    context=context,
                    prompt=structured_prompt,
                    answer="I could not generate an answer right now. Please try again.",
                    llm_model=llm_response.model,
                    llm_latency_ms=llm_response.latency_ms,
                    sanitized_question=sanitized_question,
                    pii_detected=guardrail_result.pii_detected,
                    cache_hit=llm_response.cache_hit,
                    route=llm_response.route,
                    error=llm_response.error,
                    retrieval_queries=retrieval_queries,
                    request_id=llm_response.request_id,
                )

            logger.info("RAG pipeline completed successfully.")
            return await self._build_result(
                question=question,
                sanitized_question=sanitized_question,
                retrieved_chunks=retrieved_chunks,
                context=context,
                prompt=structured_prompt,
                llm_response=llm_response,
                pii_detected=guardrail_result.pii_detected,
                retrieval_queries=retrieval_queries,
            )
        except RetrievalError as exc:
            logger.exception("Retrieval failed during RAG processing.")
            return RAGResult(
                question=question,
                retrieved_chunks=[],
                context="",
                prompt="",
                answer="I could not retrieve enough relevant context right now.",
                error=str(exc),
            )
        except GatewayError as exc:
            logger.exception("LLM gateway failed during RAG processing.")
            return RAGResult(
                question=question,
                retrieved_chunks=[],
                context="",
                prompt="",
                answer="The language model service is unavailable right now.",
                error=str(exc),
            )

    # Normalize and bound conversation history before prompt injection.
    def _normalize_conversation_history(
        self, conversation_history: list[dict[str, str]] | None
    ) -> list[dict[str, str]]:
        """Normalize conversation history and apply both turn and token limits.

        This method uses a two-stage limiting strategy:
        1. First limit by turns (conversation_memory_max_turns) as a safety measure
        2. Then limit by tokens (conversation_memory_max_tokens) for accurate prompt sizing

        Token-based limiting is more accurate than turn-based because:
        - Different turns can have vastly different lengths
        - Prompt costs are based on tokens, not turns
        - This prevents unexpectedly long prompts from exceeding context windows
        """
        if not conversation_history:
            return []

        normalized_history: list[dict[str, str]] = []
        for turn in conversation_history:
            if isinstance(turn, dict) and "role" in turn and "content" in turn:
                normalized_history.append(
                    {
                        "role": str(turn.get("role", "user")),
                        "content": str(turn.get("content", "")),
                    }
                )

        # First limit by turns as a safety measure (prevents unbounded growth)
        turn_limited = normalized_history[-settings.conversation_memory_max_turns * 2 :]

        # Then limit by tokens (more accurate for prompt length and cost control)
        token_limited = self._limit_by_tokens(
            turn_limited, settings.conversation_memory_max_tokens
        )

        return token_limited

    def _limit_by_tokens(
        self, conversation_history: list[dict[str, str]], max_tokens: int
    ) -> list[dict[str, str]]:
        """Limit conversation history by estimated token count.

        This method processes conversation history from most recent to oldest,
        including turns until adding another turn would exceed the token limit.
        This ensures the most recent context is preserved while staying within
        the token budget for the LLM prompt.

        Token estimation uses a simple heuristic (~4 characters per token),
        which is reasonably accurate for English text. For production use,
        consider using a proper tokenizer like tiktoken for more accuracy.
        """
        if not conversation_history:
            return []

        # Estimate tokens (rough approximation: ~4 chars per token)
        def estimate_tokens(text: str) -> int:
            return len(text) // 4

        # Build history from most recent to oldest, tracking token count
        limited_history: list[dict[str, str]] = []
        total_tokens = 0

        # Process in reverse order (most recent first)
        for turn in reversed(conversation_history):
            turn_tokens = estimate_tokens(turn.get("content", ""))

            if total_tokens + turn_tokens > max_tokens and limited_history:
                # Adding this turn would exceed limit, stop here
                break

            limited_history.insert(0, turn)
            total_tokens += turn_tokens

        logger.info(
            "Conversation history limited to %d turns (%d estimated tokens, max %d)",
            len(limited_history),
            total_tokens,
            max_tokens,
        )

        return limited_history

    # Maintain bounded local conversation memory after each RAG answer.
    def update_conversation_history(
        self,
        conversation_history: list[dict[str, str]] | None,
        question: str,
        answer: str,
    ) -> list[dict[str, str]]:
        # FEATURE: Conversation memory support
        history = self._normalize_conversation_history(conversation_history)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        return history[-settings.conversation_memory_max_turns * 2 :]

    # Retrieve and deduplicate chunks across original, expanded, and decomposed queries.
    def _retrieve_for_queries(
        self,
        queries: list[str],
        metadata_filter: dict[str, Any] | None,
    ) -> list[RetrievalResult]:
        merged_results: list[RetrievalResult] = []
        seen_chunks: set[tuple[str, str, str]] = set()

        for query in queries:
            try:
                retrieved_results = self._retriever.retrieve(
                    query,
                    metadata_filter=metadata_filter,
                )
            except Exception as exc:
                raise RetrievalError(
                    f"Retrieval failed for query '{query}': {exc}"
                ) from exc

            for result in retrieved_results:
                chunk_key = (
                    result.source,
                    str(result.page),
                    result.document.page_content,
                )
                if chunk_key in seen_chunks:
                    continue
                seen_chunks.add(chunk_key)
                merged_results.append(result)

        return [
            RetrievalResult(
                rank=rank,
                document=result.document,
                distance=result.distance,
                retrieval_strategy=result.retrieval_strategy,
                rerank_score=result.rerank_score,
            )
            for rank, result in enumerate(merged_results[: self._retriever.k], start=1)
        ]

    # Merge configured and request-level metadata filters for retrieval.
    def _build_metadata_filter(
        self,
        request_filter: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        configured_filter: dict[str, Any] = {}
        if settings.retrieval_filter_metadata_json:
            try:
                configured_filter = parse_metadata_filter(
                    settings.retrieval_filter_metadata_json
                )
            except ValueError as exc:
                logger.warning("Configured metadata filter ignored. error=%s", exc)

        try:
            return build_metadata_filter(
                source=settings.retrieval_filter_source,
                file_type=settings.retrieval_filter_file_type,
                custom_metadata={**configured_filter, **(request_filter or {})},
                allowed_keys=settings.allowed_metadata_filter_keys or None,
            )
        except ValueError as exc:
            raise RetrievalError(f"Invalid metadata filter: {exc}") from exc

    # Build the final RAG result after structured parsing and output guardrails.
    async def _build_result(
        self,
        question: str,
        sanitized_question: str,
        retrieved_chunks: list[RetrievalResult],
        context: str,
        prompt: str,
        llm_response: LLMResponse,
        pii_detected: list[str],
        retrieval_queries: list[str],
    ) -> RAGResult:
        usage = llm_response.usage
        structured_output = RAGService._parse_structured_output(llm_response.content)
        answer = (
            structured_output.get("answer")
            if structured_output
            else llm_response.content
        )
        citations = RAGService._extract_citations(structured_output)
        claims = RAGService._extract_claims(structured_output)
        output_guardrail_result = await self._output_guardrails.validate(
            answer=answer or "",
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            claims=claims,
        )

        # FEATURE: Hallucination mitigation / unsupported-answer abstention
        if structured_output:
            structured_output["answer"] = output_guardrail_result.answer
            structured_output["citations"] = output_guardrail_result.citations

        if (
            output_guardrail_result.is_grounded is False
            and not output_guardrail_result.answer
        ):
            answer = "I don't have enough information in the provided documents."
        elif output_guardrail_result.is_grounded is False:
            answer = output_guardrail_result.answer
        else:
            answer = output_guardrail_result.answer

        return RAGResult(
            question=question,
            retrieved_chunks=retrieved_chunks,
            context=context,
            prompt=prompt,
            answer=answer,
            llm_model=llm_response.model,
            llm_latency_ms=llm_response.latency_ms,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost_usd=usage.cost_usd,
            sanitized_question=sanitized_question,
            pii_detected=pii_detected,
            cache_hit=llm_response.cache_hit,
            route=llm_response.route,
            structured_output=structured_output,
            error=llm_response.error,
            retrieval_queries=retrieval_queries,
            is_grounded=output_guardrail_result.is_grounded,
            grounding_reason=output_guardrail_result.grounding_reason,
            output_blocked=output_guardrail_result.is_blocked,
            grounding_confidence=self._compute_grounding_confidence(retrieved_chunks),
            request_id=llm_response.request_id,
        )

    # Estimate final answer grounding confidence from retrieved chunk scores.
    @staticmethod
    def _compute_grounding_confidence(
        retrieved_chunks: list[RetrievalResult],
    ) -> float | None:
        if not retrieved_chunks:
            return None

        # FEATURE: Grounding confidence signal
        top_score = max(
            (
                chunk.distance
                for chunk in retrieved_chunks
                if chunk.distance is not None
            ),
            default=0.0,
        )
        return round(min(1.0, max(0.0, 1.0 - (abs(top_score) / 10.0))), 3)

    # Extract citation objects from structured LLM output.
    @staticmethod
    def _extract_citations(
        structured_output: dict[str, Any] | None,
    ) -> list[dict[str, str]]:
        if not structured_output:
            return []

        citations = structured_output.get("citations", [])
        if not isinstance(citations, list):
            return []

        valid_citations: list[dict[str, str]] = []
        for citation in citations:
            if not isinstance(citation, dict):
                continue
            valid_citations.append(
                {
                    "source": str(citation.get("source", "Unknown")),
                    "page": str(citation.get("page", "Unknown")),
                }
            )

        return valid_citations

    # Extract claim-level citation records from structured LLM output.
    @staticmethod
    def _extract_claims(
        structured_output: dict[str, Any] | None,
    ) -> list[dict[str, object]]:
        if not structured_output:
            return []

        claims = structured_output.get("claims", [])
        if not isinstance(claims, list):
            return []

        valid_claims: list[dict[str, object]] = []
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            citations = claim.get("citations", [])
            if not isinstance(citations, list):
                citations = []
            valid_claims.append(
                {
                    "text": str(claim.get("text", "")),
                    "citations": citations,
                }
            )

        return valid_claims

    # Append strict JSON and citation instructions to the RAG prompt using Pydantic schema.
    @staticmethod
    def _with_structured_output_instructions(prompt: str) -> str:
        from src.schemas.structured_output import Citation, Claim

        schema_example = json.dumps(
            StructuredRAGOutput(
                answer="clear answer based only on the provided context",
                claims=[
                    Claim(
                        text="single factual statement",
                        citations=[
                            Citation(
                                source="source filename or path",
                                page="page number or Unknown",
                            )
                        ],
                    )
                ],
                citations=[
                    Citation(
                        source="source filename or path", page="page number or Unknown"
                    )
                ],
            ).to_dict(),
            indent=2,
        )

        return (
            prompt
            + "\n\nReturn ONLY valid JSON with this shape:\n"
            + schema_example
            + "\n\nEvery factual statement in answer must appear as one claim with at least one valid citation.\n"
        )

    # Parse the model response into the expected structured answer schema.
    @staticmethod
    def _parse_structured_output(content: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON content for structured output.")
            return None

        if not isinstance(parsed, dict):
            return None

        answer = parsed.get("answer")
        citations = parsed.get("citations", [])
        claims = parsed.get("claims", [])
        if not isinstance(answer, str):
            return None
        if not isinstance(citations, list):
            parsed["citations"] = []
        if not isinstance(claims, list):
            parsed["claims"] = []

        return parsed
