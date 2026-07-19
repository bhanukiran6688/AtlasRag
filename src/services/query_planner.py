import hashlib
import json
import re
from functools import lru_cache

from src.config.settings import settings
from src.llm.base import LLMGateway
from src.utils.logger import get_logger


logger = get_logger(__name__)


class QueryPlanner:
    """Builds optional expansion and decomposition queries for retrieval with caching."""

    def __init__(self, llm_gateway: LLMGateway) -> None:
        self._llm_gateway = llm_gateway
        self._expansion_cache: dict[str, list[str]] = {}
        self._decomposition_cache: dict[str, list[str]] = {}
        self._cache_max_size = 100

    async def build_queries(
        self,
        question: str,
        *,
        use_query_expansion: bool,
        use_query_decomposition: bool,
        retrieval_confidence: float | None = None,
    ) -> list[str]:
        queries = [question]
        use_query_expansion = use_query_expansion and self._should_expand(question, retrieval_confidence)
        use_query_decomposition = use_query_decomposition and self._should_decompose(question, retrieval_confidence)
        planning_calls_remaining = max(settings.cost_optimization_max_query_planning_calls, 0)

        if use_query_expansion:
            expanded_queries = []
            if planning_calls_remaining > 0:
                expanded_queries = await self._generate_variants_cached(question)
                planning_calls_remaining -= 1
            queries.extend(expanded_queries or self._heuristic_variants(question))

        if use_query_decomposition:
            subquestions = []
            if planning_calls_remaining > 0:
                subquestions = await self._generate_subquestions_cached(question)
            queries.extend(subquestions or self._heuristic_subquestions(question))

        return self._deduplicate(queries)

    def _should_expand(self, question: str, retrieval_confidence: float | None) -> bool:
        if not settings.query_planning_adaptive_enabled:
            return True
        return self._is_low_confidence(retrieval_confidence) or len(question.split()) >= settings.query_planning_min_words_for_expansion

    def _should_decompose(self, question: str, retrieval_confidence: float | None) -> bool:
        if not settings.query_planning_adaptive_enabled:
            return True
        has_multiple_parts = bool(
            re.search(r"\b(and|or|versus|vs|compare|difference|steps)\b", question, re.IGNORECASE)
        )
        return (
            self._is_low_confidence(retrieval_confidence)
            or has_multiple_parts
            or len(question.split()) >= settings.query_planning_min_words_for_decomposition
        )

    @staticmethod
    def _is_low_confidence(retrieval_confidence: float | None) -> bool:
        return (
            retrieval_confidence is not None
            and retrieval_confidence < settings.query_planning_low_confidence_threshold
        )

    async def _generate_variants_cached(self, question: str) -> list[str]:
        """Generate query variants with caching to reduce LLM calls."""
        cache_key = self._get_cache_key(question)
        
        if cache_key in self._expansion_cache:
            logger.info("Query expansion cache hit for: %s", question[:50])
            return self._expansion_cache[cache_key]
        
        variants = await self._generate_variants(question)
        
        # Cache the result
        if len(self._expansion_cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._expansion_cache))
            del self._expansion_cache[oldest_key]
        
        self._expansion_cache[cache_key] = variants
        logger.info("Cached query expansion results for: %s", question[:50])
        
        return variants

    async def _generate_subquestions_cached(self, question: str) -> list[str]:
        """Generate subquestions with caching to reduce LLM calls."""
        cache_key = self._get_cache_key(question)
        
        if cache_key in self._decomposition_cache:
            logger.info("Query decomposition cache hit for: %s", question[:50])
            return self._decomposition_cache[cache_key]
        
        subquestions = await self._generate_subquestions(question)
        
        # Cache the result
        if len(self._decomposition_cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._decomposition_cache))
            del self._decomposition_cache[oldest_key]
        
        self._decomposition_cache[cache_key] = subquestions
        logger.info("Cached query decomposition results for: %s", question[:50])
        
        return subquestions

    def _get_cache_key(self, question: str) -> str:
        """Generate a cache key from the question."""
        normalized = " ".join(question.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()

    async def _generate_variants(self, question: str) -> list[str]:
        prompt = (
            "Generate up to 3 alternative search queries for a vector database. "
            "Keep each query short and preserve the original meaning.\n\n"
            f"Question: {question}\n\n"
            'Return ONLY valid JSON: {"queries": ["query 1", "query 2"]}'
        )
        return await self._generate_list(
            prompt,
            question,
            "queries",
            settings.query_expansion_max_variants,
            "Query variant generation failed; falling back to heuristics.",
        )

    async def _generate_subquestions(self, question: str) -> list[str]:
        prompt = (
            "Break the user question into up to 3 focused retrieval subquestions. "
            "Use this only to improve document search, not to answer the question.\n\n"
            f"Question: {question}\n\n"
            'Return ONLY valid JSON: {"subquestions": ["question 1", "question 2"]}'
        )
        return await self._generate_list(
            prompt,
            question,
            "subquestions",
            settings.query_decomposition_max_subquestions,
            "Sub-question generation failed; falling back to heuristics.",
        )

    async def _generate_list(
        self,
        prompt: str,
        question: str,
        key: str,
        limit: int,
        error_message: str,
    ) -> list[str]:
        try:
            response = await self._llm_gateway.generate(
                prompt,
                routing_text=question,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            logger.warning("%s error=%s", error_message, exc)
            return []
        return self._extract_string_list(response.content, key, limit)

    def _heuristic_variants(self, question: str) -> list[str]:
        normalized_question = re.sub(r"\s+", " ", question).strip()
        if not normalized_question:
            return []
        keywords = self._extract_keywords(normalized_question)
        variants = [normalized_question, f"{normalized_question} details", f"{normalized_question} examples"]
        if keywords:
            variants.extend((" ".join(keywords), f"{keywords[0]} {keywords[-1]}"))
        return [variant for variant in variants if variant][:settings.query_expansion_max_variants]

    def _heuristic_subquestions(self, question: str) -> list[str]:
        normalized_question = re.sub(r"\s+", " ", question).strip()
        if not normalized_question:
            return []
        parts = [part.strip() for part in re.split(r"\b(and|or|but)\b", normalized_question, flags=re.IGNORECASE) if part.strip()]
        if len(parts) <= 1:
            keywords = self._extract_keywords(normalized_question)
            return [f"What is {keywords[0]}?"] if keywords else []
        subquestions = [part for part in parts if len(part.split()) <= 8]
        return subquestions[:settings.query_decomposition_max_subquestions] or [normalized_question]

    @staticmethod
    def _extract_keywords(question: str) -> list[str]:
        cleaned_question = re.sub(r"[^a-zA-Z0-9\s]", " ", question).lower()
        stop_words = {"the", "a", "an", "is", "what", "how", "why", "who", "when", "where", "can", "does", "do", "you", "your", "for", "to", "of", "and", "or", "but"}
        return [token for token in cleaned_question.split() if token not in stop_words][:4]

    @staticmethod
    def _deduplicate(queries: list[str]) -> list[str]:
        deduplicated_queries: list[str] = []
        seen_queries: set[str] = set()
        for query in queries:
            normalized_query = " ".join(query.split())
            if normalized_query and normalized_query.lower() not in seen_queries:
                seen_queries.add(normalized_query.lower())
                deduplicated_queries.append(normalized_query)
        return deduplicated_queries

    @staticmethod
    def _extract_string_list(content: str, key: str, limit: int) -> list[str]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return []
        values = parsed.get(key, []) if isinstance(parsed, dict) else []
        return [str(value).strip() for value in values if str(value).strip()][:limit] if isinstance(values, list) else []
