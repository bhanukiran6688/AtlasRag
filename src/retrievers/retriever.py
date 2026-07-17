import math
import re
from collections import Counter
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from langchain_core.documents import Document

from src.config.settings import settings
from src.utils.logger import get_logger
from src.vectorstores.base import VectorStore


logger = get_logger(__name__)


@dataclass(slots=True)
class RetrievalResult:
    """
    Represents a retrieved document together with retrieval metadata.
    """

    rank: int
    document: Document
    distance: float
    retrieval_strategy: str = "similarity"
    rerank_score: float | None = None

    @property
    def source(self) -> str:
        return self.document.metadata.get("source", "Unknown")

    @property
    def page(self) -> int | None:
        return self.document.metadata.get("page")

    @property
    def chunk_id(self) -> str:
        return self.document.metadata.get("chunk_id") or self.document.metadata.get("chunk_index", "Unknown")

    @property
    def chunk_length(self) -> int:
        return len(self.document.page_content)


class Retriever:
    """
    Retrieves relevant chunks using configurable production retrieval strategies.
    """

    def __init__(self, vector_store: VectorStore, k: int | None = None, score_threshold: float = 0.8) -> None:
        self._vector_store = vector_store
        self._k = k or settings.retrieval_top_k
        self._score_threshold = score_threshold
        self._reranker = None
        self._reranker_model_name: str | None = None
        self.last_retrieval_time_ms: float = 0.0
        self.last_total_results: int = 0
        self.last_returned_results: int = 0
        logger.info(
            "Retriever initialized with strategy=%s, k=%d, candidates=%d",
            settings.retrieval_strategy,
            self._k,
            settings.retrieval_candidate_k,
        )

    @property
    def k(self) -> int:
        return self._k

    @property
    def score_threshold(self) -> float:
        return self._score_threshold

    def retrieve(
        self,
        query: str,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        logger.info("Retrieving documents for query: %s", query)
        if metadata_filter:
            logger.info("Applying metadata filter: %s", metadata_filter)

        start = perf_counter()
        try:
            results = self._retrieve_by_strategy(query=query, metadata_filter=metadata_filter)
            results = self._rerank(query=query, results=results)
        except Exception:
            logger.exception("Retrieval failed.")
            raise

        self.last_retrieval_time_ms = (perf_counter() - start) * 1000
        self.last_total_results = len(results)
        # Production note: reordering and deduplication happen after retrieval to keep
        # the final context diverse and more robust for downstream prompting.
        selected_results = self._apply_lost_middle_reordering(results[: self._k])
        # FEATURE: Retrieval deduplication
        final_results = self._deduplicate_results(selected_results)
        final_results = [
            RetrievalResult(
                rank=rank,
                document=result.document,
                distance=result.distance,
                retrieval_strategy=result.retrieval_strategy,
                rerank_score=result.rerank_score,
            )
            for rank, result in enumerate(final_results, start=1)
        ]
        self.last_returned_results = len(final_results)
        logger.info(
            "Retrieved %d/%d chunks in %.2f ms.",
            self.last_returned_results,
            self.last_total_results,
            self.last_retrieval_time_ms,
        )
        return final_results

    # Dispatch retrieval to similarity, MMR, or hybrid search based on configuration.
    def _retrieve_by_strategy(
        self,
        query: str,
        metadata_filter: dict[str, Any] | None,
    ) -> list[RetrievalResult]:
        strategy = settings.retrieval_strategy.lower()

        if strategy == "similarity":
            return self._similarity_search(query=query, metadata_filter=metadata_filter)
        if strategy == "mmr":
            return self._mmr_search(query=query, metadata_filter=metadata_filter)
        if strategy == "hybrid":
            return self._hybrid_search(query=query, metadata_filter=metadata_filter)

        logger.warning("Unsupported retrieval strategy '%s'. Falling back to similarity.", strategy)
        return self._similarity_search(query=query, metadata_filter=metadata_filter)

    # Retrieve dense vector candidates using the configured vector store.
    def _similarity_search(
        self,
        query: str,
        metadata_filter: dict[str, Any] | None,
    ) -> list[RetrievalResult]:
        dense_results = self._vector_store.similarity_search_with_score(
            query=query,
            k=settings.retrieval_candidate_k,
            metadata_filter=metadata_filter,
        )
        return [
            RetrievalResult(rank=rank, document=document, distance=distance, retrieval_strategy="similarity")
            for rank, (document, distance) in enumerate(dense_results, start=1)
        ]

    # Retrieve diverse dense candidates using Maximum Marginal Relevance.
    def _mmr_search(
        self,
        query: str,
        metadata_filter: dict[str, Any] | None,
    ) -> list[RetrievalResult]:
        documents = self._vector_store.max_marginal_relevance_search(
            query=query,
            k=settings.retrieval_candidate_k,
            fetch_k=max(settings.retrieval_candidate_k, self._k),
            lambda_mult=settings.retrieval_mmr_lambda_mult,
            metadata_filter=metadata_filter,
        )
        return [
            RetrievalResult(rank=rank, document=document, distance=float(rank), retrieval_strategy="mmr")
            for rank, document in enumerate(documents, start=1)
        ]

    # Combine dense retrieval and BM25 ranking with Reciprocal Rank Fusion.
    def _hybrid_search(
        self,
        query: str,
        metadata_filter: dict[str, Any] | None,
    ) -> list[RetrievalResult]:
        dense_results = self._similarity_search(query=query, metadata_filter=metadata_filter)
        bm25_results = self._bm25_rank(query=query, results=dense_results)
        fused_results = self._reciprocal_rank_fusion(
            ranked_lists=[dense_results, bm25_results],
            strategy="hybrid",
        )
        logger.info(
            "Hybrid retrieval fused %d dense and %d BM25 candidates.",
            len(dense_results),
            len(bm25_results),
        )
        return fused_results

    # Rank dense candidates lexically with a local BM25 scorer.
    def _bm25_rank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        if not results:
            return []

        tokenized_docs = [self._tokenize(result.document.page_content) for result in results]
        query_tokens = self._tokenize(query)
        doc_count = len(tokenized_docs)
        avg_doc_len = sum(len(tokens) for tokens in tokenized_docs) / max(doc_count, 1)
        doc_freq = Counter(token for tokens in tokenized_docs for token in set(tokens))
        k1 = 1.5
        b = 0.75

        scored_results: list[tuple[RetrievalResult, float]] = []
        for result, tokens in zip(results, tokenized_docs):
            term_counts = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                if token not in term_counts:
                    continue
                idf = math.log(1 + (doc_count - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5))
                tf = term_counts[token]
                denominator = tf + k1 * (1 - b + b * len(tokens) / max(avg_doc_len, 1))
                score += idf * (tf * (k1 + 1)) / denominator
            scored_results.append((result, score))

        scored_results.sort(key=lambda item: item[1], reverse=True)
        return [
            RetrievalResult(
                rank=rank,
                document=result.document,
                distance=-score,
                retrieval_strategy="bm25",
            )
            for rank, (result, score) in enumerate(scored_results, start=1)
        ]

    # Fuse multiple ranked retrieval lists into one ranking with RRF.
    def _reciprocal_rank_fusion(
        self,
        ranked_lists: list[list[RetrievalResult]],
        strategy: str,
    ) -> list[RetrievalResult]:
        scores: dict[str, float] = {}
        documents: dict[str, Document] = {}
        distances: dict[str, float] = {}

        for ranked_list in ranked_lists:
            for rank, result in enumerate(ranked_list, start=1):
                key = self._result_key(result)
                scores[key] = scores.get(key, 0.0) + 1.0 / (settings.retrieval_rrf_k + rank)
                documents[key] = result.document
                distances[key] = min(distances.get(key, result.distance), result.distance)

        ranked_keys = sorted(scores, key=scores.get, reverse=True)
        return [
            RetrievalResult(
                rank=rank,
                document=documents[key],
                distance=distances[key],
                retrieval_strategy=strategy,
            )
            for rank, key in enumerate(ranked_keys, start=1)
        ]

    # Rerank retrieval candidates using a cross-encoder when enabled.
    def _rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        if not settings.retrieval_enable_reranking or not results:
            return results

        start = perf_counter()
        candidate_results = results[: settings.retrieval_candidate_k]
        batch_size = max(settings.reranker_batch_size, 1)

        try:
            reranker = self._get_reranker()
            scores: list[float] = []
            pairs = [(query, result.document.page_content) for result in candidate_results]
            for batch_start in range(0, len(pairs), batch_size):
                batch_scores = reranker.predict(pairs[batch_start : batch_start + batch_size])
                scores.extend(float(score) for score in batch_scores)
        except Exception:
            logger.exception("Cross-encoder reranking failed. Returning original retrieval order.")
            return results

        reranked: list[RetrievalResult] = []
        for result, score in zip(candidate_results, scores):
            reranked.append(
                RetrievalResult(
                    rank=result.rank,
                    document=result.document,
                    distance=result.distance,
                    retrieval_strategy=f"{result.retrieval_strategy}+rerank",
                    rerank_score=float(score),
                )
            )

        reranked.sort(key=lambda result: result.rerank_score if result.rerank_score is not None else float("-inf"), reverse=True)
        elapsed_ms = (perf_counter() - start) * 1000
        logger.info(
            "Reranked %d candidates with %s in %.2f ms using batch_size=%d.",
            len(reranked),
            self._reranker_model_name or settings.reranker_model,
            elapsed_ms,
            batch_size,
        )
        return reranked + results[len(candidate_results) :]

    # Lazily load the cross-encoder reranker to avoid startup overhead.
    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            model_names = [settings.reranker_model]
            if settings.reranker_fallback_model:
                model_names.append(settings.reranker_fallback_model)

            last_error: Exception | None = None
            for model_name in model_names:
                try:
                    self._reranker = CrossEncoder(model_name)
                    self._reranker_model_name = model_name
                    if settings.reranker_enable_warmup:
                        self._reranker.predict([("warmup", "warmup")])
                    logger.info("Loaded cross-encoder reranker model: %s", model_name)
                    break
                except Exception as exc:
                    last_error = exc
                    logger.warning("Failed to load reranker model '%s'.", model_name, exc_info=True)

            if self._reranker is None and last_error is not None:
                raise last_error
        return self._reranker

    # Reorder final chunks to reduce lost-in-the-middle attention loss.
    def _apply_lost_middle_reordering(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        if not settings.retrieval_enable_lost_middle_reordering or len(results) <= 2:
            return results

        front: list[RetrievalResult] = []
        back: list[RetrievalResult] = []
        for index, result in enumerate(results):
            if index % 2 == 0:
                front.append(result)
            else:
                back.append(result)

        reordered = front + list(reversed(back))
        return [
            RetrievalResult(
                rank=rank,
                document=result.document,
                distance=result.distance,
                retrieval_strategy=result.retrieval_strategy,
                rerank_score=result.rerank_score,
            )
            for rank, result in enumerate(reordered, start=1)
        ]

    def _deduplicate_results(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        seen_keys: set[tuple[str, str, str]] = set()
        deduped: list[RetrievalResult] = []
        for result in results:
            chunk_key = (
                result.source,
                str(result.page),
                result.document.page_content,
            )
            if chunk_key in seen_keys:
                continue
            seen_keys.add(chunk_key)
            deduped.append(result)
        return deduped

    # Build a stable key for deduplicating and fusing retrieval results.
    def _result_key(self, result: RetrievalResult) -> str:
        return str(result.chunk_id) if result.chunk_id != "Unknown" else result.document.page_content

    # Tokenize text for lightweight BM25 scoring.
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())
