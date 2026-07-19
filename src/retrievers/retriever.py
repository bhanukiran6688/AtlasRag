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
from src.retrievers.bm25_index import BM25Index

# Import NLTK for improved tokenization
try:
    import nltk
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


logger = get_logger(__name__)


@dataclass(slots=True)
class RetrievalResult:
    """
    Represents a retrieved document together with retrieval metadata.

    Attributes:
        rank: The position of this result in the retrieval list (1-based)
        document: The retrieved document with content and metadata
        distance: The raw retrieval score (semantics vary by strategy)
        retrieval_strategy: The strategy used to retrieve this document
        rerank_score: Optional cross-encoder reranking score (higher is better)
        normalized_score: Normalized score in 0-1 range (1.0 = best match)
                          This provides consistent score interpretation across strategies
    """

    rank: int
    document: Document
    distance: float
    retrieval_strategy: str = "similarity"
    rerank_score: float | None = None
    normalized_score: float | None = None

    @property
    def source(self) -> str:
        return self.document.metadata.get("source", "Unknown")

    @property
    def page(self) -> int | None:
        return self.document.metadata.get("page")

    @property
    def chunk_id(self) -> str:
        return self.document.metadata.get("chunk_id") or self.document.metadata.get(
            "chunk_index", "Unknown"
        )

    @property
    def chunk_length(self) -> int:
        return len(self.document.page_content)


class Retriever:
    """
    Retrieves relevant chunks using configurable production retrieval strategies.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        k: int | None = None,
        score_threshold: float = 0.8,
    ) -> None:
        self._vector_store = vector_store
        self._k = k or settings.retrieval_top_k
        self._score_threshold = score_threshold
        self._reranker = None
        self._reranker_model_name: str | None = None
        self.last_retrieval_time_ms: float = 0.0
        self.last_total_results: int = 0
        self.last_returned_results: int = 0

        # RAG Concept: Full Corpus BM25 Index
        # Initialize persistent BM25 index for corpus-level lexical search
        # This enables true hybrid search where BM25 searches the entire corpus,
        # not just the dense vector candidates
        self._bm25_index = BM25Index()

        # Initialize improved tokenization with NLTK if available
        # NLTK provides stemming (reducing words to root forms) and stopword filtering
        # This improves BM25 retrieval quality by normalizing word variations
        # and removing common words that don't add semantic value
        self._stemmer = None
        self._stop_words = set()
        if NLTK_AVAILABLE:
            try:
                # Download required NLTK data if not present
                nltk.download("punkt", quiet=True)
                nltk.download("stopwords", quiet=True)
                self._stemmer = PorterStemmer()
                self._stop_words = set(stopwords.words("english"))
                logger.info(
                    "NLTK tokenization enabled with stemming and stopword filtering"
                )
            except Exception as exc:
                logger.warning("Failed to initialize NLTK tokenization: %s", exc)

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
            results = self._retrieve_by_strategy(
                query=query, metadata_filter=metadata_filter
            )
            results = self._rerank(query=query, results=results)
        except Exception:
            logger.exception("Retrieval failed.")
            raise

        self.last_retrieval_time_ms = (perf_counter() - start) * 1000
        self.last_total_results = len(results)

        # FEATURE: Parent-child retrieval - if enabled, replace child chunks with parent chunks
        if settings.enable_parent_child_retrieval:
            results = self._expand_to_parent_chunks(results)

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
                normalized_score=result.normalized_score,
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

        logger.warning(
            "Unsupported retrieval strategy '%s'. Falling back to similarity.", strategy
        )
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
        results = [
            RetrievalResult(
                rank=rank,
                document=document,
                distance=distance,
                retrieval_strategy="similarity",
            )
            for rank, (document, distance) in enumerate(dense_results, start=1)
        ]
        return self._normalize_scores(results, strategy="similarity")

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
        results = [
            RetrievalResult(
                rank=rank,
                document=document,
                distance=float(rank),
                retrieval_strategy="mmr",
            )
            for rank, document in enumerate(documents, start=1)
        ]
        return self._normalize_scores(results, strategy="mmr")

    # Combine dense retrieval and full corpus BM25 search with Reciprocal Rank Fusion.
    # RAG Concept: True Hybrid Search
    # - Dense search captures semantic meaning
    # - BM25 search over full corpus captures exact keyword matches
    # - RRF fuses both rankings for best results
    def _hybrid_search(
        self,
        query: str,
        metadata_filter: dict[str, Any] | None,
    ) -> list[RetrievalResult]:
        # Get dense vector results (semantic search)
        dense_results = self._similarity_search(
            query=query, metadata_filter=metadata_filter
        )

        # Get BM25 results from full corpus (lexical search)
        # RAG Concept: Full Corpus BM25
        # - Unlike candidate-level BM25 which only ranks dense results,
        # - Full corpus BM25 searches ALL documents independently
        # - This preserves lexical-only matches that dense search might miss
        bm25_ranked = self._bm25_index.search(
            query=query, k=settings.retrieval_candidate_k
        )

        # Convert BM25 results to RetrievalResult format
        # Need to fetch actual documents from vector store using doc_ids
        bm25_results = self._bm25_results_to_retrieval_results(
            bm25_ranked, metadata_filter
        )

        # Fuse dense and BM25 results using Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            ranked_lists=[dense_results, bm25_results],
            strategy="hybrid",
        )

        logger.info(
            "Full corpus hybrid retrieval fused %d dense and %d BM25 candidates.",
            len(dense_results),
            len(bm25_results),
        )

        return self._normalize_scores(fused_results, strategy="hybrid")

    # Convert BM25 ranked results to RetrievalResult format by fetching documents from vector store
    # RAG Concept: Bridging Lexical and Vector Search
    # - BM25 returns (doc_id, score) tuples
    # - Need to fetch actual document content from vector store
    # - This bridges the gap between lexical index and vector store
    def _bm25_results_to_retrieval_results(
        self,
        bm25_ranked: list[tuple[str, float]],
        metadata_filter: dict[str, Any] | None,
    ) -> list[RetrievalResult]:
        """
        Convert BM25 search results to RetrievalResult objects.

        Args:
            bm25_ranked: List of (doc_id, bm25_score) tuples from BM25 search
            metadata_filter: Optional metadata filter to apply

        Returns:
            List of RetrievalResult objects with documents fetched from vector store
        """
        if not bm25_ranked:
            return []

        results = []

        for rank, (doc_id, bm25_score) in enumerate(bm25_ranked, start=1):
            try:
                # Fetch document from vector store using doc_id
                # Note: This requires vector store to support lookup by doc_id
                # If not supported, we may need to store document content in BM25 index
                docs = self._vector_store.similarity_search_with_score(
                    query="",  # Empty query since we're filtering by doc_id
                    k=1,
                    metadata_filter={"chunk_id": doc_id} if doc_id else None,
                )

                if docs:
                    document, distance = docs[0]
                    # Use negative BM25 score as distance (higher BM25 = better = lower distance)
                    results.append(
                        RetrievalResult(
                            rank=rank,
                            document=document,
                            distance=-bm25_score,  # Negative because BM25 is higher-is-better
                            retrieval_strategy="bm25_full_corpus",
                        )
                    )
                else:
                    logger.warning("Could not find document %s in vector store", doc_id)

            except Exception as exc:
                logger.warning(
                    "Failed to fetch document %s from vector store: %s", doc_id, exc
                )

        return results

    # Rank dense candidates lexically with a local BM25 scorer.
    def _bm25_rank(
        self, query: str, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        if not results:
            return []

        tokenized_docs = [
            self._tokenize(result.document.page_content) for result in results
        ]
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
                idf = math.log(
                    1 + (doc_count - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5)
                )
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
                scores[key] = scores.get(key, 0.0) + 1.0 / (
                    settings.retrieval_rrf_k + rank
                )
                documents[key] = result.document
                distances[key] = min(
                    distances.get(key, result.distance), result.distance
                )

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
    def _rerank(
        self, query: str, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        if not settings.retrieval_enable_reranking or not results:
            return results

        start = perf_counter()
        candidate_results = results[: settings.retrieval_candidate_k]
        batch_size = max(settings.reranker_batch_size, 1)

        try:
            reranker = self._get_reranker()
            scores: list[float] = []
            pairs = [
                (query, result.document.page_content) for result in candidate_results
            ]
            for batch_start in range(0, len(pairs), batch_size):
                batch_scores = reranker.predict(
                    pairs[batch_start : batch_start + batch_size]
                )
                scores.extend(float(score) for score in batch_scores)
        except Exception:
            logger.exception(
                "Cross-encoder reranking failed. Returning original retrieval order."
            )
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

        reranked.sort(
            key=lambda result: (
                result.rerank_score
                if result.rerank_score is not None
                else float("-inf")
            ),
            reverse=True,
        )
        elapsed_ms = (perf_counter() - start) * 1000
        logger.info(
            "Reranked %d candidates with %s in %.2f ms using batch_size=%d.",
            len(reranked),
            self._reranker_model_name or settings.reranker_model,
            elapsed_ms,
            batch_size,
        )
        return reranked + results[len(candidate_results) :]

    # Expand child chunks to their parent chunks for better context.
    def _expand_to_parent_chunks(
        self, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """
        Replace child chunks with their parent chunks when parent-child retrieval is enabled.
        This provides larger context chunks for answer generation while maintaining
        precise search through smaller child chunks.
        """
        parent_chunks_map = {}
        child_results = []

        for result in results:
            is_parent = result.document.metadata.get("is_parent", False)
            parent_id = result.document.metadata.get("parent_id")

            if is_parent:
                # This is already a parent chunk, keep it
                if parent_id not in parent_chunks_map:
                    parent_chunks_map[parent_id] = result
            else:
                # This is a child chunk, track it for parent lookup
                child_results.append((parent_id, result))

        # For each child result, try to find its parent chunk
        expanded_results = []
        seen_parent_ids = set()

        # First, add any parent chunks that were directly retrieved
        for parent_id, parent_result in parent_chunks_map.items():
            expanded_results.append(parent_result)
            seen_parent_ids.add(parent_id)

        # Then, for child chunks, fetch their parents from the vector store
        for parent_id, child_result in child_results:
            if parent_id and parent_id not in seen_parent_ids:
                # Try to find the parent chunk in the vector store
                try:
                    parent_filter = {"parent_id": parent_id, "is_parent": True}
                    parent_docs = self._vector_store.similarity_search_with_score(
                        query="",  # Empty query since we're filtering by metadata
                        k=1,
                        metadata_filter=parent_filter,
                    )

                    if parent_docs:
                        parent_doc, parent_score = parent_docs[0]
                        parent_result = RetrievalResult(
                            rank=child_result.rank,
                            document=parent_doc,
                            distance=child_result.distance,  # Keep child's distance score
                            retrieval_strategy="parent_child",
                            rerank_score=child_result.rerank_score,
                        )
                        expanded_results.append(parent_result)
                        seen_parent_ids.add(parent_id)
                        logger.debug("Expanded child chunk to parent: %s", parent_id)
                    else:
                        # Parent not found, keep the child chunk
                        expanded_results.append(child_result)
                except Exception as exc:
                    logger.warning(
                        "Failed to fetch parent chunk for %s: %s", parent_id, exc
                    )
                    expanded_results.append(child_result)
            else:
                # No parent_id or already seen, keep the child
                expanded_results.append(child_result)

        logger.info(
            "Expanded %d results to %d chunks using parent-child retrieval",
            len(results),
            len(expanded_results),
        )
        return expanded_results

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
                    logger.warning(
                        "Failed to load reranker model '%s'.", model_name, exc_info=True
                    )

            if self._reranker is None and last_error is not None:
                raise last_error
        return self._reranker

    # Reorder final chunks to reduce lost-in-the-middle attention loss.
    def _apply_lost_middle_reordering(
        self, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
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

    def _deduplicate_results(
        self, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
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
        return (
            str(result.chunk_id)
            if result.chunk_id != "Unknown"
            else result.document.page_content
        )

    # Tokenize text for lightweight BM25 scoring with improved NLP when available.
    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text with stemming and stopword filtering when NLTK is available.

        This method performs text tokenization for BM25 scoring with the following improvements:
        - Basic tokenization: extracts alphanumeric tokens, converts to lowercase
        - Stemming (if NLTK available): reduces words to root forms (e.g., "running" -> "run")
        - Stopword filtering (if NLTK available): removes common words like "the", "a", "is"
        - Length filtering: removes very short tokens (< 3 chars) that are likely noise

        These improvements help BM25 match documents based on semantic meaning rather
        than exact word matches, improving retrieval quality for variations of the same concept.
        """
        # Basic tokenization: extract alphanumeric tokens and convert to lowercase
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())

        # Apply stemming and stopword filtering if NLTK is available
        if NLTK_AVAILABLE and self._stemmer:
            tokens = [
                self._stemmer.stem(token)
                for token in tokens
                if token not in self._stop_words and len(token) > 2
            ]

        return tokens

    # Normalize scores across different retrieval strategies to 0-1 range.
    def _normalize_scores(
        self, results: list[RetrievalResult], strategy: str
    ) -> list[RetrievalResult]:
        """Normalize retrieval scores to 0-1 range for consistent comparison.

        Different retrieval strategies use different score semantics:
        - Similarity: lower distance is better (cosine distance)
        - MMR: lower rank is better (1 = best)
        - Hybrid: higher RRF score is better
        - BM25: higher score is better

        This method normalizes all scores to a 0-1 range where 1.0 represents the best match.
        This makes scores comparable across strategies and easier to interpret for users.

        The normalized_score field is added to each RetrievalResult, while the original
        distance field is preserved for debugging and strategy-specific logic.
        """
        if not results:
            return results

        normalized_results = []

        if strategy == "similarity":
            # Similarity distance: lower is better (cosine distance)
            # Convert to 0-1 where 1 is best match
            distances = [r.distance for r in results if r.distance is not None]
            if distances:
                min_dist, max_dist = min(distances), max(distances)
                if max_dist > min_dist:
                    for result in results:
                        if result.distance is not None:
                            normalized = 1.0 - (result.distance - min_dist) / (
                                max_dist - min_dist
                            )
                            result.normalized_score = round(normalized, 4)
                        else:
                            result.normalized_score = 0.0
                        normalized_results.append(result)
                else:
                    for result in results:
                        result.normalized_score = (
                            1.0 if result.distance is not None else 0.0
                        )
                        normalized_results.append(result)
            else:
                for result in results:
                    result.normalized_score = 0.0
                    normalized_results.append(result)

        elif strategy == "mmr":
            # MMR uses rank as distance: lower rank is better
            # Convert to 0-1 where 1 is best match
            max_rank = max((r.distance for r in results), default=1)
            for result in results:
                normalized = 1.0 - (result.distance - 1) / max(1, max_rank - 1)
                result.normalized_score = round(normalized, 4)
                normalized_results.append(result)

        elif strategy == "hybrid":
            # Hybrid uses RRF scores: higher is better
            # Already in reasonable range, just normalize to 0-1
            rrf_scores = [r.distance for r in results if r.distance is not None]
            if rrf_scores:
                min_score, max_score = min(rrf_scores), max(rrf_scores)
                if max_score > min_score:
                    for result in results:
                        if result.distance is not None:
                            normalized = (result.distance - min_score) / (
                                max_score - min_score
                            )
                            result.normalized_score = round(normalized, 4)
                        else:
                            result.normalized_score = 0.0
                        normalized_results.append(result)
                else:
                    for result in results:
                        result.normalized_score = (
                            1.0 if result.distance is not None else 0.0
                        )
                        normalized_results.append(result)
            else:
                for result in results:
                    result.normalized_score = 0.0
                    normalized_results.append(result)

        elif strategy == "bm25":
            # BM25 scores: higher is better
            # Already in reasonable range, just normalize to 0-1
            bm25_scores = [abs(r.distance) for r in results if r.distance is not None]
            if bm25_scores:
                min_score, max_score = min(bm25_scores), max(bm25_scores)
                if max_score > min_score:
                    for result in results:
                        if result.distance is not None:
                            normalized = (abs(result.distance) - min_score) / (
                                max_score - min_score
                            )
                            result.normalized_score = round(normalized, 4)
                        else:
                            result.normalized_score = 0.0
                        normalized_results.append(result)
                else:
                    for result in results:
                        result.normalized_score = (
                            1.0 if result.distance is not None else 0.0
                        )
                        normalized_results.append(result)
            else:
                for result in results:
                    result.normalized_score = 0.0
                    normalized_results.append(result)
        else:
            # Unknown strategy, just set to 0
            for result in results:
                result.normalized_score = 0.0
                normalized_results.append(result)

        return normalized_results
