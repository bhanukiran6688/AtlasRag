"""
BM25 Index for Full Corpus Hybrid Search

This module implements a persistent BM25 (Best Matching 25) lexical index
that operates over the complete document corpus, not just dense vector candidates.

RAG Concept: Hybrid Search
- Dense vector search captures semantic meaning but may miss exact keyword matches
- BM25 lexical search excels at keyword matching but misses semantic relationships
- Full corpus hybrid search combines both by maintaining separate indices and fusing results

BM25 Algorithm:
- Scores documents based on term frequency (TF) and inverse document frequency (IDF)
- Uses document length normalization to penalize very long documents
- Formula: score(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl))
  where k1 controls term saturation and b controls length normalization
"""

import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import math

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class BM25Document:
    """
    Represents a document in the BM25 index with its term frequencies.

    RAG Concept: Document Representation for Lexical Search
    - Unlike vector embeddings which capture semantic meaning, BM25 uses explicit term frequencies
    - Each document is represented by the count of each term it contains
    - This allows precise keyword matching that vector search might miss
    """

    doc_id: str
    term_freqs: Dict[str, int] = field(default_factory=dict)
    doc_length: int = 0
    metadata: dict = field(default_factory=dict)


class BM25Index:
    """
    Persistent BM25 index for full corpus lexical search.

    RAG Concept: Full Corpus Hybrid Search
    - Current implementation: BM25 only ranks dense vector candidates (candidate-level hybrid)
    - This implementation: BM25 indexes ALL documents (corpus-level hybrid)
    - Benefit: Lexical-only matches are preserved even if dense retrieval doesn't include them

    Use Case:
    - User searches for "API endpoint timeout"
    - Vector search might find documents about "API" and "timeout" separately
    - BM25 will find documents containing the exact phrase "API endpoint timeout"
    - RRF fuses both rankings for better results
    """

    def __init__(
        self, k1: float = 1.5, b: float = 0.75, index_path: Path | None = None
    ):
        """
        Initialize BM25 index with tunable parameters.

        Args:
            k1: Term saturation parameter (higher = more emphasis on term frequency)
                 - Typical values: 1.2-2.0
                 - Higher k1 = documents with more term occurrences rank higher
            b: Length normalization parameter (0 = no normalization, 1 = full normalization)
                - Typical values: 0.75
                - Higher b = longer documents penalized more
            index_path: Path to persist the index on disk
        """
        self.k1 = k1
        self.b = b
        self.index_path = index_path or settings.data_dir / "bm25_index.pkl"

        # BM25 index structures
        self.documents: Dict[str, BM25Document] = {}  # doc_id -> BM25Document
        self.doc_freqs: Dict[str, int] = defaultdict(
            int
        )  # term -> number of docs containing term
        self.avg_doc_length: float = 0.0
        self.total_docs: int = 0

        # Load existing index if available
        self._load_index()

        logger.info(
            "BM25 index initialized with k1=%.2f, b=%.2f, docs=%d",
            self.k1,
            self.b,
            self.total_docs,
        )

    def add_document(
        self, doc_id: str, content: str, metadata: dict | None = None
    ) -> None:
        """
        Add a document to the BM25 index.

        RAG Concept: Document Ingestion for Lexical Search
        - Tokenizes document content into terms
        - Computes term frequencies for the document
        - Updates global document frequency statistics
        - Recomputes average document length for normalization

        Args:
            doc_id: Unique document identifier
            content: Document text content
            metadata: Optional document metadata (source, page, etc.)
        """
        if doc_id in self.documents:
            logger.debug("Document %s already indexed, skipping", doc_id)
            return

        # Tokenize content into terms (simple whitespace tokenization)
        # In production, use better tokenization: stemming, stopword removal, etc.
        terms = self._tokenize(content)
        term_freqs = defaultdict(int)

        for term in terms:
            term_freqs[term] += 1

        # Create BM25 document
        doc = BM25Document(
            doc_id=doc_id,
            term_freqs=dict(term_freqs),
            doc_length=len(terms),
            metadata=metadata or {},
        )

        # Update index
        self.documents[doc_id] = doc

        # Update document frequencies for each term
        for term in term_freqs.keys():
            self.doc_freqs[term] += 1

        # Update statistics
        self.total_docs += 1
        self._update_avg_doc_length()

        logger.debug(
            "Added document %s to BM25 index with %d terms", doc_id, len(terms)
        )

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search the BM25 index for relevant documents.

        RAG Concept: Lexical Retrieval
        - Scores documents based on term overlap with query
        - Uses BM25 scoring formula: TF * IDF with length normalization
        - Returns top-k documents by BM25 score

        BM25 Scoring:
        For each query term qi and document D:
        IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5))
        TF component = f(qi,D) * (k1 + 1) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl))
        Score = Σ IDF(qi) * TF component

        Args:
            query: Search query text
            k: Number of top results to return

        Returns:
            List of (doc_id, score) tuples sorted by score (descending)
        """
        if self.total_docs == 0:
            logger.warning("BM25 index is empty, returning no results")
            return []

        # Tokenize query
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        # Compute BM25 scores for each document
        scores = {}

        for doc_id, doc in self.documents.items():
            score = 0.0

            for term in query_terms:
                if term not in doc.term_freqs:
                    continue

                # Compute IDF (Inverse Document Frequency)
                # IDF measures how rare the term is across the corpus
                # Rare terms get higher IDF, common terms get lower IDF
                df = self.doc_freqs[term]
                idf = math.log((self.total_docs - df + 0.5) / (df + 0.5))

                # Compute TF (Term Frequency) component with length normalization
                # TF measures how frequently the term appears in this document
                # Length normalization prevents long documents from dominating
                tf = doc.term_freqs[term]
                length_norm = (
                    1 - self.b + self.b * (doc.doc_length / self.avg_doc_length)
                )
                tf_component = tf * (self.k1 + 1) / (tf + self.k1 * length_norm)

                # Add term's contribution to document score
                score += idf * tf_component

            if score > 0:
                scores[doc_id] = score

        # Sort by score (descending) and return top-k
        ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        logger.info(
            "BM25 search for '%s' returned %d results (query terms: %d)",
            query,
            len(ranked_results),
            len(query_terms),
        )

        return ranked_results

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms for BM25 indexing.

        RAG Concept: Text Tokenization for Lexical Search
        - Simple whitespace tokenization (can be enhanced)
        - Lowercasing for case-insensitive matching
        - In production, add: stemming, stopword removal, n-grams

        Args:
            text: Input text to tokenize

        Returns:
            List of terms
        """
        # Simple tokenization: split on whitespace and lowercase
        terms = text.lower().split()

        # Remove very short terms (likely noise)
        terms = [term for term in terms if len(term) > 2]

        return terms

    def _update_avg_doc_length(self) -> None:
        """
        Update average document length for BM25 normalization.

        RAG Concept: Length Normalization
        - BM25 penalizes very long documents to prevent them from dominating results
        - Uses average document length as reference point
        - Parameter 'b' controls how strongly to normalize
        """
        if self.total_docs == 0:
            self.avg_doc_length = 0.0
            return

        total_length = sum(doc.doc_length for doc in self.documents.values())
        self.avg_doc_length = total_length / self.total_docs

    def _load_index(self) -> None:
        """
        Load BM25 index from disk if it exists.

        RAG Concept: Persistent Index
        - Avoids re-indexing documents on every startup
        - Enables incremental updates to the index
        - Critical for production RAG systems with large document sets
        """
        if self.index_path.exists():
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    self.documents = data.get("documents", {})
                    self.doc_freqs = defaultdict(int, data.get("doc_freqs", {}))
                    self.avg_doc_length = data.get("avg_doc_length", 0.0)
                    self.total_docs = data.get("total_docs", 0)
                logger.info(
                    "Loaded BM25 index from %s with %d documents",
                    self.index_path,
                    self.total_docs,
                )
            except Exception as exc:
                logger.warning("Failed to load BM25 index: %s. Starting fresh.", exc)

    def save_index(self) -> None:
        """
        Save BM25 index to disk.

        RAG Concept: Index Persistence
        - Persists index state for fast startup
        - Enables recovery after crashes
        - Supports incremental indexing workflows
        """
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "documents": self.documents,
                "doc_freqs": dict(self.doc_freqs),
                "avg_doc_length": self.avg_doc_length,
                "total_docs": self.total_docs,
            }

            with open(self.index_path, "wb") as f:
                pickle.dump(data, f)

            logger.info(
                "Saved BM25 index to %s with %d documents",
                self.index_path,
                self.total_docs,
            )
        except Exception as exc:
            logger.error("Failed to save BM25 index: %s", exc)

    def delete_document(self, doc_id: str) -> None:
        """
        Remove a document from the BM25 index.

        RAG Concept: Document Lifecycle Management
        - Supports deletion of outdated documents
        - Maintains index consistency
        - Requires updating document frequencies

        Args:
            doc_id: Document identifier to remove
        """
        if doc_id not in self.documents:
            logger.debug("Document %s not in index, skipping deletion", doc_id)
            return

        doc = self.documents[doc_id]

        # Update document frequencies
        for term in doc.term_freqs.keys():
            self.doc_freqs[term] -= 1
            if self.doc_freqs[term] <= 0:
                del self.doc_freqs[term]

        # Remove document
        del self.documents[doc_id]
        self.total_docs -= 1

        # Recompute average document length
        self._update_avg_doc_length()

        logger.info("Deleted document %s from BM25 index", doc_id)

    def clear(self) -> None:
        """
        Clear all documents from the BM25 index.

        RAG Concept: Index Reset
        - Useful for full re-indexing workflows
        - Clears all index state
        - Requires re-adding all documents
        """
        self.documents.clear()
        self.doc_freqs.clear()
        self.avg_doc_length = 0.0
        self.total_docs = 0
        logger.info("Cleared BM25 index")
