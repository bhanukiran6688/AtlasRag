import re
from typing import TYPE_CHECKING

from src.config.settings import settings
from src.utils.citations import (
    citation_source_name,
    normalize_citation_page,
    normalize_citation_value,
)

if TYPE_CHECKING:
    from src.retrievers.retriever import RetrievalResult


class ClaimSupportVerifier:
    """Checks that cited context contains meaningful terms from each claim."""

    _stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }

    # RAG feature: reject claims whose cited chunks do not provide enough lexical support.
    def verify(
        self,
        claim_text: str,
        citations: list[dict[str, str]],
        retrieved_chunks: list["RetrievalResult"],
    ) -> bool:
        claim_terms = self._meaningful_terms(claim_text)
        if not claim_terms:
            return True

        supporting_text = " ".join(
            chunk.document.page_content
            for chunk in retrieved_chunks
            if self._matches_any_citation(chunk, citations)
        )
        if not supporting_text:
            return False

        supported_terms = self._meaningful_terms(supporting_text)
        coverage = len(claim_terms & supported_terms) / len(claim_terms)
        return coverage >= settings.guardrails_min_claim_token_coverage

    @classmethod
    def _meaningful_terms(cls, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
            if len(token) > 2 and token not in cls._stop_words
        }

    @staticmethod
    def _matches_any_citation(
        chunk: "RetrievalResult", citations: list[dict[str, str]]
    ) -> bool:
        chunk_page = normalize_citation_page(chunk.page)
        chunk_sources = {
            normalize_citation_value(chunk.source),
            citation_source_name(chunk.source),
        }
        return any(
            normalize_citation_page(citation.get("page")) == chunk_page
            and (
                normalize_citation_value(str(citation.get("source", "")))
                in chunk_sources
                or citation_source_name(str(citation.get("source", "")))
                in chunk_sources
            )
            for citation in citations
        )
