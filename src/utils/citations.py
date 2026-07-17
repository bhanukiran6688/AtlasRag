from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.retrievers.retriever import RetrievalResult


def normalize_citation_value(value: str) -> str:
    """Normalize source metadata for citation comparisons."""
    return " ".join(value.split()).lower()


def citation_source_name(value: str) -> str:
    """Return the filename-style citation label from a source path."""
    normalized_path = value.replace("\\", "/")
    return normalized_path.rsplit("/", maxsplit=1)[-1].strip().lower()


def normalize_citation_page(value: object) -> str:
    """Normalize optional page metadata for citation comparisons."""
    return "unknown" if value is None else str(value).strip().lower()


def filter_valid_citations(
    citations: list[dict[str, str]],
    retrieved_chunks: list["RetrievalResult"],
) -> list[dict[str, str]]:
    """Keep citations that match a source and page in the retrieved context."""
    valid_sources: set[tuple[str, str]] = set()
    for result in retrieved_chunks:
        page = normalize_citation_page(result.page)
        valid_sources.add((normalize_citation_value(result.source), page))
        valid_sources.add((citation_source_name(result.source), page))

    return [
        citation
        for citation in citations
        if (
            (normalize_citation_value(str(citation.get("source", ""))), normalize_citation_page(citation.get("page")))
            in valid_sources
            or (citation_source_name(str(citation.get("source", ""))), normalize_citation_page(citation.get("page")))
            in valid_sources
        )
    ]
