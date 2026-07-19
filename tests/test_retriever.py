from typing import Any

from langchain_core.documents import Document

from src.config.settings import settings
from src.retrievers.retriever import Retriever


class FakeVectorStore:
    def similarity_search_with_score(
        self,
        query: str,
        k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        del query, k, metadata_filter
        return [
            (
                Document(
                    page_content="The policy starts in 2024.",
                    metadata={"source": "policy.pdf", "page": 1},
                ),
                0.1,
            ),
            (
                Document(
                    page_content="Contractors must enroll before June.",
                    metadata={"source": "guide.pdf", "page": 2},
                ),
                0.2,
            ),
        ]

    def max_marginal_relevance_search(self, **_: Any) -> list[Document]:
        return [
            Document(
                page_content="Diverse policy context.",
                metadata={"source": "policy.pdf", "page": 1},
            )
        ]


def test_similarity_retrieval_returns_ranked_results(monkeypatch) -> None:
    monkeypatch.setattr(settings, "retrieval_strategy", "similarity")
    monkeypatch.setattr(settings, "retrieval_enable_reranking", False)
    retriever = Retriever(FakeVectorStore(), k=2)

    results = retriever.retrieve("When does the policy start?")

    assert [result.rank for result in results] == [1, 2]
    assert results[0].source == "policy.pdf"


def test_mmr_retrieval_uses_vector_store_mmr(monkeypatch) -> None:
    monkeypatch.setattr(settings, "retrieval_strategy", "mmr")
    monkeypatch.setattr(settings, "retrieval_enable_reranking", False)
    retriever = Retriever(FakeVectorStore(), k=1)

    results = retriever.retrieve("Find diverse context")

    assert len(results) == 1
    assert results[0].retrieval_strategy == "mmr"
