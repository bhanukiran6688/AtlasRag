from typing import Any

from src.config.settings import settings
from src.utils.exceptions import VectorStoreError
from src.vectorstores.base import VectorStore


class PineconeStore(VectorStore):

    # Initialize the Pinecone vector store wrapper with configured embeddings.
    def __init__(self, embeddings):
        # Lazy import to reduce startup time.
        from langchain_pinecone import PineconeVectorStore

        self._index = PineconeVectorStore(
            index_name=settings.pinecone_index,
            embedding=embeddings,
            pinecone_api_key=settings.pinecone_api_key,
        )

    @property
    def provider(self) -> str:
        return "Pinecone"

    @property
    def name(self) -> str:
        return settings.pinecone_index

    @property
    def location(self) -> str:
        return "Pinecone Cloud"

    # Validate that the Pinecone vector store object was initialized.
    def validate_connection(self) -> None:
        if getattr(self, "_index", None) is None:
            raise VectorStoreError("Pinecone vector store is not initialized.")

    # Delete indexed Pinecone chunks during document re-indexing.
    def delete_documents(self, document_ids: list[str]) -> None:
        if not document_ids:
            return
        try:
            self._index.delete(ids=document_ids)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete Pinecone documents: {type(exc).__name__}"
            ) from exc

    # Add documents to Pinecone using stable chunk IDs when available.
    def add_documents(self, documents):
        ids = [document.metadata.get("chunk_id") for document in documents]
        try:
            if all(ids):
                self._index.add_documents(documents, ids=ids)
                return

            self._index.add_documents(documents)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to add documents to Pinecone: {type(exc).__name__}"
            ) from exc

    # Run dense similarity search in Pinecone with optional metadata filters.
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 6,
        metadata_filter: dict[str, Any] | None = None,
    ):
        try:
            return self._index.similarity_search_with_score(
                query=query,
                k=k,
                filter=metadata_filter,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Pinecone similarity search failed: {type(exc).__name__}"
            ) from exc

    # Run Pinecone MMR retrieval for diverse context selection.
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 6,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        metadata_filter: dict[str, Any] | None = None,
    ):
        try:
            return self._index.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=metadata_filter,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Pinecone MMR search failed: {type(exc).__name__}"
            ) from exc
