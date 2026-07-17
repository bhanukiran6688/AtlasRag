from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from typing import Any

from src.utils.logger import get_logger
from src.config.settings import settings
from src.utils.exceptions import VectorStoreError
from src.vectorstores.base import VectorStore


logger = get_logger(__name__)


class ChromaStore(VectorStore):
    """
    Handles interactions with the Chroma vector database.
    """

    def __init__(self, embeddings: Embeddings, collection_name: str = "documents") -> None:
        # Lazy import to reduce startup time.
        from langchain_chroma import Chroma

        self.collection_name = collection_name
        self._vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(settings.chroma_dir)
        )

        logger.info("Initialized Chroma collection: %s",collection_name)

    @property
    def name(self) -> str:
        return self.collection_name

    @property
    def location(self) -> str:
        return str(settings.chroma_dir)

    # Validate that the Chroma vector store object was initialized.
    def validate_connection(self) -> None:
        if getattr(self, "_vector_store", None) is None:
            raise VectorStoreError("Chroma vector store is not initialized.")

    # Delete indexed Chroma chunks during document re-indexing.
    def delete_documents(self, document_ids: list[str]) -> None:
        if not document_ids:
            return
        try:
            self._vector_store.delete(ids=document_ids)
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete Chroma documents: {type(exc).__name__}") from exc

    def add_documents(self, documents: list[Document]) -> None:
        """
        Add documents to the vector store.
        """

        ids = []
        for index, document in enumerate(documents):
            source = document.metadata.get("source", "unknown")
            page = document.metadata.get("page", 0)
            ids.append(document.metadata.get("chunk_id", f"{source}:{page}:{index}"))

        try:
            self._vector_store.add_documents(documents=documents, ids=ids)
        except Exception as exc:
            raise VectorStoreError(f"Failed to add documents to Chroma: {type(exc).__name__}") from exc
        logger.info(
            "Indexed %d chunks from %s.",
            len(documents),
            documents[0].metadata.get("source", "Unknown")
        )

    # Run dense similarity search in Chroma with optional metadata filters.
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 6,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        try:
            return self._vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=metadata_filter,
            )
        except Exception as exc:
            raise VectorStoreError(f"Chroma similarity search failed: {type(exc).__name__}") from exc

    # Run Chroma MMR retrieval for diverse context selection.
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 6,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        try:
            return self._vector_store.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=metadata_filter,
            )
        except Exception as exc:
            raise VectorStoreError(f"Chroma MMR search failed: {type(exc).__name__}") from exc
