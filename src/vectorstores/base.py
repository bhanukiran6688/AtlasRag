from typing import Any
from abc import ABC, abstractmethod
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config.settings import settings


class VectorStore(ABC):
    """
    Abstract base class for vector store implementations.
    """

    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None:
        """
        Add documents to the vector store.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_connection(self) -> None:
        """Validate that the backend is reachable and configured."""
        raise NotImplementedError

    @abstractmethod
    def delete_documents(self, document_ids: list[str]) -> None:
        """Remove previously indexed chunks by their document identifiers."""
        raise NotImplementedError

    @abstractmethod
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 6,
        metadata_filter: dict[str, Any] | None = None,
    ):
        """
        Return retrieved documents together with their distance.
        """
        raise NotImplementedError

    @abstractmethod
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 6,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Return diverse documents using Maximum Marginal Relevance.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable vector store name.
        """

    @property
    @abstractmethod
    def location(self) -> str:
        """
        Human-readable storage location.
        """


class VectorStoreFactory:

    # Create the configured vector store after validating backend settings.
    @staticmethod
    def create(embeddings: Embeddings) -> VectorStore:
        from src.vectorstores.chroma_store import ChromaStore
        from src.vectorstores.pinecone_store import PineconeStore

        settings.validate_vector_store_configuration()

        provider = settings.vector_store.lower()
        if provider == "chroma":
            return ChromaStore(embeddings)
        if provider == "pinecone":
            return PineconeStore(embeddings)

        raise ValueError(f"Unsupported vector store: {provider}")
