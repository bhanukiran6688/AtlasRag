from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HuggingFaceEmbeddingAdapter:
    """
    Adapter to make SentenceTransformer compatible with LangChain's
    Embeddings interface.
    """

    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    # Embed document chunks into normalized dense vectors for vector search.
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    # Embed user queries into the same vector space as indexed chunks.
    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()


class EmbeddingGenerator:
    """
    Creates and provides the configured embedding model.
    """

    def __init__(self) -> None:
        settings.validate_embedding_configuration()

        provider = settings.embedding_provider.lower()
        model = settings.embedding_model

        if provider == "huggingface":
            self._embeddings = HuggingFaceEmbeddingAdapter(model)

        elif provider == "gemini":
            self._embeddings = GoogleGenerativeAIEmbeddings(model=model)

        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

        logger.info(
            "Initialized %s embedding model: %s",
            provider,
            model,
        )

    def get_embeddings(self):
        """
        Return the configured embedding model.
        """
        return self._embeddings
