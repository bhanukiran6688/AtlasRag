import pytest

from src.config.settings import Settings
from src.utils.exceptions import RAGConfigurationError


def test_validate_startup_configuration_requires_pinecone_api_key() -> None:
    settings = Settings(
        vector_store="pinecone",
        pinecone_api_key="",
        pinecone_index="rag-index",
        embedding_provider="huggingface",
        google_api_key="",
        groq_api_key="",
        cerebras_api_key="",
        llm_primary_model="gemini/gemini-2.5-flash",
    )

    with pytest.raises(RAGConfigurationError, match="pinecone_api_key"):
        settings.validate_startup_configuration()
