from pathlib import Path
from dotenv import load_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.exceptions import RAGConfigurationError

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Centralize all RAG runtime configuration from environment and defaults.
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    chunk_size: int = 800
    chunk_overlap: int = 100
    enable_parent_child_retrieval: bool = False
    log_level: str = "INFO"
    data_dir: Path = PROJECT_ROOT / "data"
    documents_dir: Path = data_dir / "documents"
    chroma_dir: Path = data_dir / "chroma_db"
    vector_store: str = "pinecone"
    embedding_provider: str = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    pinecone_api_key: str = ""
    pinecone_index: str = "rag-index"
    retrieval_filter_source: str = ""
    retrieval_filter_file_type: str = ""
    retrieval_filter_metadata_json: str = ""
    retrieval_allowed_metadata_filter_keys: str = ""
    retrieval_strategy: str = "mmr"
    retrieval_top_k: int = 5
    retrieval_candidate_k: int = 20
    retrieval_mmr_lambda_mult: float = 0.5
    retrieval_rrf_k: int = 60
    retrieval_enable_lost_middle_reordering: bool = True
    retrieval_enable_reranking: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_fallback_model: str = ""
    reranker_batch_size: int = 16
    reranker_enable_warmup: bool = False
    query_expansion_max_variants: int = 2
    query_decomposition_max_subquestions: int = 2
    query_planning_adaptive_enabled: bool = True
    query_planning_low_confidence_threshold: float = 0.45
    query_planning_min_words_for_expansion: int = 8
    query_planning_min_words_for_decomposition: int = 12
    max_context_tokens: int = 1800
    conversation_memory_max_turns: int = 4
    conversation_memory_max_tokens: int = 2000
    context_compression_max_sentences: int = 3
    cost_optimization_max_context_chunks: int = 4
    cost_optimization_enable_prompt_simplification: bool = True
    cost_optimization_max_query_planning_calls: int = 1
    llm_max_estimated_prompt_tokens: int = 6000
    llm_estimated_cost_per_1k_tokens: float = 0.0
    llm_max_estimated_cost_usd_per_request: float = 0.0
    guardrails_enable_claim_support_check: bool = True
    guardrails_min_claim_token_coverage: float = 0.8

    # LiteLLM gateway configuration. Provider-specific details must stay here
    # and inside the gateway, not in RAGService or Streamlit.
    llm_primary_model: str = "gemini/gemini-2.5-flash"
    llm_fallback_models: str = "groq/llama-3.3-70b-versatile,cerebras/llama3.1-8b"
    llm_code_model: str = "gemini/gemini-2.5-flash"
    llm_summary_model: str = "groq/llama-3.3-70b-versatile"
    llm_general_model: str = "gemini/gemini-2.5-flash"
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 1024
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3
    llm_retry_backoff_seconds: float = 1.0
    llm_enable_cost_tracking: bool = True
    llm_rate_limit_requests: int = 30
    llm_rate_limit_window_seconds: int = 60
    startup_validate_configuration: bool = True
    enable_health_checks: bool = True
    ingestion_retry_attempts: int = 2

    google_api_key: str = ""
    google_api_key2: str = ""
    google_api_key3: str = ""
    groq_api_key: str = ""
    cerebras_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("CEREBRAS_API_KEY", "CEREBRAS"),
    )

    # Collect all configured Gemini keys for gateway key rotation.
    @property
    def gemini_api_keys(self) -> list[str]:
        return [
            key
            for key in (
                self.google_api_key,
                self.google_api_key2,
                self.google_api_key3,
            )
            if key
        ]

    # Parse configured fallback model names for gateway retry chains.
    @property
    def fallback_model_names(self) -> list[str]:
        return [
            model.strip()
            for model in self.llm_fallback_models.split(",")
            if model.strip()
        ]

    # Parse optional metadata filter allow-list for multi-user deployments.
    @property
    def allowed_metadata_filter_keys(self) -> set[str]:
        return {
            key.strip()
            for key in self.retrieval_allowed_metadata_filter_keys.split(",")
            if key.strip()
        }

    # Validate embedding provider and credentials before creating embeddings.
    def validate_embedding_configuration(self) -> None:
        if not self.embedding_model.strip():
            raise RAGConfigurationError("embedding_model must not be empty.")

        provider = self.embedding_provider.lower()
        if provider not in {"huggingface", "gemini"}:
            raise RAGConfigurationError(f"Unsupported embedding provider: {provider}")

        if provider == "gemini" and not self.gemini_api_keys:
            raise RAGConfigurationError(
                "Google API key is required when embedding_provider is 'gemini'."
            )

    # Validate vector store provider and required connection settings.
    def validate_vector_store_configuration(self) -> None:
        provider = self.vector_store.lower()
        if provider not in {"chroma", "pinecone"}:
            raise RAGConfigurationError(f"Unsupported vector store: {provider}")

        if provider == "pinecone":
            if not self.pinecone_api_key:
                raise RAGConfigurationError(
                    "pinecone_api_key is required when vector_store is 'pinecone'."
                )
            if not self.pinecone_index.strip():
                raise RAGConfigurationError(
                    "pinecone_index must not be empty when vector_store is 'pinecone'."
                )

    # Validate LLM gateway model and provider credentials before startup.
    def validate_llm_configuration(self) -> None:
        if not self.llm_primary_model.strip():
            raise RAGConfigurationError("llm_primary_model must not be empty.")

        if (
            not self.gemini_api_keys
            and not self.groq_api_key
            and not self.cerebras_api_key
        ):
            raise RAGConfigurationError(
                "At least one LLM API key must be configured for startup."
            )

    # Run all startup checks required for the RAG application to boot safely.
    def validate_startup_configuration(self) -> None:
        self.validate_embedding_configuration()
        self.validate_vector_store_configuration()
        self.validate_llm_configuration()


settings = Settings()
