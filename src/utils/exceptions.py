class RAGConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


class IngestionError(RuntimeError):
    """Raised when document ingestion or indexing fails."""


class RetrievalError(RuntimeError):
    """Raised when retrieval fails due to upstream or runtime issues."""


class VectorStoreError(RetrievalError):
    """Raised when vector store operations fail."""


class GatewayError(RuntimeError):
    """Raised when the LLM gateway cannot complete a request."""
