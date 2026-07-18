from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class RAGRequest(BaseModel):
    question: str = Field(..., description="The user's question", min_length=1)
    use_query_expansion: bool = Field(default=False, description="Enable query expansion")
    use_query_decomposition: bool = Field(default=False, description="Enable query decomposition")
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata filter for retrieval")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default=None, description="Conversation history for context")


class Citation(BaseModel):
    source: str = Field(..., description="Source document")
    page: str = Field(..., description="Page number")


class Claim(BaseModel):
    text: str = Field(..., description="Claim text")
    citations: List[Citation] = Field(default_factory=list, description="Citations supporting the claim")


class RetrievedChunk(BaseModel):
    rank: int = Field(..., description="Retrieval rank")
    source: str = Field(..., description="Source document")
    page: Optional[int] = Field(default=None, description="Page number")
    distance: float = Field(..., description="Distance/score")
    chunk_length: int = Field(..., description="Chunk length in characters")
    retrieval_strategy: str = Field(..., description="Retrieval strategy used")
    rerank_score: Optional[float] = Field(default=None, description="Rerank score if applicable")
    content: str = Field(..., description="Chunk content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")


class RAGResponse(BaseModel):
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list, description="Retrieved chunks")
    context: str = Field(..., description="Built context")
    llm_model: Optional[str] = Field(default=None, description="LLM model used")
    llm_latency_ms: Optional[float] = Field(default=None, description="LLM latency in milliseconds")
    total_tokens: int = Field(default=0, description="Total tokens used")
    cost_usd: Optional[float] = Field(default=None, description="Cost in USD")
    sanitized_question: Optional[str] = Field(default=None, description="Sanitized question after PII redaction")
    is_blocked: bool = Field(default=False, description="Whether the request was blocked")
    blocked_reason: Optional[str] = Field(default=None, description="Reason for blocking")
    pii_detected: List[str] = Field(default_factory=list, description="PII fields detected")
    cache_hit: bool = Field(default=False, description="Whether the result was cached")
    route: Optional[str] = Field(default=None, description="LLM routing decision")
    structured_output: Optional[Dict[str, Any]] = Field(default=None, description="Structured LLM output")
    error: Optional[str] = Field(default=None, description="Error message if applicable")
    retrieval_queries: Optional[List[str]] = Field(default=None, description="Queries used for retrieval")
    is_grounded: bool = Field(default=True, description="Whether the answer is grounded in context")
    grounding_reason: Optional[str] = Field(default=None, description="Reasoning for grounding decision")
    grounding_confidence: Optional[float] = Field(default=None, description="Grounding confidence score")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(default_factory=dict, description="Component status")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
