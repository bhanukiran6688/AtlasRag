from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
import uuid

from src.api.models import RAGRequest, RAGResponse, HealthResponse, RetrievedChunk
from src.services.factory import create_rag_service
from src.utils.logger import get_logger
from src.utils.exceptions import RAGConfigurationError, GatewayError, RetrievalError

logger = get_logger(__name__)

# Global RAG service instance
rag_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global rag_service

    # Startup
    logger.info("Starting RAG API service...")
    try:
        rag_service = create_rag_service()
        logger.info("RAG service initialized successfully")
    except Exception as exc:
        logger.error(f"Failed to initialize RAG service: {exc}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down RAG API service...")


# Create FastAPI app
app = FastAPI(
    title="AtlasRAG API",
    description="Retrieval-Augmented Generation API with hybrid search, guardrails, and citation enforcement",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    components = {
        "rag_service": "healthy" if rag_service else "uninitialized",
        "retriever": (
            "healthy" if rag_service and rag_service.retriever else "unavailable"
        ),
    }

    return HealthResponse(
        status=(
            "healthy"
            if all(v == "healthy" for v in components.values())
            else "degraded"
        ),
        version="1.0.0",
        components=components,
    )


@app.post("/query", response_model=RAGResponse)
async def query(request: RAGRequest):
    """
    Process a RAG query with optional query expansion, decomposition, and metadata filtering.

    Args:
        request: RAG request with question and optional parameters

    Returns:
        RAG response with answer, retrieved chunks, and metadata
    """
    if not rag_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service not initialized",
        )

    request_id = str(uuid.uuid4())
    logger.info(f"Processing query request {request_id}: {request.question[:100]}...")

    start_time = time.time()

    try:
        # Process the query through RAG service
        result = rag_service.process(
            question=request.question,
            use_query_expansion=request.use_query_expansion,
            use_query_decomposition=request.use_query_decomposition,
            metadata_filter=request.metadata_filter,
            conversation_history=request.conversation_history,
        )

        # Convert RetrievalResult to RetrievedChunk
        retrieved_chunks = []
        for chunk in result.retrieved_chunks:
            retrieved_chunks.append(
                RetrievedChunk(
                    rank=chunk.rank,
                    source=chunk.source,
                    page=chunk.page,
                    distance=chunk.distance,
                    chunk_length=chunk.chunk_length,
                    retrieval_strategy=chunk.retrieval_strategy,
                    rerank_score=chunk.rerank_score,
                    content=chunk.document.page_content,
                    metadata=chunk.document.metadata,
                )
            )

        # Build response
        response = RAGResponse(
            question=result.question,
            answer=result.answer,
            retrieved_chunks=retrieved_chunks,
            context=result.context,
            llm_model=result.llm_model,
            llm_latency_ms=result.llm_latency_ms,
            total_tokens=result.total_tokens,
            cost_usd=result.cost_usd,
            sanitized_question=result.sanitized_question,
            is_blocked=result.is_blocked,
            blocked_reason=result.blocked_reason,
            pii_detected=result.pii_detected or [],
            cache_hit=result.cache_hit,
            route=result.route,
            structured_output=result.structured_output,
            error=result.error,
            retrieval_queries=result.retrieval_queries,
            is_grounded=result.is_grounded,
            grounding_reason=result.grounding_reason,
            grounding_confidence=result.grounding_confidence,
            request_id=request_id,
        )

        processing_time = time.time() - start_time
        logger.info(f"Query {request_id} completed in {processing_time:.2f}s")

        return response

    except RAGConfigurationError as exc:
        logger.error(f"Configuration error processing query {request_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration error: {str(exc)}",
        )
    except GatewayError as exc:
        logger.error(f"Gateway error processing query {request_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM gateway error: {str(exc)}",
        )
    except RetrievalError as exc:
        logger.error(f"Retrieval error processing query {request_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval error: {str(exc)}",
        )
    except Exception as exc:
        logger.exception(f"Unexpected error processing query {request_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(exc)}",
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AtlasRAG API",
        "version": "1.0.0",
        "description": "Retrieval-Augmented Generation API",
        "endpoints": {"health": "/health", "query": "/query", "docs": "/docs"},
    }
