import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from src.services.rag_service import RAGService, RAGResult
from src.retrievers.retriever import RetrievalResult
from src.llm.base import LLMResponse, LLMUsage
from langchain_core.documents import Document


@pytest.fixture
def mock_dependencies():
    retriever = MagicMock()
    context_builder = MagicMock()
    prompt_builder = MagicMock()
    llm_gateway = MagicMock()
    return retriever, context_builder, prompt_builder, llm_gateway


@pytest.mark.asyncio
async def test_rag_service_aprocess_basic(mock_dependencies):
    retriever, context_builder, prompt_builder, llm_gateway = mock_dependencies
    service = RAGService(retriever, context_builder, prompt_builder, llm_gateway)

    # Setup mocks
    doc = Document(page_content="RAG is cool", metadata={"source": "test.txt", "page": "1"})
    retrieval_result = RetrievalResult(rank=1, document=doc, distance=0.1)
    retriever.retrieve.return_value = [retrieval_result]
    retriever.k = 5

    context_builder.build.return_value = "Context: RAG is cool"
    prompt_builder.build.return_value = "Prompt with Context: RAG is cool"

    async def mock_generate(*args, **kwargs):
        return LLMResponse(
            content='{"answer": "RAG is cool", "citations": [{"source": "test.txt", "page": "1"}], "claims": [{"text": "RAG is cool", "citations": [{"source": "test.txt", "page": "1"}]}]}',
            model="test-model",
            latency_ms=50.0,
            usage=LLMUsage(10, 10, 20, 0.001),
        )

    llm_gateway.generate = AsyncMock(side_effect=mock_generate)

    result = await service.aprocess("What is RAG?")

    assert isinstance(result, RAGResult)
    assert result.answer == "RAG is cool"
    assert result.llm_model == "test-model"
    retriever.retrieve.assert_called()
    llm_gateway.generate.assert_called()


@pytest.mark.asyncio
async def test_rag_service_empty_retrieval(mock_dependencies):
    retriever, context_builder, prompt_builder, llm_gateway = mock_dependencies
    service = RAGService(retriever, context_builder, prompt_builder, llm_gateway)

    retriever.retrieve.return_value = []
    context_builder.build.return_value = ""

    result = await service.aprocess("What is RAG?")

    assert result.answer == ""
    assert result.context == ""
    llm_gateway.generate.assert_not_called()
