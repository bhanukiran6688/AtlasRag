import pytest
from src.prompts.prompt_builder import PromptBuilder, ContextBuilder
from src.retrievers.retriever import RetrievalResult
from langchain_core.documents import Document


def test_context_builder_empty():
    assert ContextBuilder.build([]) == ""


def test_context_builder_basic():
    doc1 = Document(
        page_content="Content 1", metadata={"source": "doc1.txt", "page": 1}
    )
    doc2 = Document(
        page_content="Content 2", metadata={"source": "doc2.txt", "page": 2}
    )

    results = [
        RetrievalResult(rank=1, document=doc1, distance=0.1),
        RetrievalResult(rank=2, document=doc2, distance=0.2),
    ]

    context = ContextBuilder.build(results)
    assert "Content 1" in context
    assert "Content 2" in context
    assert "Source : doc1.txt" in context
    assert "Page   : 1" in context


def test_context_builder_deduplication():
    doc1 = Document(
        page_content="Duplicate content", metadata={"source": "doc1.txt", "page": 1}
    )
    doc2 = Document(
        page_content="Duplicate content", metadata={"source": "doc2.txt", "page": 2}
    )

    results = [
        RetrievalResult(rank=1, document=doc1, distance=0.1),
        RetrievalResult(rank=2, document=doc2, distance=0.2),
    ]

    context = ContextBuilder.build(results)
    assert context.count("Duplicate content") == 1


def test_prompt_builder_basic():
    question = "What is RAG?"
    context = "RAG stands for Retrieval-Augmented Generation."

    prompt = PromptBuilder.build(question, context)

    assert question in prompt
    assert context in prompt
    assert "You are a helpful AI assistant." in prompt


def test_prompt_builder_with_history():
    question = "And how does it work?"
    context = "It works by retrieving relevant documents."
    history = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "RAG is..."},
    ]

    prompt = PromptBuilder.build(question, context, conversation_history=history)

    assert "User: What is RAG?" in prompt
    assert "Assistant: RAG is..." in prompt


def test_prompt_builder_empty_errors():
    with pytest.raises(ValueError, match="Question cannot be empty"):
        PromptBuilder.build("", "context")

    with pytest.raises(ValueError, match="Context cannot be empty"):
        PromptBuilder.build("question", "")
