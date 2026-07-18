import pytest
from unittest.mock import Mock
from langchain_core.documents import Document
from src.prompts.prompt_builder import ContextBuilder, PromptBuilder
from src.retrievers.retriever import RetrievalResult


class TestContextBuilder:
    def test_build_returns_empty_string_for_empty_results(self):
        result = ContextBuilder.build([])
        assert result == ""

    def test_build_removes_duplicate_chunks(self, monkeypatch):
        monkeypatch.setattr("src.config.settings.settings", Mock(
            max_context_tokens=1000,
            context_compression_max_sentences=3
        ))
        
        chunk_content = "This is test content"
        results = [
            RetrievalResult(
                rank=1,
                document=Document(page_content=chunk_content, metadata={"source": "test.pdf", "page": 1}),
                distance=0.1
            ),
            RetrievalResult(
                rank=2,
                document=Document(page_content=chunk_content, metadata={"source": "test.pdf", "page": 2}),
                distance=0.2
            )
        ]
        
        result = ContextBuilder.build(results)
        assert result.count(chunk_content) == 1

    def test_build_includes_document_metadata(self, monkeypatch):
        monkeypatch.setattr("src.config.settings.settings", Mock(
            max_context_tokens=1000,
            context_compression_max_sentences=3
        ))
        
        results = [
            RetrievalResult(
                rank=1,
                document=Document(page_content="Test content", metadata={"source": "test.pdf", "page": 1}),
                distance=0.1
            )
        ]
        
        result = ContextBuilder.build(results)
        assert "Document 1" in result
        assert "Source : test.pdf" in result
        assert "Page   : 1" in result

    def test_count_tokens_estimates_token_count(self):
        text = "This is a test sentence with seven words"
        result = ContextBuilder._count_tokens(text)
        assert result == 7

    def test_truncate_context_returns_full_context_when_under_limit(self, monkeypatch):
        monkeypatch.setattr("src.config.settings.settings", Mock(
            max_context_tokens=1000,
            context_compression_max_sentences=3
        ))
        
        short_context = "Short context"
        result = ContextBuilder._truncate_context(short_context, max_tokens=100)
        assert result == short_context

    def test_truncate_context_truncates_when_over_limit(self, monkeypatch):
        monkeypatch.setattr("src.config.settings.settings", Mock(
            max_context_tokens=1000,
            context_compression_max_sentences=3
        ))
        
        long_context = "word " * 100
        result = ContextBuilder._truncate_context(long_context, max_tokens=10)
        assert len(result.split()) <= 15  # Allow some margin

    def test_compress_context_returns_empty_for_empty_input(self, monkeypatch):
        monkeypatch.setattr("src.config.settings.settings", Mock(
            max_context_tokens=1000,
            context_compression_max_sentences=3
        ))
        
        result = ContextBuilder._compress_context("")
        assert result == ""

    def test_trim_context_to_chunks_returns_full_context_when_under_limit(self, monkeypatch):
        monkeypatch.setattr("src.config.settings.settings", Mock(
            max_context_tokens=1000,
            context_compression_max_sentences=3
        ))
        
        context = "chunk1\n\nchunk2"
        result = ContextBuilder._trim_context_to_chunks(context, max_context_chunks=5)
        assert result == context

    def test_trim_context_to_chunks_truncates_when_over_limit(self, monkeypatch):
        monkeypatch.setattr("src.config.settings.settings", Mock(
            max_context_tokens=1000,
            context_compression_max_sentences=3
        ))
        
        context = "chunk1\n\nchunk2\n\nchunk3\n\nchunk4"
        result = ContextBuilder._trim_context_to_chunks(context, max_context_chunks=2)
        assert "chunk3" not in result
        assert "chunk4" not in result


class TestPromptBuilder:
    def test_build_raises_error_for_empty_question(self, monkeypatch):
        monkeypatch.setattr("src.config.settings.settings", Mock(
            max_context_tokens=1000,
            context_compression_max_sentences=3,
            cost_optimization_max_context_chunks=4
        ))
        
        with pytest.raises(ValueError, match="Question cannot be empty"):
            PromptBuilder.build("", "some context")

    def test_build_raises_error_for_empty_context(self, monkeypatch):
        monkeypatch.setattr("src.config.settings.settings", Mock(
            max_context_tokens=1000,
            context_compression_max_sentences=3,
            cost_optimization_max_context_chunks=4
        ))
        
        with pytest.raises(ValueError, match="Context cannot be empty"):
            PromptBuilder.build("test question", "")

    def test_build_includes_conversation_history(self, monkeypatch):
        monkeypatch.setattr("src.config.settings.settings", Mock(
            max_context_tokens=1000,
            context_compression_max_sentences=3,
            cost_optimization_max_context_chunks=4
        ))
        
        history = [{"role": "user", "content": "previous question"}, {"role": "assistant", "content": "previous answer"}]
        result = PromptBuilder.build("test question", "test context", conversation_history=history)
        
        assert "previous question" in result
        assert "previous answer" in result

    def test_build_handles_empty_conversation_history(self, monkeypatch):
        monkeypatch.setattr("src.config.settings.settings", Mock(
            max_context_tokens=1000,
            context_compression_max_sentences=3,
            cost_optimization_max_context_chunks=4
        ))
        
        result = PromptBuilder.build("test question", "test context", conversation_history=None)
        assert "No prior conversation history" in result

    def test_build_includes_context(self, monkeypatch):
        monkeypatch.setattr("src.config.settings.settings", Mock(
            max_context_tokens=1000,
            context_compression_max_sentences=3,
            cost_optimization_max_context_chunks=4
        ))
        
        result = PromptBuilder.build("test question", "test context")
        assert "test context" in result

    def test_build_includes_question(self, monkeypatch):
        monkeypatch.setattr("src.config.settings.settings", Mock(
            max_context_tokens=1000,
            context_compression_max_sentences=3,
            cost_optimization_max_context_chunks=4
        ))
        
        result = PromptBuilder.build("test question", "test context")
        assert "test question" in result

    def test_build_includes_system_instructions(self, monkeypatch):
        monkeypatch.setattr("src.config.settings.settings", Mock(
            max_context_tokens=1000,
            context_compression_max_sentences=3,
            cost_optimization_max_context_chunks=4
        ))
        
        result = PromptBuilder.build("test question", "test context")
        assert "You are a helpful AI assistant" in result
        assert "Answer the user's question using ONLY the provided context" in result
