import pytest
from unittest.mock import Mock, patch
from src.embeddings.embedding_generator import EmbeddingGenerator, HuggingFaceEmbeddingAdapter


class TestHuggingFaceEmbeddingAdapter:
    def test_embed_documents_returns_list_of_vectors(self, monkeypatch):
        mock_model = Mock()
        mock_model.encode.return_value = Mock(
            tolist=Mock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        )
        
        monkeypatch.setattr("src.embeddings.embedding_generator.SentenceTransformer", Mock(return_value=mock_model))
        
        adapter = HuggingFaceEmbeddingAdapter("test-model")
        result = adapter.embed_documents(["text1", "text2"])
        
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_model.encode.assert_called_once()

    def test_embed_query_returns_vector(self, monkeypatch):
        mock_model = Mock()
        mock_model.encode.return_value = Mock(
            tolist=Mock(return_value=[0.1, 0.2, 0.3])
        )
        
        monkeypatch.setattr("src.embeddings.embedding_generator.SentenceTransformer", Mock(return_value=mock_model))
        
        adapter = HuggingFaceEmbeddingAdapter("test-model")
        result = adapter.embed_query("test query")
        
        assert result == [0.1, 0.2, 0.3]
        mock_model.encode.assert_called_once()


class TestEmbeddingGenerator:
    def test_huggingface_provider_initialization(self, monkeypatch):
        monkeypatch.setattr("src.config.settings.settings", Mock(
            embedding_provider="huggingface",
            embedding_model="test-model",
            validate_embedding_configuration=Mock()
        ))
        
        with patch("src.embeddings.embedding_generator.HuggingFaceEmbeddingAdapter") as mock_adapter:
            mock_adapter.assert_called_once_with("test-model")

    def test_gemini_provider_initialization(self, monkeypatch):
        monkeypatch.setattr("src.config.settings.settings", Mock(
            embedding_provider="gemini",
            embedding_model="test-model",
            validate_embedding_configuration=Mock()
        ))
        
        with patch("src.embeddings.embedding_generator.GoogleGenerativeAIEmbeddings") as mock_embeddings:
            mock_embeddings.assert_called_once_with(model="test-model")

    def test_unsupported_provider_raises_error(self, monkeypatch):
        monkeypatch.setattr("src.config.settings.settings", Mock(
            embedding_provider="unsupported",
            embedding_model="test-model",
            validate_embedding_configuration=Mock()
        ))
        
        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            EmbeddingGenerator()

    def test_get_embeddings_returns_configured_embeddings(self, monkeypatch):
        mock_embeddings = Mock()
        monkeypatch.setattr("src.config.settings.settings", Mock(
            embedding_provider="huggingface",
            embedding_model="test-model",
            validate_embedding_configuration=Mock()
        ))
        
        with patch("src.embeddings.embedding_generator.HuggingFaceEmbeddingAdapter", return_value=mock_embeddings):
            generator = EmbeddingGenerator()
            result = generator.get_embeddings()
            assert result == mock_embeddings
