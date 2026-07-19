import pytest
from unittest.mock import Mock
from langchain_core.documents import Document
from src.chunking.text_splitter import DocumentSplitter


class TestDocumentSplitter:
    def test_splitter_raises_error_when_overlap_equals_size(self, monkeypatch):
        monkeypatch.setattr(
            "src.config.settings.settings", Mock(chunk_size=100, chunk_overlap=100)
        )

        with pytest.raises(
            ValueError, match="chunk_overlap must be smaller than chunk_size"
        ):
            DocumentSplitter()

    def test_splitter_raises_error_when_overlap_exceeds_size(self, monkeypatch):
        monkeypatch.setattr(
            "src.config.settings.settings", Mock(chunk_size=100, chunk_overlap=150)
        )

        with pytest.raises(
            ValueError, match="chunk_overlap must be smaller than chunk_size"
        ):
            DocumentSplitter()

    def test_split_documents_returns_empty_list_for_empty_input(self, monkeypatch):
        monkeypatch.setattr(
            "src.config.settings.settings", Mock(chunk_size=100, chunk_overlap=20)
        )

        splitter = DocumentSplitter()
        result = splitter.split_documents([])
        assert result == []

    def test_split_documents_skips_csv_format(self, monkeypatch):
        monkeypatch.setattr(
            "src.config.settings.settings", Mock(chunk_size=100, chunk_overlap=20)
        )

        splitter = DocumentSplitter()
        documents = [Document(page_content="col1,col2\nval1,val2", metadata={})]
        result = splitter.split_documents(documents, file_type="csv")

        assert len(result) == 1
        assert result[0].page_content == "col1,col2\nval1,val2"

    def test_split_documents_skips_json_format(self, monkeypatch):
        monkeypatch.setattr(
            "src.config.settings.settings", Mock(chunk_size=100, chunk_overlap=20)
        )

        splitter = DocumentSplitter()
        documents = [Document(page_content='{"key": "value"}', metadata={})]
        result = splitter.split_documents(documents, file_type="json")

        assert len(result) == 1
        assert result[0].page_content == '{"key": "value"}'

    def test_split_documents_splits_long_text(self, monkeypatch):
        monkeypatch.setattr(
            "src.config.settings.settings", Mock(chunk_size=50, chunk_overlap=10)
        )

        splitter = DocumentSplitter()
        long_text = (
            "This is a long text that should be split into multiple chunks for testing purposes. "
            * 5
        )
        documents = [Document(page_content=long_text, metadata={})]
        result = splitter.split_documents(documents)

        assert len(result) > 1
        for chunk in result:
            assert len(chunk.page_content) <= 60  # chunk_size + some margin

    def test_split_documents_preserves_metadata(self, monkeypatch):
        monkeypatch.setattr(
            "src.config.settings.settings", Mock(chunk_size=100, chunk_overlap=20)
        )

        splitter = DocumentSplitter()
        documents = [
            Document(
                page_content="Test content", metadata={"source": "test.pdf", "page": 1}
            )
        ]
        result = splitter.split_documents(documents)

        assert len(result) > 0
        assert result[0].metadata["source"] == "test.pdf"
        assert result[0].metadata["page"] == 1
