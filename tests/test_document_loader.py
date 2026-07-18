import pytest
from langchain_core.documents import Document
from src.loaders.document_loader import DocumentLoader, DocumentCleaner


class TestDocumentCleaner:
    def test_clean_documents_returns_empty_list_for_empty_input(self):
        result = DocumentCleaner.clean_documents([])
        assert result == []

    def test_clean_documents_normalizes_line_endings(self):
        documents = [Document(page_content="line1\r\nline2", metadata={})]
        result = DocumentCleaner.clean_documents(documents)
        assert "\r\n" not in result[0].page_content
        assert "line1\nline2" in result[0].page_content

    def test_clean_documents_removes_page_numbers(self):
        documents = [Document(page_content="Page 1 of 5\nSome content", metadata={})]
        result = DocumentCleaner.clean_documents(documents)
        assert "Page 1 of 5" not in result[0].page_content

    def test_clean_documents_removes_trailing_whitespace(self):
        documents = [Document(page_content="line1   \nline2\t", metadata={})]
        result = DocumentCleaner.clean_documents(documents)
        assert result[0].page_content == "line1\nline2"

    def test_clean_documents_reduces_multiple_spaces(self):
        documents = [Document(page_content="word1  word2   word3", metadata={})]
        result = DocumentCleaner.clean_documents(documents)
        assert "  " not in result[0].page_content
        assert "word1 word2 word3" in result[0].page_content

    def test_clean_documents_preserves_metadata(self):
        documents = [Document(page_content="content", metadata={"source": "test.pdf", "page": 1})]
        result = DocumentCleaner.clean_documents(documents)
        assert result[0].metadata["source"] == "test.pdf"
        assert result[0].metadata["page"] == 1


class TestDocumentLoader:
    def test_load_file_raises_error_for_unsupported_extension(self, tmp_path):
        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.write_text("content")
        
        with pytest.raises(ValueError, match="Unsupported file extension"):
            DocumentLoader.load_file(unsupported_file)

    def test_loaders_dict_contains_all_supported_formats(self):
        expected_extensions = {".pdf", ".txt", ".text", ".md", ".markdown", ".docx", ".csv", ".json"}
        actual_extensions = set(DocumentLoader._LOADERS.keys())
        assert actual_extensions == expected_extensions

    def test_load_pdf_is_callable(self):
        assert callable(DocumentLoader.load_pdf)

    def test_load_text_is_callable(self):
        assert callable(DocumentLoader.load_text)

    def test_load_markdown_is_callable(self):
        assert callable(DocumentLoader.load_markdown)

    def test_load_docx_is_callable(self):
        assert callable(DocumentLoader.load_docx)

    def test_load_csv_is_callable(self):
        assert callable(DocumentLoader.load_csv)

    def test_load_json_is_callable(self):
        assert callable(DocumentLoader.load_json)

    def test_load_file_with_path_string(self, tmp_path):
        text_file = tmp_path / "test.txt"
        text_file.write_text("test content")
        
        with pytest.raises(ValueError, match="Unsupported file extension"):
            DocumentLoader.load_file(str(text_file).replace(".txt", ".xyz"))

    def test_load_file_with_path_object(self, tmp_path):
        text_file = tmp_path / "test.txt"
        text_file.write_text("test content")
        
        with pytest.raises(ValueError, match="Unsupported file extension"):
            DocumentLoader.load_file(text_file.with_suffix(".xyz"))
