import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.index import process_and_index_file, IndexManifest
from langchain_core.documents import Document

@pytest.fixture
def mock_index_deps():
    splitter = MagicMock()
    vector_store = MagicMock()
    manifest = MagicMock(spec=IndexManifest)
    return splitter, vector_store, manifest

def test_process_and_index_file_success(mock_index_deps, tmp_path):
    splitter, vector_store, manifest = mock_index_deps
    
    # Setup mock file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")
    
    # Setup mocks
    manifest.has_current_file.return_value = False
    manifest.get_file.return_value = None
    
    doc = Document(page_content="Test content", metadata={})
    splitter.split_documents.return_value = [doc]
    
    with patch("src.loaders.document_loader.DocumentLoader.load_file", return_value=[doc]):
        count = process_and_index_file(test_file, splitter, vector_store, manifest)
        
    assert count == 1
    vector_store.add_documents.assert_called_once()
    manifest.record_file.assert_called_once()

def test_process_and_index_file_skips_unchanged(mock_index_deps, tmp_path):
    splitter, vector_store, manifest = mock_index_deps
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")
    
    manifest.has_current_file.return_value = True
    
    count = process_and_index_file(test_file, splitter, vector_store, manifest)
    
    assert count == 0
    vector_store.add_documents.assert_not_called()
