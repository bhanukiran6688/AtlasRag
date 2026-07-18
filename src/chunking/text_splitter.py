from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentSplitter:
    """
    Splits unstructured documents into overlapping chunks. CSV and JSON are treated as structured data and are
    returned unchanged.
    
    Supports parent-child retrieval where smaller child chunks are used for search and larger parent chunks
    provide better context for answer generation.
    """

    def __init__(self, enable_parent_child: bool = False) -> None:
        if settings.chunk_overlap >= settings.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        # Production note: chunk size and overlap are important retrieval tuning knobs
        # and should be validated explicitly before indexing begins.
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )
        
        # Parent chunk splitter for larger context chunks
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size * 3,  # Parent chunks are 3x larger
            chunk_overlap=settings.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )
        
        self.enable_parent_child = enable_parent_child

    def split_documents(self, documents: list[Document], file_type: str | None = None) -> list[Document]:
        """
        Split documents into chunks. Structured formats (CSV and JSON) are not recursively split.
        
        If parent-child retrieval is enabled, creates both child chunks (for search) and parent chunks (for context).
        """

        if not documents:
            return []

        if file_type is not None and file_type.lower() in ("csv", "json"):
            logger.info("Skipping recursive splitting for structured data format: %s", file_type)
            return documents

        if self.enable_parent_child:
            return self._split_documents_parent_child(documents)
        
        chunks = self._splitter.split_documents(documents)
        logger.info("Processed %d chunks.", len(chunks))
        return chunks

    def _split_documents_parent_child(self, documents: list[Document]) -> list[Document]:
        """
        Split documents into parent-child chunks for improved retrieval.
        
        Child chunks: Smaller, used for vector search
        Parent chunks: Larger, provide better context for answers
        """
        all_chunks = []
        
        for doc in documents:
            # Create parent chunks first
            parent_chunks = self._parent_splitter.split_documents([doc])
            
            # For each parent chunk, create child chunks
            for parent_idx, parent_chunk in enumerate(parent_chunks):
                # Mark this as a parent chunk
                parent_chunk.metadata["is_parent"] = True
                parent_chunk.metadata["parent_id"] = f"parent_{parent_idx}"
                all_chunks.append(parent_chunk)
                
                # Create child chunks from this parent
                child_chunks = self._splitter.split_documents([parent_chunk])
                
                for child_idx, child_chunk in enumerate(child_chunks):
                    # Link child to parent
                    child_chunk.metadata["is_parent"] = False
                    child_chunk.metadata["parent_id"] = f"parent_{parent_idx}"
                    child_chunk.metadata["child_id"] = f"child_{parent_idx}_{child_idx}"
                    all_chunks.append(child_chunk)
        
        logger.info("Processed %d chunks (%d parent, %d child) with parent-child retrieval.", 
                   len(all_chunks), len(parent_chunks), len(all_chunks) - len(parent_chunks))
        return all_chunks