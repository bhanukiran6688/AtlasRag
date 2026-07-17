from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentSplitter:
    """
    Splits unstructured documents into overlapping chunks. CSV and JSON are treated as structured data and are
    returned unchanged.
    """

    def __init__(self) -> None:
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

    def split_documents(self, documents: list[Document], file_type: str | None = None) -> list[Document]:
        """
        Split documents into chunks. Structured formats (CSV and JSON) are not recursively split.
        """

        if not documents:
            return []

        if file_type is not None and file_type.lower() in ("csv", "json"):
            logger.info("Skipping recursive splitting for structured data format: %s", file_type)
            return documents

        chunks = self._splitter.split_documents(documents)
        logger.info("Processed %d chunks.", len(chunks))
        return chunks