import re
from pathlib import Path
from typing import ClassVar
from collections.abc import Callable
from langchain_core.documents import Document

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentCleaner:
    """
    Cleans document content before chunking.
    """

    @classmethod
    def clean_documents(cls,documents: list[Document]) -> list[Document]:
        """
        Clean all loaded documents.
        """

        cleaned_documents = []
        for document in documents:
            cleaned_documents.append(
                Document(
	                page_content=cls._clean_text(document.page_content),
                    metadata=document.metadata
                )
            )

        logger.info("Cleaned %d document(s).",len(cleaned_documents),)
        return cleaned_documents

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Normalize text by removing common PDF artifacts.
        """

        text = text.replace("\r\n", "\n")
        for pattern in (
            r"(?im)^page\s+\d+(\s+of\s+\d+)?\s*$",
            r"(?im)^page\s*[:\-]?\s*\d+\s*$",
            r"(?im)^pg\.?\s*\d+\s*$",
            r"(?im)^\d+\s*/\s*\d+\s*$",
        ):
            text = re.sub(pattern, "", text)

        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()


class DocumentLoader:
    """
    Loads supported document formats into LangChain Documents.
    """

    _LOADERS: ClassVar[dict[str, Callable[[str | Path], list[Document]]]]

    # Load PDF pages as LangChain documents for ingestion.
    @staticmethod
    def load_pdf(file_path: str | Path) -> list[Document]:
        from langchain_community.document_loaders import PyPDFLoader

        logger.info("Loading PDF: %s", file_path)
        return PyPDFLoader(str(file_path)).load()

    # Load plain text files as LangChain documents for ingestion.
    @staticmethod
    def load_text(file_path: str | Path) -> list[Document]:
        from langchain_community.document_loaders import TextLoader

        logger.info("Loading Text File: %s", file_path)
        return TextLoader(str(file_path), encoding="utf-8",).load()

    # Load Markdown files while preserving document metadata.
    @staticmethod
    def load_markdown(file_path: str | Path) -> list[Document]:
        from langchain_community.document_loaders import UnstructuredMarkdownLoader

        logger.info("Loading Markdown File: %s", file_path)
        return UnstructuredMarkdownLoader(str(file_path)).load()

    # Load DOCX files through the LangChain community loader.
    @staticmethod
    def load_docx(file_path: str | Path) -> list[Document]:
        from langchain_community.document_loaders import Docx2txtLoader
        logger.info("Loading Word Document: %s", file_path)
        return Docx2txtLoader(str(file_path)).load()  # type: ignore

    # Load CSV rows as document records for structured retrieval.
    @staticmethod
    def load_csv(file_path: str | Path) -> list[Document]:
        from langchain_community.document_loaders import CSVLoader

        logger.info("Loading CSV File: %s", file_path)
        return CSVLoader(str(file_path)).load()

    # Load JSON content as document records for structured retrieval.
    @staticmethod
    def load_json(file_path: str | Path) -> list[Document]:
        from langchain_community.document_loaders import JSONLoader
        logger.info("Loading JSON File: %s", file_path)
        return JSONLoader(
            str(file_path),
            jq_schema=".",
            text_content=False
        ).load()

    @classmethod
    def load_file(cls, file_path: str | Path) -> list[Document]:
        """
        Load a file using the appropriate loader based on its extension.
        """
        file_path = Path(file_path)

        # Production note: keep loader selection centralized so new formats can be added
        # without scattering format-specific logic across the ingestion pipeline.
        loader = cls._LOADERS.get(file_path.suffix.lower())

        if loader is None:
            raise ValueError(
                f"Unsupported file extension: {file_path.suffix}"
            )

        try:
            return loader(file_path)

        except Exception:
            logger.exception(
                "Failed to load file: %s",
                file_path,
            )
            raise


DocumentLoader._LOADERS = {
    ".pdf": DocumentLoader.load_pdf,
    ".txt": DocumentLoader.load_text,
    ".text": DocumentLoader.load_text,
    ".md": DocumentLoader.load_markdown,
    ".markdown": DocumentLoader.load_markdown,
    ".docx": DocumentLoader.load_docx,
    ".csv": DocumentLoader.load_csv,
    ".json": DocumentLoader.load_json,
}
