import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from src.utils.exceptions import IngestionError

from src.chunking.text_splitter import DocumentSplitter
from src.config.settings import settings
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.loaders.document_loader import DocumentCleaner, DocumentLoader
from src.vectorstores.base import VectorStore, VectorStoreFactory
from src.retrievers.bm25_index import BM25Index

MANIFEST_PATH = settings.data_dir / "index_manifest.json"


# Store file-level indexing state for incremental ingestion.
@dataclass(slots=True)
class IndexedFile:
    path: str
    content_hash: str
    chunks_indexed: int
    deleted: bool = False
    chunk_ids: list[str] = field(default_factory=list)


class IndexManifest:
    """
    Tracks indexed files so repeated runs only add new or changed documents.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._files = self._load()

    def has_current_file(self, file_path: Path, content_hash: str) -> bool:
        indexed_file = self._files.get(self._manifest_key(file_path))
        return (
            indexed_file is not None
            and not indexed_file.deleted
            and indexed_file.content_hash == content_hash
        )

    def record_file(
        self,
        file_path: Path,
        content_hash: str,
        chunks_indexed: int,
        chunk_ids: list[str] | None = None,
    ) -> None:
        self._files[self._manifest_key(file_path)] = IndexedFile(
            path=str(file_path),
            content_hash=content_hash,
            chunks_indexed=chunks_indexed,
            deleted=False,
            chunk_ids=list(chunk_ids or []),
        )

    def mark_deleted(self, file_path: Path) -> None:
        key = self._manifest_key(file_path)
        indexed_file = self._files.get(key)
        if indexed_file is None:
            return
        indexed_file.deleted = True

    def is_deleted(self, file_path: Path) -> bool:
        indexed_file = self._files.get(self._manifest_key(file_path))
        return bool(indexed_file and indexed_file.deleted)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "files": {
                key: {
                    "path": indexed_file.path,
                    "content_hash": indexed_file.content_hash,
                    "chunks_indexed": indexed_file.chunks_indexed,
                    "deleted": indexed_file.deleted,
                    "chunk_ids": indexed_file.chunk_ids,
                }
                for key, indexed_file in sorted(self._files.items())
            }
        }
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load(self) -> dict[str, IndexedFile]:
        if not self._path.exists():
            return {}

        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"Index manifest is invalid: {self._path}. Starting fresh.")
            return {}

        files = payload.get("files", {})
        if not isinstance(files, dict):
            return {}

        indexed_files: dict[str, IndexedFile] = {}
        for key, value in files.items():
            if not isinstance(value, dict):
                continue
            indexed_files[key] = IndexedFile(
                path=str(value.get("path", "")),
                content_hash=str(value.get("content_hash", "")),
                chunks_indexed=int(value.get("chunks_indexed", 0)),
                deleted=bool(value.get("deleted", False)),
                chunk_ids=list(value.get("chunk_ids", []) or []),
            )
        return indexed_files

    def remove_missing_files(self, current_paths: set[str]) -> None:
        # FEATURE: Document lifecycle tracking for deletions and re-indexing
        for key, indexed_file in list(self._files.items()):
            if key in current_paths:
                continue
            indexed_file.deleted = True

    def get_file(self, file_path: Path) -> IndexedFile | None:
        return self._files.get(self._manifest_key(file_path))

    @staticmethod
    def _manifest_key(file_path: Path) -> str:
        return str(file_path.resolve())


# Calculate a stable content hash for document change detection.
def calculate_file_hash(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Attach stable metadata used for deduplication, citations, and vector IDs.
def add_chunk_identity(chunks, file_path: Path, content_hash: str) -> list[str]:
    document_id = hashlib.sha256(str(file_path.resolve()).encode("utf-8")).hexdigest()
    file_type = file_path.suffix.lstrip(".").lower()
    chunk_ids: list[str] = []
    for index, chunk in enumerate(chunks):
        chunk.metadata["document_id"] = document_id
        chunk.metadata["document_hash"] = content_hash
        chunk.metadata["file_type"] = file_type
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_id"] = f"{document_id}:{content_hash}:{index}"
        # FEATURE: Document versioning / lifecycle metadata
        chunk.metadata["indexed_at"] = perf_counter()
        chunk.metadata["source_path"] = str(file_path.resolve())
        chunk_ids.append(chunk.metadata["chunk_id"])
    return chunk_ids


# FEATURE: Incremental indexing support
# Keep track of whether a file was previously indexed so unchanged files can be skipped quickly.
def should_index_file(
    file_path: Path, manifest: IndexManifest, content_hash: str
) -> bool:
    return not manifest.has_current_file(file_path=file_path, content_hash=content_hash)


def process_and_index_file(
    file_path: Path,
    splitter: DocumentSplitter,
    vector_store: VectorStore,
    manifest: IndexManifest,
    bm25_index: BM25Index | None = None,
) -> int:
    """
    Load, clean, split, identify, and index a single document.

    RAG Concept: Multi-Index Ingestion
    - Documents are indexed in both vector store (for semantic search) and BM25 index (for lexical search)
    - This enables true hybrid search where both semantic and lexical retrieval work over the full corpus
    - Vector embeddings capture meaning, BM25 captures exact keyword matches
    """

    try:
        content_hash = calculate_file_hash(file_path)
        if not should_index_file(
            file_path=file_path, manifest=manifest, content_hash=content_hash
        ):
            print(f"Skipping unchanged file: {file_path.name}")
            return 0

        previous_entry = manifest.get_file(file_path)
        if previous_entry and previous_entry.chunk_ids:
            # RAG Concept: Document Deletion from Multiple Indices
            # When re-indexing, remove old chunks from both vector store and BM25 index
            vector_store.delete_documents(previous_entry.chunk_ids)
            if bm25_index:
                for chunk_id in previous_entry.chunk_ids:
                    bm25_index.delete_document(chunk_id)

        documents = DocumentCleaner.clean_documents(DocumentLoader.load_file(file_path))
        chunks = splitter.split_documents(
            documents=documents, file_type=file_path.suffix.lstrip(".")
        )
        chunk_ids = add_chunk_identity(
            chunks=chunks, file_path=file_path, content_hash=content_hash
        )

        # Index in vector store for semantic search
        vector_store.add_documents(chunks)

        # RAG Concept: BM25 Indexing for Lexical Search
        # Index each chunk in the BM25 index for full corpus lexical search
        # This enables hybrid search where BM25 can find documents that dense search might miss
        if bm25_index:
            for chunk in chunks:
                chunk_id = chunk.metadata.get("chunk_id")
                if chunk_id:
                    bm25_index.add_document(
                        doc_id=chunk_id,
                        content=chunk.page_content,
                        metadata=chunk.metadata,
                    )
            # Persist BM25 index after each file for crash recovery
            bm25_index.save_index()

        manifest.record_file(
            file_path=file_path,
            content_hash=content_hash,
            chunks_indexed=len(chunks),
            chunk_ids=chunk_ids,
        )
        print(f"Indexed {len(chunks)} chunks from {file_path.name}")
        return len(chunks)

    except ValueError as exc:
        print(f"Skipping {file_path.name}: {exc}")
        return 0

    except Exception as exc:
        print(f"Failed to process {file_path.name}: {exc}", file=sys.stderr)
        return 0


# Run the document indexing workflow for all configured source documents.
def main() -> None:
    overall_start = perf_counter()

    print("\n" + "=" * 70)
    print("Starting RAG Document Indexing")
    print("=" * 70)

    if not settings.documents_dir.exists():
        raise IngestionError(
            f"Documents directory does not exist: {settings.documents_dir}"
        )

    splitter = DocumentSplitter(
        enable_parent_child=settings.enable_parent_child_retrieval
    )
    embedding_generator = EmbeddingGenerator()
    vector_store = VectorStoreFactory.create(
        embeddings=embedding_generator.get_embeddings()
    )
    manifest = IndexManifest(MANIFEST_PATH)
    files = sorted(file for file in settings.documents_dir.iterdir() if file.is_file())
    manifest.remove_missing_files({str(file.resolve()) for file in files})

    if not files:
        print(f"No files found in {settings.documents_dir.resolve()}")
        return

    # Initialize BM25 index for full corpus lexical search
    bm25_index = BM25Index()

    total_files = len(files)
    files_processed = 0
    chunks_indexed = 0

    for index, file in enumerate(files, start=1):
        file_start = perf_counter()
        print(f"\n[{index}/{total_files}] Processing: {file.name}")
        chunk_count = process_and_index_file(
            file_path=file,
            splitter=splitter,
            vector_store=vector_store,
            manifest=manifest,
            bm25_index=bm25_index,
        )
        print(f"Finished {file.name} in {perf_counter() - file_start:.3f} sec")

        chunks_indexed += chunk_count
        if chunk_count > 0:
            files_processed += 1

    manifest.save()

    # Save BM25 index after all files processed
    bm25_index.save_index()

    print("\n" + "=" * 70)
    print("Indexing Summary")
    print("=" * 70)
    print(f"Files Processed : {files_processed}")
    print(f"Chunks Indexed  : {chunks_indexed}")
    print(f"Vector Store    : {settings.vector_store.title()}")
    print(f"Store Name      : {vector_store.name}")
    print(f"Location        : {vector_store.location}")
    print(f"Manifest        : {MANIFEST_PATH}")
    print(f"BM25 Index      : {bm25_index.index_path}")
    print(f"BM25 Documents  : {bm25_index.total_docs}")
    print(f"Total Runtime   : {perf_counter() - overall_start:.3f} sec")
    print("=" * 70)

    if files_processed == 0:
        print("\nNo new or changed documents were indexed.")
    else:
        print("\nIndexing completed successfully.")


if __name__ == "__main__":
    main()
