# Metadata Schema Reindexing Guide

## Overview

This document explains when and how to reindex your vector store when metadata schemas change in the AtlasRAG system.

**Why this matters:** Vector stores store document embeddings along with their metadata. When you change the metadata schema (add fields, rename fields, change structure), old documents won't have the new metadata. This causes issues with:
- Metadata filters missing old documents
- Parent-child retrieval failing on old chunks
- Inconsistent BM25 indexing if document IDs change

This guide provides procedures to safely reindex your data when metadata schemas evolve.

## When Reindexing is Required

Reindexing is required when:

1. **Adding new metadata fields** to documents during ingestion
2. **Changing metadata field names** or structure
3. **Modifying metadata filter logic** that depends on specific fields
4. **Adding parent-child retrieval** with new `parent_id` or `is_parent` fields
5. **Changing chunk_id generation** logic

## Why Reindexing Matters

The vector store stores document embeddings along with their metadata. When you change the metadata schema:

- **Old documents** will lack the new metadata fields
- **Metadata filters** may miss old documents that don't have the required fields
- **Parent-child retrieval** may fail if old chunks don't have parent metadata
- **BM25 indexing** may be inconsistent if document IDs change

## Reindexing Procedure

### Step 1: Backup Current Index

```bash
# For ChromaDB
cp -r data/chroma_db data/chroma_db_backup_$(date +%Y%m%d)

# For Pinecone, export your index if needed
# (Pinecone has built-in backup features)
```

### Step 2: Clear Existing Index

```python
from src.index import IndexManager

index_manager = IndexManager()
index_manager.clear_index()
```

### Step 3: Re-run Ingestion

```bash
python src/index.py --reindex
```

Or programmatically:

```python
from src.index import IndexManager

index_manager = IndexManager()
index_manager.ingest_documents(
    documents_dir="data/documents",
    reindex=True  # Force full reindex
)
```

### Step 4: Verify Metadata

After reindexing, verify that documents have the expected metadata:

```python
from src.vectorstores.base import VectorStore

vector_store = VectorStore()
results = vector_store.similarity_search_with_score(
    query="test query",
    k=5,
    metadata_filter=None
)

for doc, score in results:
    print(f"Metadata: {doc.metadata}")
```

## Metadata Schema Best Practices

### 1. Version Your Metadata Schema

Include a `metadata_version` field:

```python
metadata = {
    "source": "document.pdf",
    "page": 1,
    "chunk_id": "doc_1_chunk_0",
    "metadata_version": "2.0",  # Increment when schema changes
    # ... other fields
}
```

### 2. Use Optional Fields for New Additions

When adding new fields, make them optional to maintain backward compatibility:

```python
# Old documents won't have this field
metadata = {
    "source": "document.pdf",
    "page": 1,
    "chunk_id": "doc_1_chunk_0",
    "parent_id": metadata.get("parent_id"),  # Optional
    "is_parent": metadata.get("is_parent", False),  # Optional with default
}
```

### 3. Document Schema Changes

Maintain a changelog in this file:

```markdown
## Schema Changelog

### Version 2.0 (2024-01-15)
- Added `parent_id` field for parent-child retrieval
- Added `is_parent` boolean field
- **Reindex required: Yes**

### Version 1.0 (2024-01-01)
- Initial schema with `source`, `page`, `chunk_id`
- **Reindex required: No**
```

## Partial Reindexing Strategies

If you have a large corpus and can't afford full reindexing:

### Strategy 1: Gradual Migration

Reindex documents in batches by date or source:

```python
from src.index import IndexManager

index_manager = IndexManager()
index_manager.ingest_documents(
    documents_dir="data/documents/2024",
    reindex=True
)
```

### Strategy 2: Metadata Backfill

Update metadata for existing documents without full reindex:

```python
# This requires custom implementation based on your vector store
# Not all vector stores support in-place metadata updates
```

## Testing After Reindexing

After reindexing, run these tests:

```bash
# Test retrieval with metadata filters
pytest tests/test_metadata_filters.py -v

# Test parent-child retrieval if applicable
pytest tests/test_retriever.py -v -k parent

# Test full RAG pipeline
pytest tests/test_rag_service_integration.py -v
```

## Monitoring

Monitor these metrics after reindexing:

1. **Retrieval recall** - Are relevant documents still being found?
2. **Metadata filter coverage** - Do filters return expected results?
3. **Index size** - Has it grown unexpectedly?
4. **Query latency** - Is performance acceptable?

## Rollback Procedure

If reindexing causes issues:

```bash
# Restore from backup
cp -r data/chroma_db_backup_YYYYMMDD data/chroma_db

# Or for Pinecone, restore to previous snapshot
```

## Contact

For questions about metadata schema changes, refer to the main README or project documentation.
