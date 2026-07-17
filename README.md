# Production RAG Platform

A modular Retrieval-Augmented Generation system built to demonstrate production-oriented AI engineering patterns: configurable ingestion, vector search, MMR retrieval, hybrid retrieval, reranking, citation enforcement, guardrails, LLM gateway routing, caching, and a Streamlit UI.

This project is not a minimal demo. It is structured as a small RAG platform where each major concern is separated into a focused component.

## Problem Statement

LLM applications become unreliable when they answer from memory, couple business logic to one model vendor, or rebuild indexes blindly. This project addresses those problems by:

- grounding answers in indexed documents
- routing all model calls through a provider-neutral gateway
- tracking document versions during ingestion
- enforcing citations and output checks
- supporting configurable retrieval strategies
- keeping Streamlit as UI only

## Architecture

```text
Documents
  -> DocumentLoader
  -> DocumentCleaner
  -> DocumentSplitter
  -> IndexManifest + chunk metadata
  -> EmbeddingGenerator
  -> Chroma/Pinecone VectorStore

User Question
  -> Streamlit UI
  -> RAGService
  -> InputGuardrails
  -> Query Expansion / Decomposition
  -> Retriever
       -> Similarity / MMR / Hybrid BM25 + Dense RRF
       -> Optional Cross-Encoder Reranking
       -> Lost-in-the-Middle Reordering
  -> ContextBuilder
  -> PromptBuilder
  -> LiteLLMGateway
       -> LiteLLM Router
       -> Gemini / Groq / Cerebras
  -> OutputGuardrails
  -> Answer + Citations
```

## Folder Structure

```text
.
├── app.py                         # Streamlit UI
├── src
│   ├── cache                      # In-memory TTL response cache
│   ├── chunking                   # Recursive text splitting
│   ├── config                     # Pydantic settings and validation
│   ├── embeddings                 # HuggingFace/Gemini embedding adapters
│   ├── guardrails                 # Input and output safety checks
│   ├── llm                        # LiteLLM gateway and routing
│   ├── loaders                    # PDF/TXT/Markdown/DOCX/CSV/JSON loading
│   ├── prompts                    # Context and prompt construction
│   ├── retrievers                 # Similarity, MMR, hybrid, reranking
│   ├── services                   # RAG orchestration
│   ├── utils                      # Logging and custom exceptions
│   ├── vectorstores               # Chroma and Pinecone adapters
│   └── index.py                   # Incremental indexing entrypoint
├── tests                          # Startup and manifest tests
├── requirements.txt
└── README.md
```

## Tech Stack

- Python
- Streamlit
- LangChain document loaders and splitters
- HuggingFace SentenceTransformer embeddings
- Gemini embeddings support
- Chroma
- Pinecone
- LiteLLM Router
- SentenceTransformers CrossEncoder reranking
- Pydantic Settings

## End-to-End RAG Pipeline

1. Place documents in `data/documents`.
2. Run the indexer.
3. Loader converts files into LangChain `Document` objects.
4. Cleaner normalizes text and removes common PDF artifacts.
5. Splitter chunks unstructured documents.
6. Index manifest skips unchanged documents.
7. Stable metadata is attached: `document_id`, `document_hash`, `chunk_id`, `file_type`, `source_path`.
8. Embeddings are generated.
9. Chunks are stored in Chroma or Pinecone.
10. User asks a question in Streamlit.
11. Input guardrails sanitize or block unsafe input.
12. Optional query expansion/decomposition improves recall.
13. Retriever fetches candidates using configured strategy.
14. Context builder deduplicates, compresses, and trims context.
15. Prompt builder adds conversation history and answer rules.
16. LiteLLM gateway selects model, checks cache/rate limits, and handles fallback.
17. Output guardrails validate citations and grounding.
18. UI displays answer, citations, retrieval chunks, metrics, and grounding confidence.

## Features

- Multi-format ingestion: PDF, TXT, Markdown, DOCX, CSV, JSON
- Incremental indexing with file hash manifest
- Re-indexing support for changed files
- Stable chunk IDs for vector stores
- Chroma and Pinecone support
- Metadata filtering by source, file type, and custom metadata
- Similarity search
- Maximum Marginal Relevance
- Hybrid dense + BM25 retrieval
- Reciprocal Rank Fusion
- Optional cross-encoder reranking with `BAAI/bge-reranker-base`
- Lost-in-the-middle mitigation
- Query expansion and decomposition
- Context window management
- Lightweight context compression
- LiteLLM gateway
- Gemini/Groq/Cerebras routing and fallback
- Multiple Gemini key support
- In-memory LLM response caching
- Local rate limiting
- Structured JSON answers
- Claim-level citation enforcement
- Input guardrails for PII, prompt injection, forbidden topics
- Output guardrails for prompt leakage and ungrounded answers
- Per-session in-memory conversation memory
- Streamlit UI with retrieval diagnostics

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Create `.env` in the project root:

```env
LOG_LEVEL=INFO
DATA_DIR=data

GOOGLE_API_KEY=""
GOOGLE_API_KEY2=""
GOOGLE_API_KEY3=""
GROQ_API_KEY=""
CEREBRAS_API_KEY=""

EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

VECTOR_STORE=chroma
PINECONE_API_KEY=""
PINECONE_INDEX=rag-index
```

Important retrieval settings live in `src/config/settings.py`:

```python
retrieval_strategy = "mmr"  # similarity | mmr | hybrid
retrieval_top_k = 5
retrieval_candidate_k = 20
retrieval_enable_reranking = True
retrieval_enable_lost_middle_reordering = True
```

## Usage

Index documents:

```bash
python -m src.index
```

Run the app:

```bash
streamlit run app.py
```

In Streamlit, you can:

- ask questions
- enable query expansion
- enable query decomposition
- apply metadata filters
- inspect retrieved chunks
- inspect citations
- view cache, route, latency, tokens, and grounding confidence

## Design Decisions

- Use LangChain for loaders, splitters, and vector store integrations only.
- Keep orchestration in custom `RAGService`.
- Keep provider-specific LLM logic inside `LiteLLMGateway`.
- Use configuration for retrieval strategy and provider selection.
- Keep guardrails independent of LiteLLM so they protect all providers.
- Use in-memory caching now; Redis can replace the cache later.
- Keep BM25 local to retrieved candidates until a full document store exists.

## Performance Optimizations

- Incremental indexing avoids reprocessing unchanged files.
- Stable chunk IDs support re-indexing changed documents.
- In-memory LLM response cache reduces duplicate calls.
- Context trimming and compression reduce prompt size.
- Retrieval candidate count and Top-K are configurable.
- Optional reranking can be disabled for lower latency.
- Query expansion/decomposition are user-controlled to avoid unnecessary calls.

## Security Considerations

- `.env` is ignored by Git.
- Input guardrails redact PII and block common injection attempts.
- Output guardrails block prompt leakage.
- Claim-level citation checks reduce unsupported answers.
- Secrets should be managed by environment variables or a secret manager.
- Current guardrails are deterministic and should be tested against adversarial examples.

## Known Limitations

- No HTTP API yet.
- Guardrails are rule-based, not a full policy engine.
- In-memory cache is not shared across processes.
- Conversation memory is intentionally per-session and not durable across restarts.
- Observability & Evaluation is not implemented.

## Future Improvements

- Full corpus BM25 index
- Parent-child retrieval
- Metadata-aware UI filters from discovered corpus values
- RAG evaluation harness
- Redis cache and distributed rate limiting
- Background ingestion workers
- FastAPI service layer
- CI/CD pipeline
- Docker deployment
- Prompt regression tests

## Deployment

For local deployment:

```bash
streamlit run app.py
```

For production deployment, package with:

- environment-based secrets
- managed vector database
- dependency-pinned image
- process manager or container platform

## Highlights

- Built a modular production-style RAG architecture from scratch.
- Implemented provider-neutral LLM gateway using LiteLLM.
- Added MMR, hybrid retrieval, RRF, reranking, and lost-in-the-middle mitigation.
- Implemented claim-level citation enforcement and output grounding checks.
- Built incremental indexing with stable chunk identity and manifest tracking.
- Separated Streamlit UI from business logic.
- Added configurable Chroma/Pinecone vector store abstraction.