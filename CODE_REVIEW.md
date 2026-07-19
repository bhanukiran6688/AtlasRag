# Staff Engineer Code Review

This review documents findings only. It does not apply fixes automatically.

## Executive Summary

The repository has a strong modular RAG architecture for a portfolio project: ingestion, embeddings, vector stores, retrieval, gateway, guardrails, context building, and UI are separated. The main risks are production hardening gaps: test coverage, full hybrid indexing, and some synchronous operations that will not scale.

## Code Smells

### RAGService still owns several orchestration details
- File: `src/services/rag_service.py`
- Issue: orchestration, structured parsing, memory updates, and result construction remain in one class.
- Impact: harder unit testing and future maintenance.
- Recommendation: extract structured output parsing and memory updates when the service grows further.

### Retriever mixes several retrieval responsibilities
- File: `src/retrievers/retriever.py`
- Issue: dense search, BM25, RRF, reranking, lost-middle ordering, and tokenization live in one class.
- Impact: useful for learning, but large for production.
- Recommendation: split into strategy classes when behavior stabilizes.

## Architecture Issues

### Hybrid search is not full corpus hybrid
- File: `src/retrievers/retriever.py`
- Issue: BM25 ranks dense candidates, not the complete document corpus.
- Impact: lexical-only matches may be missed if dense retrieval does not include them.
- Recommendation: add a persistent BM25/OpenSearch index over all chunks.

### Output guardrails are deterministic only
- File: `src/guardrails/input_guardrails.py`
- Issue: prompt leakage and citation enforcement use rules, not semantic verification.
- Impact: sophisticated hallucinations can pass if citations are valid but claim content is unsupported.
- Recommendation: add verifier model or entailment scoring.

## Performance Bottlenecks

### Cross-encoder reranking can be expensive
- File: `src/retrievers/retriever.py`
- Issue: reranker loads locally and still scores synchronously, even with batching controls.
- Impact: latency spikes, especially on CPU.
- Recommendation: benchmark deployment defaults and disable by default in low-resource setups.

### Query expansion and decomposition add LLM calls
- File: `src/services/rag_service.py`
- Issue: optional features still increase latency and cost when enabled.
- Impact: controlled by adaptive triggers and query-planning call limits, but still worth monitoring.
- Recommendation: add metrics for how often adaptive planning triggers and whether it improves answers.

### Ingestion is synchronous
- File: `src/index.py`
- Issue: files are processed one by one.
- Impact: slow for large document sets.
- Recommendation: add batching and background workers later.

## Security Concerns

### Guardrails still use deterministic checks
- File: `src/guardrails/input_guardrails.py`
- Issue: adversarial tests and claim-context support checks cover common failures, but a model-based classifier is absent.
- Impact: sophisticated jailbreaks and semantic misinterpretations remain possible.
- Recommendation: add an optional verifier model for high-assurance deployments.

## Maintainability Problems

## RAG Anti-Patterns

### Full hybrid search is overstated if not documented carefully
- Issue: hybrid is candidate-level, not corpus-level.
- Impact: recruiters may ask about this; be ready to explain the trade-off.
- Recommendation: clearly describe it as first-stage dense + local BM25 candidate fusion.

### Citation support is lexical rather than semantic
- Issue: cited claim terms are checked against context, but paraphrases and logical contradictions require semantic verification.
- Impact: remaining hallucination risk.
- Recommendation: add claim-context entailment verification for high-assurance deployments.

### Prompt includes conversation history without retrieval-aware memory
- Issue: memory is injected directly into prompt.
- Impact: prior turns can influence answer even when not grounded.
- Recommendation: summarize or retrieve memory separately with guardrails.

## Prompt Engineering Issues

### Citation instruction may conflict with abstention
- Issue: all factual statements require claims, but abstention answers do not.
- Impact: handled by output guardrail markers, but worth testing.
- Recommendation: add tests for insufficient-information responses.

## Retrieval Weaknesses

### Score semantics vary by strategy
- Issue: similarity distance, MMR rank, BM25 negative score, and rerank score differ.
- Impact: metrics can confuse users.
- Recommendation: expose separate fields or normalize scores per strategy.

## Latency and Cost Opportunities

1. Disable reranking unless high accuracy is required.
2. Track per-request cost in a local metrics file or dashboard.
3. Add adaptive retrieval strategy selection.
4. Add model routing based on answer complexity.
5. Batch embedding during ingestion.
6. Tune reranker batch size per deployment target.