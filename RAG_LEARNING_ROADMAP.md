# RAG Learning Roadmap

This roadmap uses this repository as the practical foundation for mastering Retrieval-Augmented Generation. The goal is to build production-grade RAG systems independently, explain design trade-offs in interviews, and design future enterprise RAG platforms without assistance.

## Stage 1: RAG Foundations

### 1. Document Loading and Parsing
- Why it matters: ingestion quality determines downstream retrieval quality.
- Core concepts: loaders, file types, parsing errors, metadata preservation.
- Hands-on exercises: add one new loader; inspect metadata from PDF, CSV, and JSON files.
- Production scenarios: a PDF parser drops tables; a CSV row should be retrieved as one record.
- Common mistakes: treating all formats as plain text; losing source/page metadata.
- Interview focus: explain how parsing quality affects answer faithfulness.
- Recommended resources: LangChain document loader docs, Unstructured docs, PyPDF docs.
- Estimated time: 4-6 hours.

### 2. Cleaning and Chunking
- Why it matters: chunk shape controls retrieval granularity.
- Core concepts: recursive splitting, overlap, semantic boundaries, structured-file exceptions.
- Hands-on exercises: tune `chunk_size` and `chunk_overlap`; compare retrieved chunks.
- Production scenarios: policy docs need paragraph-level chunks; CSV should not be recursively split.
- Common mistakes: very small chunks with missing context; very large chunks with noisy retrieval.
- Interview focus: explain chunk-size trade-offs.
- Recommended resources: LangChain text splitter docs, Pinecone chunking guides.
- Estimated time: 6-8 hours.

### 3. Embeddings
- Why it matters: embedding choice defines semantic retrieval behavior.
- Core concepts: embedding dimensions, normalization, domain models, query/document embedding parity.
- Hands-on exercises: switch HuggingFace model; compare retrieval results.
- Production scenarios: general embeddings fail on medical/legal vocabulary.
- Common mistakes: changing embedding models without reindexing.
- Interview focus: when to fine-tune or swap embedding models.
- Recommended resources: SentenceTransformers docs, MTEB leaderboard.
- Estimated time: 6-8 hours.

## Stage 2: Indexing and Vector Stores

### 4. Incremental Indexing
- Why it matters: production systems cannot rebuild all data on every change.
- Core concepts: file hashing, manifests, stable IDs, re-indexing, delete handling.
- Hands-on exercises: modify a document and confirm only changed chunks are replaced.
- Production scenarios: a nightly sync updates 20 of 10,000 files.
- Common mistakes: duplicate vectors after every indexing run.
- Interview focus: describe idempotent ingestion.
- Recommended resources: Pinecone upsert docs, Chroma IDs docs.
- Estimated time: 6-10 hours.

### 5. Metadata Modeling
- Why it matters: filters enforce scope, tenancy, and explainability.
- Core concepts: source, page, file type, namespaces, custom metadata.
- Hands-on exercises: add metadata filters for file type and source in the UI.
- Production scenarios: HR users can search only HR documents.
- Common mistakes: inconsistent metadata keys across loaders.
- Interview focus: metadata filtering vs post-filtering.
- Recommended resources: Pinecone metadata filtering docs, Chroma filtering docs.
- Estimated time: 4-6 hours.

## Stage 3: Retrieval Quality

### 6. Similarity Search and MMR
- Why it matters: retrieval should balance relevance and diversity.
- Core concepts: nearest neighbors, MMR lambda, candidate count, Top-K.
- Hands-on exercises: compare `similarity` and `mmr` strategies.
- Production scenarios: similar chunks crowd out diverse supporting evidence.
- Common mistakes: using MMR without enough fetch candidates.
- Interview focus: explain MMR mathematically and operationally.
- Recommended resources: LangChain retriever docs, Carbonell and Goldstein MMR paper.
- Estimated time: 6-8 hours.

### 7. Hybrid Search and RRF
- Why it matters: dense retrieval misses exact keywords; lexical retrieval misses paraphrases.
- Core concepts: BM25, dense vectors, RRF, ranking fusion.
- Hands-on exercises: improve local hybrid search with a full corpus BM25 index.
- Production scenarios: product IDs and policy codes require lexical matching.
- Common mistakes: running BM25 only over dense candidates and calling it full hybrid.
- Interview focus: explain why hybrid beats dense-only retrieval.
- Recommended resources: Elasticsearch BM25 docs, Azure hybrid search docs, RRF papers.
- Estimated time: 8-12 hours.

### 8. Cross-Encoder Reranking
- Why it matters: rerankers improve candidate ordering before context construction.
- Core concepts: bi-encoder vs cross-encoder, latency, batching, Top-N candidates.
- Hands-on exercises: benchmark reranking Top 20 to Top 5.
- Production scenarios: high-value queries require accuracy over speed.
- Common mistakes: reranking too many candidates synchronously.
- Interview focus: explain reranker latency/quality trade-offs.
- Recommended resources: BAAI bge-reranker docs, SentenceTransformers CrossEncoder docs.
- Estimated time: 6-10 hours.

### 9. Parent-Child Retrieval
- Why it matters: small chunks retrieve well, larger parent chunks answer well.
- Core concepts: child chunks, parent documents, metadata links, context expansion.
- Hands-on exercises: add parent IDs during indexing and retrieve parent context.
- Production scenarios: a small clause is found but answer needs surrounding policy.
- Common mistakes: returning tiny fragments to the LLM.
- Interview focus: design parent-child retrieval for long PDFs.
- Recommended resources: LangChain ParentDocumentRetriever docs.
- Estimated time: 8-12 hours.

## Stage 4: Prompting, Context, and Grounding

### 10. Context Engineering
- Why it matters: retrieval quality can be wasted by poor context assembly.
- Core concepts: ordering, deduplication, compression, source labels, context budgets.
- Hands-on exercises: compare answer quality before/after lost-in-the-middle reordering.
- Production scenarios: long context causes the model to ignore middle evidence.
- Common mistakes: blindly stuffing all retrieved chunks into the prompt.
- Interview focus: explain context window management strategies.
- Recommended resources: Lost in the Middle paper, Anthropic context engineering posts.
- Estimated time: 8-12 hours.

### 11. Citation Enforcement
- Why it matters: users need traceable answers.
- Core concepts: claim-level citations, source/page validation, abstention.
- Hands-on exercises: add tests where the model returns invalid citations.
- Production scenarios: regulated teams require source-backed factual claims.
- Common mistakes: asking for citations without validating them.
- Interview focus: how to prevent hallucinated citations.
- Recommended resources: RAGAS faithfulness docs, OpenAI structured output docs.
- Estimated time: 6-10 hours.

### 12. Hallucination Mitigation
- Why it matters: RAG systems must abstain when support is missing.
- Core concepts: grounded prompts, verifier pass, entailment, confidence.
- Hands-on exercises: build a verifier that checks answer claims against context.
- Production scenarios: retrieved chunks are irrelevant but model still answers.
- Common mistakes: relying only on "use context only" prompt text.
- Interview focus: layered hallucination defense.
- Recommended resources: RAGAS, Self-RAG paper, CRAG paper.
- Estimated time: 10-14 hours.

## Stage 5: LLM Gateway and Cost Control

### 13. LLM Routing and Fallback
- Why it matters: one provider outage should not break the app.
- Core concepts: model routing, fallback chains, retries, timeouts, key rotation.
- Hands-on exercises: simulate provider failure and confirm fallback.
- Production scenarios: Gemini quota is exhausted; route to Groq/Cerebras.
- Common mistakes: provider-specific logic leaking into service code.
- Interview focus: design a multi-provider gateway.
- Recommended resources: LiteLLM Router docs.
- Estimated time: 6-8 hours.

### 14. Caching and Rate Limiting
- Why it matters: reduces cost and protects provider budgets.
- Core concepts: TTL cache, cache keys, semantic cache, per-user limits.
- Hands-on exercises: replace in-memory cache with Redis.
- Production scenarios: repeated FAQs should not call the LLM every time.
- Common mistakes: caching unsafe personalized answers globally.
- Interview focus: cache correctness in RAG.
- Recommended resources: Redis docs, LiteLLM caching docs.
- Estimated time: 8-10 hours.

## Stage 6: Safety and Security

### 15. Guardrails
- Why it matters: user input and model output are both attack surfaces.
- Core concepts: PII redaction, prompt injection, jailbreaks, output leakage.
- Hands-on exercises: add guardrail unit tests with adversarial prompts.
- Production scenarios: user asks the app to reveal system prompts.
- Common mistakes: regex-only guardrails without tests.
- Interview focus: layered defense for RAG.
- Recommended resources: OWASP LLM Top 10, NVIDIA NeMo Guardrails.
- Estimated time: 10-14 hours.

### 16. Multi-Tenant Security
- Why it matters: enterprise RAG must isolate user/team data.
- Core concepts: namespaces, ACL filters, metadata authorization, audit logs.
- Hands-on exercises: add `tenant_id` metadata and enforce it in retrieval.
- Production scenarios: finance documents must not be retrieved by HR users.
- Common mistakes: relying on UI filters instead of backend filters.
- Interview focus: data leakage prevention.
- Recommended resources: Pinecone namespace docs, OWASP access control guidance.
- Estimated time: 8-12 hours.

## Stage 7: Evaluation and Operations

### 17. RAG Evaluation
- Why it matters: production RAG needs regression checks.
- Core concepts: retrieval recall, faithfulness, answer correctness, latency.
- Hands-on exercises: build a golden QA set and evaluate after retrieval changes.
- Production scenarios: new chunking strategy improves recall but hurts precision.
- Common mistakes: evaluating only final answer text.
- Interview focus: metrics for retrieval vs generation.
- Recommended resources: RAGAS, DeepEval, TruLens.
- Estimated time: 10-16 hours.

### 18. Observability
- Why it matters: failures must be diagnosable in production.
- Core concepts: traces, spans, prompt logging, token usage, cost tracking.
- Hands-on exercises: add Langfuse traces around retrieval and generation.
- Production scenarios: latency spikes after reranking is enabled.
- Common mistakes: logging secrets or PII.
- Interview focus: what to log in RAG systems.
- Recommended resources: Langfuse docs, OpenTelemetry docs.
- Estimated time: 8-12 hours.

### 19. Scalable Ingestion
- Why it matters: large corpora need async and fault-tolerant ingestion.
- Core concepts: queues, workers, batching, retries, dead-letter queues.
- Hands-on exercises: move indexing into a queue-backed worker.
- Production scenarios: 1000 documents arrive at once.
- Common mistakes: embedding documents one-by-one without batching.
- Interview focus: design ingestion at scale.
- Recommended resources: Celery/RQ docs, Ray docs, Kafka basics.
- Estimated time: 12-20 hours.

## 100 Production RAG Interview Questions

1. How would you choose chunk size? Outline: balance semantic completeness, retrieval precision, and context budget.
2. When should CSV/JSON not be recursively chunked? Outline: preserve row/object semantics.
3. What metadata is essential for citation? Outline: source, page, chunk ID, document ID.
4. How do you avoid duplicate vectors? Outline: stable IDs, file hashes, upsert/delete old chunks.
5. What happens if embedding model changes? Outline: reindex all vectors because embedding space changed.
6. Chroma vs Pinecone trade-off? Outline: local simplicity vs managed scale and filtering.
7. Why use MMR? Outline: improve diversity among retrieved chunks.
8. How do you tune MMR lambda? Outline: higher relevance vs higher diversity.
9. Why hybrid search? Outline: combine exact keyword and semantic matching.
10. What is RRF? Outline: rank fusion using reciprocal rank scores.
11. Why rerank Top 20 to Top 5? Outline: broad recall first, precise ordering second.
12. Cross-encoder vs bi-encoder? Outline: cross-encoder is slower but more accurate.
13. How do you mitigate lost-in-the-middle? Outline: reorder key chunks near prompt edges.
14. How do you enforce citations? Outline: structured claims and validated source/page citations.
15. What is hallucination in RAG? Outline: answer unsupported by retrieved context.
16. How should RAG abstain? Outline: return insufficient-information response.
17. How do you test citation hallucination? Outline: fake invalid citations and assert blocking.
18. Why not trust model-provided citations? Outline: models can invent citations.
19. What is query expansion? Outline: generate alternate queries for recall.
20. What is query decomposition? Outline: split complex questions into focused subquestions.
21. When should expansion be disabled? Outline: cost-sensitive or simple queries.
22. How do you route LLMs? Outline: classify task and choose configured model.
23. Why use LiteLLM gateway? Outline: unified providers, fallback, routing, retries.
24. How do you handle provider quota errors? Outline: fallback chain and key rotation.
25. How do you rate-limit LLM calls? Outline: per-window counters or Redis.
26. How do you design cache keys? Outline: include prompt, model, params, response format.
27. What should not be cached? Outline: private or user-specific sensitive responses.
28. How do you reduce cost? Outline: caching, smaller context, routing, cheaper models.
29. How do you secure `.env`? Outline: ignore in Git and use secret managers.
30. What is prompt injection? Outline: user attempts to override system/developer instructions.
31. How do guardrails help? Outline: sanitize/block input and validate output.
32. Why output guardrails? Outline: models may leak prompts or answer unsafely.
33. What is PII redaction? Outline: replace sensitive patterns before processing/logging.
34. How do you prevent tenant data leakage? Outline: enforce tenant filters server-side.
35. How do you validate startup config? Outline: fail fast for missing providers/keys.
36. What should be logged? Outline: request IDs, timings, retrieval counts, model, not secrets.
37. What is retrieval recall? Outline: fraction of relevant evidence retrieved.
38. What is answer faithfulness? Outline: answer supported by context.
39. How do you evaluate RAG? Outline: golden questions plus retrieval and generation metrics.
40. What is a RAG regression test? Outline: test old questions after pipeline changes.
41. How do you handle changed documents? Outline: detect hash changes, delete old chunks, reindex.
42. How do you handle deleted documents? Outline: manifest lifecycle and vector deletes.
43. Why stable chunk IDs? Outline: deterministic updates and citations.
44. What is parent-child retrieval? Outline: retrieve children, return parent context.
45. When use parent-child retrieval? Outline: small chunks need surrounding context.
46. How do you handle tables? Outline: preserve structure or convert to row documents.
47. How do you handle PDFs with bad extraction? Outline: OCR/parser fallback and validation.
48. What is semantic cache? Outline: cache by meaning rather than exact prompt.
49. What is context compression? Outline: reduce context while preserving evidence.
50. What are context window risks? Outline: truncation and lost evidence.
51. How do you choose Top-K? Outline: balance recall, precision, token budget.
52. Why retrieve more than Top-K for reranking? Outline: reranker needs candidate pool.
53. What is ANN? Outline: approximate vector search for scalable nearest neighbors.
54. What is HNSW? Outline: graph-based ANN index.
55. Do you implement ANN yourself? Outline: usually configure vector DB index.
56. How do filters affect vector search? Outline: restrict candidate corpus before/inside search.
57. What if filters return no chunks? Outline: abstain or broaden filter with user consent.
58. How do you debug poor answers? Outline: inspect query, retrieved chunks, prompt, citations.
59. How do you debug high latency? Outline: measure retrieval, reranking, LLM, network.
60. How do you batch ingestion? Outline: batch embeddings and vector upserts.
61. Why async ingestion? Outline: throughput and non-blocking processing.
62. What is a dead-letter queue? Outline: store failed ingestion jobs for retry/review.
63. How do you handle huge documents? Outline: streaming parse, chunk, batch embed.
64. What is ACL-aware retrieval? Outline: retrieval filters based on user permissions.
65. How would you design namespaces? Outline: tenant/project/user-level partitioning.
66. How do you prevent prompt leakage? Outline: output checks and strict refusal.
67. What if the LLM returns non-JSON? Outline: parser fallback, retry, or abstain.
68. Why structured outputs? Outline: machine-validated answer/citation schema.
69. How do you handle model fallback quality changes? Outline: test fallback outputs and config policies.
70. How do you choose Gemini/Groq/Cerebras? Outline: latency, quality, cost, quota.
71. What is model routing by intent? Outline: route code/summary/general queries differently.
72. How do you protect logs from PII? Outline: redact before logging and limit prompt logs.
73. What are RAG antipatterns? Outline: no citations, blind stuffing, no tests, no filters.
74. How do you make RAG explainable? Outline: source chunks, citations, retrieval diagnostics.
75. What is a golden dataset? Outline: curated questions, expected evidence, expected answers.
76. What is retrieval precision? Outline: proportion of retrieved chunks that are relevant.
77. How do you tune BM25? Outline: tokenization, stopwords, field boosts, corpus index.
78. Why local BM25 over dense candidates is limited? Outline: not full corpus lexical recall.
79. How to improve current hybrid search? Outline: persistent full corpus BM25 index.
80. How do you handle multilingual docs? Outline: multilingual embeddings and language-aware parsing.
81. How do you handle versioned policies? Outline: metadata version/date filters.
82. How do you support freshness? Outline: incremental sync and timestamp filtering.
83. What is a retrieval confidence score? Outline: score based on distances/rerank/citations.
84. How do you surface uncertainty? Outline: abstention and confidence indicators.
85. How do you avoid over-retrieval? Outline: thresholds, reranking, compression.
86. How do you avoid under-retrieval? Outline: expansion, decomposition, hybrid search.
87. How do you test guardrails? Outline: adversarial prompt suite.
88. How do you handle self-harm queries? Outline: policy guardrail and safe response.
89. How do you use LangChain responsibly? Outline: use components, not opaque orchestration.
90. Why custom RAGService? Outline: clear control of pipeline and testability.
91. What is dependency injection in this repo? Outline: service receives retriever, builders, gateway.
92. How do you replace Pinecone with Chroma? Outline: config switch via factory.
93. How do you add a new vector DB? Outline: implement `VectorStore` interface.
94. How do you add a new LLM provider? Outline: configure LiteLLM route and keys.
95. How do you add new metadata filters? Outline: index metadata and pass filter dict.
96. What makes RAG production-grade? Outline: reliability, evaluation, security, observability, deployment.
97. What is missing for enterprise readiness here? Outline: API, evals, observability, CI, namespaces.
98. How would you deploy this app? Outline: container, secrets, persistent data, managed vector DB.
99. How would you pitch this project? Outline: modular production RAG with gateway, retrieval, guardrails.
100. What would you improve first? Outline: tests, `.env.example`, full BM25, parent-child retrieval.
