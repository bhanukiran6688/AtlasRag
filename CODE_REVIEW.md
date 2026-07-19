# Staff Engineer Code Review

This review documents findings and completed improvements.

## Executive Summary

The repository has a strong modular RAG architecture for a portfolio project: ingestion, embeddings, vector stores, retrieval, gateway, guardrails, context building, and UI are separated.

## Completed Improvements

The following items from the original review have been addressed:

- **Structured output with Pydantic schema**: Replaced string-built JSON with type-safe Pydantic models for better validation and maintainability
- **Query expansion/decomposition caching**: Added in-memory caching to reduce LLM calls and costs
- **Token-based conversation history limiting**: Implemented accurate token-based limiting instead of turn-based for better prompt cost control
- **Improved BM25 tokenization**: Added NLTK support with stemming and stopword filtering for better lexical retrieval
- **Normalized score semantics**: Added normalized_score field (0-1 range) for consistent score interpretation across retrieval strategies
- **Tests for insufficient-information responses**: Added test cases for abstention behavior and unsupported claim rejection
- **Metadata reindex documentation**: Created comprehensive guide for handling metadata schema changes