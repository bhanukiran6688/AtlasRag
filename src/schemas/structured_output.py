"""Pydantic schemas for structured LLM output in RAG responses.

This module provides type-safe schemas for parsing and validating structured
LLM responses, replacing fragile string-based JSON construction with proper
Pydantic models that include validation, serialization, and deserialization.
"""

from typing import Any
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Represents a citation to a source document.

    Citations link factual claims to their source documents, enabling
    traceability and verification of RAG-generated answers.
    """

    source: str = Field(description="Source filename or path")
    page: str = Field(description="Page number or 'Unknown'")


class Claim(BaseModel):
    """Represents a factual claim with supporting citations.

    Claims break down answers into verifiable factual statements, each
    backed by one or more citations to source documents.
    """

    text: str = Field(description="Single factual statement")
    citations: list[Citation] = Field(
        default_factory=list, description="List of citations supporting this claim"
    )


class StructuredRAGOutput(BaseModel):
    """Structured output schema for RAG responses with answer, claims, and citations.

    This schema enforces the structure of LLM responses, ensuring that:
    - Answers are based on provided context
    - Factual statements are backed by citations
    - Claims can be verified against source documents

    The schema is used both for prompt construction (showing the LLM the expected format)
    and for response parsing (validating the LLM's output).
    """

    answer: str = Field(description="Clear answer based only on the provided context")
    claims: list[Claim] = Field(
        default_factory=list, description="List of factual claims with citations"
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="List of all citations referenced in the answer",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert the structured output to a dictionary for JSON serialization.

        This is used when constructing prompts to show the LLM the expected output format.
        """
        return {
            "answer": self.answer,
            "claims": [
                {
                    "text": claim.text,
                    "citations": [c.model_dump() for c in claim.citations],
                }
                for claim in self.claims
            ],
            "citations": [citation.model_dump() for citation in self.citations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StructuredRAGOutput":
        """Create a StructuredRAGOutput from a dictionary.

        This is used when parsing LLM responses, converting raw JSON into
        type-safe Pydantic models with validation.
        """
        claims_data = data.get("claims", [])
        claims = [
            Claim(
                text=claim.get("text", ""),
                citations=[Citation(**cit) for cit in claim.get("citations", [])],
            )
            for claim in claims_data
        ]

        citations_data = data.get("citations", [])
        citations = [Citation(**cit) for cit in citations_data]

        return cls(
            answer=data.get("answer", ""),
            claims=claims,
            citations=citations,
        )
