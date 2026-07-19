"""Pydantic schemas for structured LLM output in RAG responses."""
from typing import Any
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Represents a citation to a source document."""
    source: str = Field(description="Source filename or path")
    page: str = Field(description="Page number or 'Unknown'")


class Claim(BaseModel):
    """Represents a factual claim with supporting citations."""
    text: str = Field(description="Single factual statement")
    citations: list[Citation] = Field(default_factory=list, description="List of citations supporting this claim")


class StructuredRAGOutput(BaseModel):
    """Structured output schema for RAG responses with answer, claims, and citations."""
    answer: str = Field(description="Clear answer based only on the provided context")
    claims: list[Claim] = Field(default_factory=list, description="List of factual claims with citations")
    citations: list[Citation] = Field(default_factory=list, description="List of all citations referenced in the answer")

    def to_dict(self) -> dict[str, Any]:
        """Convert the structured output to a dictionary."""
        return {
            "answer": self.answer,
            "claims": [
                {"text": claim.text, "citations": [c.model_dump() for c in claim.citations]}
                for claim in self.claims
            ],
            "citations": [citation.model_dump() for citation in self.citations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StructuredRAGOutput":
        """Create a StructuredRAGOutput from a dictionary."""
        claims_data = data.get("claims", [])
        claims = [
            Claim(
                text=claim.get("text", ""),
                citations=[Citation(**cit) for cit in claim.get("citations", [])]
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
