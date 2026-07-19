"""Semantic verifier for hallucination mitigation.

This module provides semantic verification of claims against retrieved context,
going beyond lexical matching to detect paraphrases and logical contradictions.
"""

from typing import Any
from src.llm.base import LLMGateway, LLMResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SemanticVerifier:
    """
    Verifies that claims are semantically supported by retrieved context.

    RAG Concept: Semantic Hallucination Mitigation
    - Lexical checks only match exact terms, missing paraphrases
    - Semantic verification uses LLM to understand meaning
    - Detects contradictions even when terms don't match exactly
    - Higher assurance but adds latency and cost

    Use Cases:
    - User asks: "What is the return policy?"
    - Context: "Items can be returned within 30 days"
    - Claim: "The return period is one month"
    - Lexical check: FAILS (no exact term match)
    - Semantic check: PASSES (understands "30 days" ≈ "one month")
    """

    def __init__(self, llm_gateway: LLMGateway) -> None:
        """Initialize semantic verifier with LLM gateway."""
        self._llm_gateway = llm_gateway

    async def verify_claim(
        self, claim: str, context: str, citations: list[dict[str, str]]
    ) -> tuple[bool, str | None]:
        """
        Verify that a claim is semantically supported by the context.

        Args:
            claim: The factual claim to verify
            context: The retrieved context that should support the claim
            citations: Citations associated with the claim

        Returns:
            Tuple of (is_supported, reason) where:
            - is_supported: True if claim is semantically supported
            - reason: Explanation of why claim is/ isn't supported
        """
        if not context.strip():
            return False, "No context provided for verification"

        prompt = f"""You are a semantic verifier for a RAG system. Your task is to determine if a claim is supported by the provided context.

Context:
{context}

Claim:
{claim}

Citations:
{', '.join([f"{c.get('source', 'Unknown')}:{c.get('page', 'Unknown')}" for c in citations])}

Analyze whether the claim is semantically supported by the context. Consider:
- Direct evidence: Does the context explicitly state the claim?
- Paraphrases: Does the context state equivalent information using different words?
- Logical inference: Can the claim be reasonably inferred from the context?
- Contradictions: Does the context contradict the claim?

Return ONLY valid JSON with this format:
{{"is_supported": true/false, "reason": "brief explanation"}}"""

        try:
            response = await self._llm_gateway.generate(
                prompt=prompt,
                model=settings.llm_primary_model,
            )

            # Parse the verification result
            import json
            result = json.loads(response.content)
            is_supported = result.get("is_supported", False)
            reason = result.get("reason", "")

            logger.info(
                "Semantic verification for claim '%s': %s - %s",
                claim[:50],
                "SUPPORTED" if is_supported else "NOT SUPPORTED",
                reason,
            )

            return is_supported, reason

        except Exception as exc:
            logger.warning("Semantic verification failed: %s. Falling back to conservative.", exc)
            # Conservative fallback: reject claim if verification fails
            return False, f"Verification failed: {exc}"

    async def verify_answer(
        self, answer: str, claims: list[dict[str, Any]], context: str
    ) -> tuple[bool, list[dict[str, Any]]]:
        """
        Verify all claims in an answer against the context.

        Args:
            answer: The complete answer text
            claims: List of claims with their citations
            context: The retrieved context

        Returns:
            Tuple of (all_supported, verified_claims) where:
            - all_supported: True if all claims are semantically supported
            - verified_claims: Claims with verification results added
        """
        if not claims:
            # No claims to verify - this is acceptable for abstention responses
            return True, []

        verified_claims = []
        all_supported = True

        for claim in claims:
            claim_text = claim.get("text", "")
            claim_citations = claim.get("citations", [])

            is_supported, reason = await self.verify_claim(
                claim=claim_text,
                context=context,
                citations=claim_citations,
            )

            verified_claim = claim.copy()
            verified_claim["is_supported"] = is_supported
            verified_claim["verification_reason"] = reason
            verified_claims.append(verified_claim)

            if not is_supported:
                all_supported = False

        logger.info(
            "Semantic verification complete: %d/%d claims supported",
            sum(1 for c in verified_claims if c["is_supported"]),
            len(verified_claims),
        )

        return all_supported, verified_claims


# Import settings at module level to avoid circular imports
from src.config.settings import settings
