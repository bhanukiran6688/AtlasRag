import re
from typing import Any
from dataclasses import dataclass, field

from src.retrievers.retriever import RetrievalResult
from src.guardrails.claim_support import ClaimSupportVerifier
from src.config.settings import settings
from src.utils.citations import filter_valid_citations
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class GuardrailResult:
    original_text: str
    sanitized_text: str
    is_blocked: bool = False
    blocked_reason: str | None = None
    pii_detected: list[str] = field(default_factory=list)


@dataclass(slots=True)
class OutputGuardrailResult:
    answer: str
    citations: list[dict[str, str]]
    is_blocked: bool = False
    blocked_reason: str | None = None
    is_grounded: bool = True
    grounding_reason: str | None = None


class InputGuardrails:
    """
    Validates and sanitizes user input before retrieval and LLM calls.

    This layer is intentionally independent of LiteLLM so the same rules
    protect every downstream model provider.
    """

    _pii_patterns = {
        "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        "PHONE_IN": re.compile(r"(\+91[-\s]?)?[6-9]\d{9}"),
        "PHONE_US": re.compile(r"(\+1[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}"),
        "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "AADHAAR": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
        "PAN": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
        "CREDIT_CARD": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
        "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    }

    # FEATURE: Stronger jailbreak protection
    _injection_patterns = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"ignore (all |the )?(previous|prior|above) (instructions?|prompts?|rules?)",
            r"disregard (the |all )?(previous|prior|earlier)",
            r"forget (everything|your instructions?|the rules?)",
            r"you are (now |a )?(DAN|jailbroken|unrestricted|unfiltered)",
            r"pretend (you are|to be) .{0,40}(no restrictions?|uncensored)",
            r"</?(system|user|assistant|im_start|im_end)>",
            r"new (instructions?|system prompt|rules?):",
            r"reveal your (system )?prompt",
            r"what (are|were) your (original )?instructions?",
            r"act as if you (have|are) no restrictions",
            r"bypass (the |all )?(safety|security|guardrails|filters)",
            r"show me the hidden (prompt|instructions?)",
        )
    ]

    _normalized_injection_patterns = (
        "ignorepreviousinstructions",
        "ignoreallpreviousinstructions",
        "disregardpreviousinstructions",
        "forgetyourinstructions",
        "revealyoursystemprompt",
        "revealtheprompt",
        "youarenowdan",
        "jailbroken",
    )

    _forbidden_topics = (
        "weapon",
        "bomb",
        "explosive",
        "hack",
        "exploit",
        "malware",
        "illegal substance",
        "self-harm",
        "suicide",
    )

    def validate(self, text: str) -> GuardrailResult:
        sanitized_text, pii_detected = self._redact_pii(text)

        injection_reason = self._find_prompt_injection(sanitized_text)
        if injection_reason:
            logger.warning("Blocked prompt injection attempt: %s", injection_reason)
            return GuardrailResult(
                original_text=text,
                sanitized_text=sanitized_text,
                is_blocked=True,
                blocked_reason="Prompt injection attempt detected.",
                pii_detected=pii_detected,
            )

        forbidden_topic = self._find_forbidden_topic(sanitized_text)
        if forbidden_topic:
            logger.warning("Blocked forbidden topic: %s", forbidden_topic)
            return GuardrailResult(
                original_text=text,
                sanitized_text=sanitized_text,
                is_blocked=True,
                blocked_reason=f"Forbidden topic detected: {forbidden_topic}.",
                pii_detected=pii_detected,
            )

        if pii_detected:
            logger.info(
                "Redacted PII types before processing: %s", ", ".join(pii_detected)
            )

        return GuardrailResult(
            original_text=text,
            sanitized_text=sanitized_text,
            pii_detected=pii_detected,
        )

    # Replace detected PII with placeholders before retrieval or LLM calls.
    def _redact_pii(self, text: str) -> tuple[str, list[str]]:
        sanitized_text = text
        detected: list[str] = []

        for label, pattern in self._pii_patterns.items():
            sanitized_text, count = pattern.subn(f"<{label}_REDACTED>", sanitized_text)
            if count:
                detected.append(label)

        return sanitized_text, detected

    # Detect prompt injection and jailbreak phrases in raw or normalized text.
    def _find_prompt_injection(self, text: str) -> str | None:
        for pattern in self._injection_patterns:
            if pattern.search(text):
                return pattern.pattern

        normalized_text = re.sub(r"[^a-zA-Z0-9]", "", text).lower()
        for pattern in self._normalized_injection_patterns:
            if pattern in normalized_text:
                return pattern

        return None

    # Detect disallowed user intents before they reach retrieval or generation.
    def _find_forbidden_topic(self, text: str) -> str | None:
        text_lower = text.lower()
        for topic in self._forbidden_topics:
            if topic in text_lower:
                return topic
        return None


class OutputGuardrails:
    """
    Validates model output before it is returned to the user.

    This is a second safety layer: input checks stop risky requests, output
    checks stop prompt leakage and reduce unsupported answers.

    RAG Concept: Semantic Hallucination Mitigation
    - Lexical checks: match exact terms in claims to context
    - Semantic checks: use LLM to understand meaning and detect paraphrases
    - Combined approach: faster lexical check first, semantic check for high-stakes scenarios
    """

    _leakage_patterns = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"system prompt",
            r"developer message",
            r"hidden instructions?",
            r"internal instructions?",
            r"confidential instructions?",
            r"i was instructed to",
        )
    ]

    _insufficient_answer_markers = (
        "i don't have enough information",
        "not present in the context",
        "provided documents do not",
        "not enough information",
    )

    def __init__(
        self,
        claim_support_verifier: ClaimSupportVerifier | None = None,
        semantic_verifier: Any = None,
    ) -> None:
        self._claim_support_verifier = claim_support_verifier or ClaimSupportVerifier()
        self._semantic_verifier = semantic_verifier

    async def validate(
        self,
        answer: str,
        citations: list[dict[str, str]],
        retrieved_chunks: list[RetrievalResult],
        claims: list[dict[str, object]] | None = None,
        context: str = "",
    ) -> OutputGuardrailResult:
        leakage_reason = self._find_prompt_leakage(answer)
        if leakage_reason:
            logger.warning("Blocked prompt leakage in LLM output: %s", leakage_reason)
            return OutputGuardrailResult(
                answer="I cannot reveal internal instructions or hidden prompts.",
                citations=[],
                is_blocked=True,
                blocked_reason="Prompt leakage detected in model output.",
                is_grounded=False,
                grounding_reason="Output contained internal-instruction leakage.",
            )

        valid_citations = filter_valid_citations(citations, retrieved_chunks)
        if self._is_insufficient_answer(answer):
            return OutputGuardrailResult(answer=answer, citations=valid_citations)

        claim_error = self._validate_claims(
            claims=claims, retrieved_chunks=retrieved_chunks
        )
        if claim_error:
            logger.warning(
                "Blocked answer because claim citation enforcement failed: %s",
                claim_error,
            )
            return OutputGuardrailResult(
                answer="I don't have enough information in the provided documents.",
                citations=[],
                is_grounded=False,
                grounding_reason=claim_error,
            )

        # RAG Concept: Semantic Verification
        # If semantic verifier is enabled and claims passed lexical checks,
        # perform additional semantic verification for higher assurance
        if (
            self._semantic_verifier
            and settings.guardrails_enable_semantic_verification
            and claims
        ):
            all_supported, verified_claims = await self._semantic_verifier.verify_answer(
                answer=answer, claims=claims, context=context
            )
            if not all_supported:
                unsupported_claims = [
                    c["text"] for c in verified_claims if not c["is_supported"]
                ]
                logger.warning(
                    "Semantic verification failed for %d claims: %s",
                    len(unsupported_claims),
                    unsupported_claims[:2],
                )
                return OutputGuardrailResult(
                    answer="I don't have enough information in the provided documents.",
                    citations=[],
                    is_grounded=False,
                    grounding_reason="Semantic verification found unsupported claims.",
                )

        if not valid_citations:
            logger.warning("Blocked ungrounded answer without valid citations.")
            return OutputGuardrailResult(
                answer="I don't have enough information in the provided documents.",
                citations=[],
                is_grounded=False,
                grounding_reason="Answer did not include a valid citation from retrieved context.",
            )

        return OutputGuardrailResult(answer=answer, citations=valid_citations)

    # Enforce that every factual claim has at least one retrieved citation.
    def _validate_claims(
        self,
        claims: list[dict[str, object]] | None,
        retrieved_chunks: list[RetrievalResult],
    ) -> str | None:
        if not claims:
            return "Answer did not include claim-level citations."

        for claim in claims:
            text = str(claim.get("text", "")).strip()
            citations = claim.get("citations", [])
            if not text:
                return "A factual claim was missing text."
            if not isinstance(citations, list):
                return f"Claim has invalid citations: {text}"

            normalized_citations = [
                {
                    "source": str(citation.get("source", "")),
                    "page": str(citation.get("page", "Unknown")),
                }
                for citation in citations
                if isinstance(citation, dict)
            ]
            if not filter_valid_citations(normalized_citations, retrieved_chunks):
                return f"Claim lacks valid retrieved citation: {text}"
            if (
                settings.guardrails_enable_claim_support_check
                and not self._claim_support_verifier.verify(
                    text,
                    normalized_citations,
                    retrieved_chunks,
                )
            ):
                return (
                    f"Claim is not sufficiently supported by its cited context: {text}"
                )

        return None

    # Detect model output that reveals hidden prompts or internal instructions.
    def _find_prompt_leakage(self, answer: str) -> str | None:
        for pattern in self._leakage_patterns:
            if pattern.search(answer):
                return pattern.pattern
        return None

    # Recognize abstention answers that should not require citations.
    def _is_insufficient_answer(self, answer: str) -> bool:
        answer_lower = answer.lower()
        return any(
            marker in answer_lower for marker in self._insufficient_answer_markers
        )
