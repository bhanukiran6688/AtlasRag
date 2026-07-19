import re

from src.config.settings import settings
from src.retrievers.retriever import RetrievalResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ContextBuilder:
    """
    Builds a formatted context string from retrieved chunks.
    """

    @staticmethod
    def build(results: list[RetrievalResult]) -> str:
        """
        Build a context string for the LLM.
        """

        if not results:
            logger.info("No documents received for context building.")
            return ""

        sections: list[str] = []
        seen_chunks: set[str] = set()
        for result in results:
            chunk = result.document.page_content.strip()
            if chunk in seen_chunks:
                continue

            seen_chunks.add(chunk)
            page = result.page if result.page is not None else "Unknown"
            sections.append(
                f"Document {result.rank}\n"
                f"Source : {result.source}\n"
                f"Page   : {page}\n\n"
                f"{chunk}"
            )

        if not sections:
            logger.info("Context is empty after duplicate removal.")
            return ""

        context = (
            "Context\n"
            + "=" * 80
            + "\n\n"
            + ("\n\n" + "-" * 80 + "\n\n").join(sections)
        )

        # FEATURE: Context window management
        context = ContextBuilder._truncate_context(
            context, max_tokens=settings.max_context_tokens
        )

        # FEATURE: Lightweight context compression
        context = ContextBuilder._compress_context(context)

        logger.info("Built context from %d retrieved chunk(s).", len(results))
        return context

    # Trim context text to a rough token budget before prompt construction.
    @staticmethod
    def _truncate_context(context: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return context

        token_count = ContextBuilder._count_tokens(context)
        if token_count <= max_tokens:
            return context

        truncated_parts: list[str] = []
        current_tokens = 0
        for part in context.split("\n\n"):
            part_tokens = ContextBuilder._count_tokens(part)
            if current_tokens + part_tokens + 4 <= max_tokens:
                truncated_parts.append(part)
                current_tokens += part_tokens
            else:
                break

        if not truncated_parts:
            return " ".join(context.split()[:max_tokens])

        return (
            "\n\n".join(truncated_parts)
            + "\n\n... [context truncated for token budget]"
        )

    # Compress context lines by keeping metadata and shortening long text lines.
    @staticmethod
    def _compress_context(context: str) -> str:
        if not context:
            return context

        # FEATURE: Lightweight context compression
        lines = context.splitlines()
        compressed_lines: list[str] = []
        for line in lines:
            if (
                line.strip().startswith("Document ")
                or line.strip().startswith("Source")
                or line.strip().startswith("Page")
            ):
                compressed_lines.append(line)
            elif line.strip():
                compressed_text = ContextBuilder._compress_text(line)
                if compressed_text and (
                    not compressed_lines
                    or compressed_lines[-1].strip() != compressed_text
                ):
                    compressed_lines.append(compressed_text)

        return "\n".join(compressed_lines)

    # Keep the first few sentences of long context text for token efficiency.
    @staticmethod
    def _compress_text(text: str) -> str:
        sentence_parts = re.split(r"(?<=[.!?])\s+", text.strip())
        if not sentence_parts:
            return text.strip()

        compressed_sentences = sentence_parts[
            : settings.context_compression_max_sentences
        ]
        compressed_text = " ".join(compressed_sentences)
        return (
            compressed_text
            if len(compressed_text) < len(text.strip())
            else text.strip()
        )

    # Limit the number of context chunks injected into the final LLM prompt.
    @staticmethod
    def _trim_context_to_chunks(context: str, max_context_chunks: int) -> str:
        if max_context_chunks <= 0:
            return context

        chunks = [chunk.strip() for chunk in context.split("\n\n") if chunk.strip()]
        if len(chunks) <= max_context_chunks:
            return context

        return (
            "\n\n".join(chunks[:max_context_chunks])
            + "\n\n... [context trimmed for cost optimization]"
        )

    # Estimate prompt size with a lightweight whitespace token approximation.
    @staticmethod
    def _count_tokens(text: str) -> int:
        return len(text.split())


class PromptBuilder:
    """
    Builds the final prompt that is sent to the LLM.
    """

    @staticmethod
    def build(
        question: str,
        context: str,
        conversation_history: list[dict[str, str]] | None = None,
        max_context_chunks: int | None = None,
        simplify_prompt: bool = True,
    ) -> str:
        """
        Build the final prompt for the LLM.
        """

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if not context.strip():
            raise ValueError("Context cannot be empty.")

        history_block = ""
        if conversation_history:
            history_lines = [
                f"{turn['role'].title()}: {turn['content']}"
                for turn in conversation_history
                if turn.get("content")
            ]
            if history_lines:
                history_block = "\n".join(history_lines)

        if max_context_chunks is None:
            max_context_chunks = settings.cost_optimization_max_context_chunks

        if simplify_prompt:
            context = ContextBuilder._trim_context_to_chunks(
                context, max_context_chunks
            )

        prompt = f"""You are a helpful AI assistant.
        Answer the user's question using ONLY the provided context.
        
        Rules:
        - Do not use outside knowledge.
        - If the answer is not present in the context, reply:
          "I don't have enough information in the provided documents."
        - Keep the answer clear and concise.
        
        ===============================================================================
        Context
        
        {context}
        
        ===============================================================================
        Conversation History
        
        {history_block if history_block else 'No prior conversation history.'}
        
        ===============================================================================
        Question
        
        {question}
        
        ===============================================================================
        Answer
        """

        logger.info("Built prompt for LLM.")
        return prompt
