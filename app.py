import warnings
import streamlit as st

from src.utils.logger import get_logger
from src.utils.exceptions import RAGConfigurationError
from src.utils.metadata_filters import (
    build_metadata_filter as build_validated_metadata_filter,
    parse_metadata_filter
)
from src.config.settings import settings
from src.services.factory import create_rag_service

logger = get_logger(__name__)

warnings.filterwarnings(
    action="ignore",
    message=r".*Accessing `__path__`.*"
)


# Build the RAG service and shared retrieval/LLM dependencies once per app session.
@st.cache_resource
def initialize_rag():
    return create_rag_service()


# Render the Streamlit RAG playground and route user options into the backend pipeline.
def main() -> None:
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    st.set_page_config(page_title="RAG Playground", page_icon="🔍", layout="wide")
    st.title("🔍 RAG Playground")
    st.caption("Phase 2 - Retrieval & Context Builder")
    query = (
        st.text_input(
            label="Ask a question",
            placeholder="Example: What is Redis?",
        )
        or ""
    )
    use_query_expansion = st.checkbox(
        label="Use query expansion",
        value=False,
        help="Generate alternative search queries before retrieval. Uses an extra LLM call.",
    )
    use_query_decomposition = st.checkbox(
        label="Use query decomposition",
        value=False,
        help="Break complex questions into smaller retrieval questions. Uses an extra LLM call.",
    )
    with st.expander("Conversation Memory", expanded=False):
        if st.session_state.conversation_history:
            for turn in st.session_state.conversation_history:
                st.write(f"{turn['role'].title()}: {turn['content']}")
        else:
            st.caption("No prior turns yet.")

    with st.expander("Metadata Filters", expanded=False):
        filter_source = st.text_input(
            label="Source", placeholder="Exact source metadata value"
        )
        filter_file_type = st.text_input(
            label="File type", placeholder="pdf, md, txt, csv, json, docx"
        )
        filter_metadata_json = st.text_area(
            label="Custom metadata JSON",
            placeholder='{"department": "finance"}',
            height=100,
        )
    if not st.button("Search", type="primary"):
        return

    if not query.strip():
        st.warning("Please enter a question.")
        return

    try:
        rag_service = initialize_rag()
    except (RAGConfigurationError, RuntimeError, ValueError) as exc:
        st.error(f"RAG initialization failed: {exc}")
        return

    metadata_filter = build_metadata_filter(
        source=filter_source,
        file_type=filter_file_type,
        custom_metadata_json=filter_metadata_json,
    )
    with st.spinner("Searching and generating answer..."):
        result = rag_service.process(
            query,
            use_query_expansion=use_query_expansion,
            use_query_decomposition=use_query_decomposition,
            metadata_filter=metadata_filter,
            conversation_history=st.session_state.conversation_history,
        )

    # FEATURE: Conversation memory support
    st.session_state.conversation_history = rag_service.update_conversation_history(
        st.session_state.conversation_history,
        query,
        result.answer,
    )
    retriever = rag_service.retriever
    if result.is_blocked:
        st.error(result.answer)
        if result.pii_detected:
            st.info(f"PII redacted before blocking: {', '.join(result.pii_detected)}")
        return

    if result.sanitized_question and result.sanitized_question != result.question:
        st.info("Sensitive information was redacted before processing the question.")
        st.text_area(
            label="Sanitized Question", value=result.sanitized_question, height=100
        )

    if result.pii_detected:
        st.caption(f"PII redacted: {', '.join(result.pii_detected)}")

    if result.retrieval_queries:
        with st.expander("Retrieval Queries", expanded=False):
            for retrieval_query in result.retrieval_queries:
                st.write(retrieval_query)

    st.subheader("Answer")
    if result.answer:
        st.write(result.answer)
    else:
        st.warning("No answer was generated.")

    if result.llm_model:
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        col1.metric(label="Model", value=result.llm_model)
        col2.metric(label="LLM Latency", value=f"{result.llm_latency_ms:.2f} ms")
        col3.metric(label="Tokens", value=result.total_tokens)
        col4.metric(
            label="Cost",
            value=f"${result.cost_usd:.6f}" if result.cost_usd is not None else "n/a",
        )
        col5.metric(label="Cache", value="Hit" if result.cache_hit else "Miss")
        col6.metric(label="Route", value=result.route or "general")
        # FEATURE: Grounding confidence visibility
        col7.metric(
            label="Grounding Confidence",
            value=(
                f"{result.grounding_confidence:.3f}"
                if result.grounding_confidence is not None
                else "n/a"
            )
        )

    if result.error:
        st.error(result.error)

    if result.structured_output and result.structured_output.get("citations"):
        st.subheader("Citations")
        st.json(result.structured_output["citations"])

    st.subheader("Retrieval Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        label="Retrieval Time",
        value=f"{retriever.last_retrieval_time_ms:.2f} ms",
    )
    col2.metric(label="Top-K", value=retriever.k)
    col3.metric(
        label="Threshold",
        value=retriever.score_threshold
    )
    col4.metric(
        label="Returned",
        value=f"{retriever.last_returned_results}/{retriever.last_total_results}"
    )
    results = result.retrieved_chunks
    if not results:
        st.warning("No relevant chunks found.")
        return

    st.subheader("Retrieved Chunks")
    for chunk in results:
        with st.expander(
            label=f"Rank {chunk.rank} | Score {chunk.distance:.4f}", expanded=False
        ):
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Distance", value=f"{chunk.distance:.4f}")
                st.metric(label="Chunk Length", value=chunk.chunk_length)
                st.metric(label="Strategy", value=chunk.retrieval_strategy)

            with col2:
                st.metric(label="Source", value=chunk.source)
                st.metric(
                    label="Page", value=chunk.page if chunk.page is not None else "-"
                )
                if chunk.rerank_score is not None:
                    st.metric(label="Rerank Score", value=f"{chunk.rerank_score:.4f}")

            st.markdown("### Metadata")
            st.json(chunk.document.metadata)
            st.markdown("### Content")
            st.write(chunk.document.page_content)

    context = result.context
    if not context:
        st.warning("No usable context could be built from the retrieved documents.")
        st.stop()

    st.subheader("Built Context")
    st.text_area(label="Context", value=context, height=350)
    st.subheader("Final Prompt")
    st.text_area(label="Prompt", value=result.prompt, height=400)


# Convert optional Streamlit metadata fields into a backend retrieval filter.
def build_metadata_filter(
    source: str,
    file_type: str,
    custom_metadata_json: str,
) -> dict | None:
    custom_filter = {}
    if custom_metadata_json.strip():
        try:
            custom_filter = parse_metadata_filter(custom_metadata_json)
        except ValueError as exc:
            st.warning(f"Custom metadata filter was ignored: {exc}")
    try:
        return build_validated_metadata_filter(
            source=source,
            file_type=file_type,
            custom_metadata=custom_filter,
            allowed_keys=settings.allowed_metadata_filter_keys or None,
        )
    except ValueError as exc:
        st.warning(f"Metadata filter was ignored: {exc}")
        return None


if __name__ == "__main__":
    main()
