from src.embeddings.embedding_generator import EmbeddingGenerator
from src.llm.base import LLMGatewayFactory
from src.prompts.prompt_builder import ContextBuilder, PromptBuilder
from src.retrievers.retriever import Retriever
from src.services.rag_service import RAGService
from src.vectorstores.base import VectorStoreFactory


def create_rag_service() -> RAGService:
    """Create the RAG application service with its production dependencies."""
    embedding_generator = EmbeddingGenerator()
    vector_store = VectorStoreFactory.create(embeddings=embedding_generator.get_embeddings())
    vector_store.validate_connection()
    llm_gateway = LLMGatewayFactory.create()
    llm_gateway.validate_connection()
    return RAGService(
        retriever=Retriever(vector_store),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm_gateway=llm_gateway,
    )
