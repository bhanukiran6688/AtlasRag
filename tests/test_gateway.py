import pytest
from typing import Mapping, Any
from src.llm.base import LLMGateway, LLMResponse, LLMUsage

class MockLLMGateway(LLMGateway):
    def __init__(self):
        self.responses = []
        self.last_prompt = None

    def validate_connection(self) -> None:
        pass

    async def generate(
        self,
        prompt: str,
        *,
        routing_text: str | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        self.last_prompt = prompt
        if self.responses:
            return self.responses.pop(0)
        
        return LLMResponse(
            content='{"answer": "Mocked answer", "citations": [], "claims": []}',
            model="mock-model",
            latency_ms=10.0,
            usage=LLMUsage(total_tokens=10, cost_usd=0.001)
        )

@pytest.mark.asyncio
async def test_mock_gateway():
    gateway = MockLLMGateway()
    response = await gateway.generate("test prompt")
    assert "Mocked answer" in response.content
    assert gateway.last_prompt == "test prompt"
