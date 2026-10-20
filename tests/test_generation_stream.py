import pytest

from app.rag.generation.base import BaseGenerationProvider


class FakeStreamingModel:
    async def astream(self, prompt):
        for chunk in ("I", "’m", " sorry", ", but", " I", " can help."):
            yield chunk


class Provider(BaseGenerationProvider):
    def _build_model(self, config):
        return FakeStreamingModel()

    async def ainvoke(self, prompt):
        return "fallback"


@pytest.mark.anyio
async def test_streaming_preserves_chunk_whitespace():
    provider = Provider(config=type("Config", (), {"model": "test"})())

    result = "".join([chunk async for chunk in provider.astream("prompt")])

    assert result == "I’m sorry, but I can help."


class FakeGeminiChunk:
    def __init__(self, content):
        self.content = content


class FakeGeminiStreamingModel:
    async def astream(self, prompt):
        yield FakeGeminiChunk([{"type": "text", "text": "Based on the context, ", "index": 0}])
        yield FakeGeminiChunk([{"type": "text", "text": "Sales department paid.", "index": 0}])
        yield FakeGeminiChunk([{"type": "text", "text": "", "extras": {"signature": "abc"}, "index": 0}])
        yield FakeGeminiChunk([])


class GeminiTestProvider(BaseGenerationProvider):
    def _build_model(self, config):
        return FakeGeminiStreamingModel()

    async def ainvoke(self, prompt):
        return "fallback"


@pytest.mark.anyio
async def test_gemini_structured_block_content_streaming():
    provider = GeminiTestProvider(config=type("Config", (), {"model": "gemini-2.0-flash"})())
    chunks = [chunk async for chunk in provider.astream("prompt")]
    assert chunks == ["Based on the context, ", "Sales department paid."]
    assert "".join(chunks) == "Based on the context, Sales department paid."
