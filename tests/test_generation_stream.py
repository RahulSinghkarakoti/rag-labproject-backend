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
