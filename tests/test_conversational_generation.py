from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.rag.generation.base import BaseGenerationProvider
from app.rag.memory.context import (
    CONVERSATIONAL_SYSTEM_PROMPT,
    FOLLOWUP_SUGGESTIONS_PROMPT,
    STRICT_SYSTEM_PROMPT,
    build_generation_prompt,
)
from app.rag.pipeline.query import QueryPipeline
from app.schemas.thread import GenerationConfig, ThreadConfig


class MockRecordingProvider(BaseGenerationProvider):
    provider_name = "test"

    def __init__(self):
        self.prompts = []

    def _build_model(self, config):
        return None

    async def ainvoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "conversational response"

    async def astream(self, prompt: str):
        self.prompts.append(prompt)
        yield "conversational "
        yield "response"


def test_generation_config_chat_mode_defaults_and_validation():
    config = GenerationConfig()
    assert config.chat_mode == "conversational"
    assert config.suggest_followups is True
    assert config.system_prompt is None

    strict_config = GenerationConfig(chat_mode="strict", suggest_followups=False, system_prompt="Custom persona")
    assert strict_config.chat_mode == "strict"
    assert strict_config.suggest_followups is False
    assert strict_config.system_prompt == "Custom persona"


def test_build_generation_prompt_conversational_mode():
    prompt = build_generation_prompt(
        query="Tell me about project goals",
        context="Doc context here",
        chat_mode="conversational",
        suggest_followups=True,
    )
    assert CONVERSATIONAL_SYSTEM_PROMPT in prompt
    assert FOLLOWUP_SUGGESTIONS_PROMPT in prompt
    assert "Query: Tell me about project goals" in prompt
    assert "Context: Doc context here" in prompt


def test_build_generation_prompt_strict_mode():
    prompt = build_generation_prompt(
        query="What is the port?",
        context="Doc context here",
        chat_mode="strict",
    )
    assert STRICT_SYSTEM_PROMPT in prompt
    assert FOLLOWUP_SUGGESTIONS_PROMPT not in prompt
    assert "Query: What is the port?" in prompt


def test_build_generation_prompt_custom_override():
    custom_system = "You are a concise pirate assistant."
    prompt = build_generation_prompt(
        query="Where is the treasure?",
        context="Doc context here",
        system_prompt=custom_system,
    )
    assert custom_system in prompt
    assert CONVERSATIONAL_SYSTEM_PROMPT not in prompt
    assert STRICT_SYSTEM_PROMPT not in prompt


@pytest.mark.anyio
async def test_query_pipeline_passes_conversational_config_to_prompt():
    provider = MockRecordingProvider()
    thread = MagicMock(id=uuid4(), knowledge_bases=[])
    message = MagicMock(content="Explain the architecture", sequence=1)
    db = AsyncMock()
    db.scalars.return_value = MagicMock()
    db.scalars.return_value.all.return_value = []

    thread_config = ThreadConfig(
        generation=GenerationConfig(
            chat_mode="conversational",
            suggest_followups=True,
        )
    )

    pipeline = QueryPipeline(
        db=db,
        config=thread_config,
        run=MagicMock(),
        thread=thread,
        message=message,
        provider=provider,
    )

    async def memory(**kwargs):
        return []

    pipeline.memory_manager.load = memory

    events = [e async for e in pipeline.execute_stream()]
    complete_event = next(e for e in events if e["type"] == "complete")
    assert complete_event["result"]["response"] == "conversational response"

    assert len(provider.prompts) > 0
    generated_prompt = provider.prompts[0]
    assert CONVERSATIONAL_SYSTEM_PROMPT in generated_prompt
    assert FOLLOWUP_SUGGESTIONS_PROMPT in generated_prompt
    assert "Explain the architecture" in generated_prompt
