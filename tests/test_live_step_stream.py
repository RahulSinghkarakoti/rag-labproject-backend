from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.models.message import Message
from app.models.run import Run
from app.models.thread import Thread
from app.rag.pipeline.query import QueryPipeline
from app.schemas.thread import ThreadConfig


class FakeRunStep:
    def __init__(self, step_type, sequence, input_data=None, config=None):
        self.id = uuid4()
        self.run_id = uuid4()
        self.step_type = step_type
        self.sequence = sequence
        self.status = "running"
        self.input = input_data
        self.output = None
        self.config = config
        self.error = None
        self.started_at = None
        self.completed_at = None
        self.metadata_ = {}


class FakeRunManager:
    def __init__(self):
        self.steps = []
        self._seq = 0

    async def start_step(self, step_type: str, input_data=None, config=None):
        self._seq += 1
        step = FakeRunStep(step_type, self._seq, input_data, config)
        self.steps.append(step)
        return step

    async def complete_step(self, step: FakeRunStep, output_data=None, metadata=None):
        step.status = "completed"
        step.output = output_data
        step.metadata_ = metadata or {}
        return step

    async def fail_step(self, step: FakeRunStep, error_message: str):
        step.status = "failed"
        step.error = {"message": error_message}
        return step


class FakeStreamingProvider:
    provider_name = "test"
    config = SimpleNamespace(model="test-model")

    async def astream(self, prompt):
        yield "Hello "
        yield "world!"

    async def ainvoke_safe(self, prompt):
        return "Hello world!"


class FakeDB:
    def __init__(self):
        pass

    async def scalars(self, query):
        class Result:
            def all(self):
                return []
        return Result()


@pytest.mark.anyio
async def test_query_pipeline_execute_stream_emits_step_lifecycle_events():
    db = FakeDB()
    thread_id = uuid4()
    run = Run(thread_id=thread_id, run_type="chat")
    run.id = uuid4()
    thread = Thread(id=thread_id)
    message = Message(
        thread_id=thread_id,
        role="human",
        content="What is RAG?",
        sequence=1,
    )
    message.id = uuid4()

    run_manager = FakeRunManager()
    provider = FakeStreamingProvider()
    config = ThreadConfig()

    pipeline = QueryPipeline(
        db=db,
        config=config,
        run=run,
        thread=thread,
        message=message,
        provider=provider,
        run_manager=run_manager,
    )

    events = []
    async for event in pipeline.execute_stream():
        events.append(event)

    event_types = [e["type"] for e in events]

    # Verify lifecycle events are yielded in order:
    assert "step_start" in event_types
    assert "step_complete" in event_types
    assert "token" in event_types
    assert "complete" in event_types

    # Find steps
    step_starts = [e["step"]["step_type"] for e in events if e["type"] == "step_start"]
    step_completes = [e["step"]["step_type"] for e in events if e["type"] == "step_complete"]

    # Verify generation and pre-generation steps
    assert "transformation" in step_starts
    assert "transformation" in step_completes
    assert "generation" in step_starts
    assert "generation" in step_completes

    tokens = [e["content"] for e in events if e["type"] == "token"]
    assert "".join(tokens) == "Hello world!"

    complete_event = next(e for e in events if e["type"] == "complete")
    assert complete_event["result"]["response"] == "Hello world!"


def test_memory_step_output_formatting():
    from app.rag.pipeline.executor import PipelineContext, PipelineStepExecutor

    ctx = PipelineContext(
        run_id="test-run",
        thread_id="test-thread",
        message_id="test-msg",
        provider_name="test-prov",
        model_name="test-model",
    )
    executor = PipelineStepExecutor(ctx)

    # Test 3-tuple output
    turns = [{"role": "human", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
    summary = "User said Hi."
    is_summarized = True

    formatted = executor._format_output_for_db("memory", (turns, summary, is_summarized))
    assert formatted == {
        "recent_turns": turns,
        "turn_count": 2,
        "summary": summary,
        "is_summarized": True,
    }

    # Test list output fallback
    formatted_list = executor._format_output_for_db("memory", turns)
    assert formatted_list == {
        "recent_turns": turns,
        "turn_count": 2,
        "summary": None,
        "is_summarized": False,
    }

    # Test None output fallback
    formatted_none = executor._format_output_for_db("memory", None)
    assert formatted_none == {
        "recent_turns": [],
        "turn_count": 0,
        "summary": None,
        "is_summarized": False,
    }

