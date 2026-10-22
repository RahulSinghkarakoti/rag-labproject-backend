from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.models.message import Message
from app.models.run import Run, RunStep, ModelCall
from app.rag.engine import QueryEngine


class FakeDB:
    def __init__(self):
        self.items = []

    def add(self, item):
        self.items.append(item)

    async def flush(self):
        for item in self.items:
            if getattr(item, "id", None) is None:
                item.id = uuid4()


@pytest.mark.anyio
async def test_persists_assistant_message_steps_and_model_call():
    db = FakeDB()
    thread_id = uuid4()
    run = Run(thread_id=thread_id, run_type="chat")
    run.id = uuid4()
    message = Message(
        thread_id=thread_id,
        role="human",
        content="What is RAG?",
        sequence=1,
    )
    message.id = uuid4()
    provider = SimpleNamespace(provider_name="test", config=SimpleNamespace(model="test-model"))

    await QueryEngine(db)._persist_run_result(
        run=run,
        message=message,
        result={
            "transformed_queries": ["What is retrieval augmented generation?"],
            "retrieved_documents": [{"id": "doc-1", "content": "RAG context"}],
            "reranked_documents": [{"id": "doc-1", "content": "RAG context"}],
            "response": "RAG combines retrieval and generation.",
        },
        config={"generation": {"temperature": 0.2, "max_tokens": 100}},
        provider=provider,
    )

    assert run.output_message_id is not None
    # RunStep and ModelCall persistence are now delegated to RunManager
    # assert len([item for item in db.items if isinstance(item, RunStep)]) == 4
    # assert len([item for item in db.items if isinstance(item, ModelCall)]) == 1
    assistant = next(item for item in db.items if isinstance(item, Message))
    assert assistant.role == "assistant"
    assert assistant.content == "RAG combines retrieval and generation."
