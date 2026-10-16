from uuid import uuid4
import pytest
from types import SimpleNamespace

from app.models.message import Message
from app.models.run import Run
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
async def test_retry_creates_new_message_with_same_parent_id():
    db = FakeDB()
    thread_id = uuid4()
    
    # Human message
    user_msg = Message(
        thread_id=thread_id,
        role="human",
        content="Explain RAG",
        sequence=1,
    )
    user_msg.id = uuid4()

    provider = SimpleNamespace(provider_name="test", config=SimpleNamespace(model="test-model"))

    # Initial Run & Assistant Response 1
    run1 = Run(thread_id=thread_id, run_type="chat")
    run1.id = uuid4()
    await QueryEngine(db)._persist_run_result(
        run=run1,
        message=user_msg,
        result={"response": "RAG Version 1"},
        config={"generation": {}},
        provider=provider,
    )

    # Retry Run & Assistant Response 2 (re-using same user_msg as input)
    run2 = Run(thread_id=thread_id, run_type="chat")
    run2.id = uuid4()
    await QueryEngine(db)._persist_run_result(
        run=run2,
        message=user_msg,
        result={"response": "RAG Version 2"},
        config={"generation": {}},
        provider=provider,
    )

    assistant_messages = [item for item in db.items if isinstance(item, Message) and item.role == "assistant"]
    assert len(assistant_messages) == 2

    # Both assistant messages should exist as separate entries and reference the same parent user_msg
    msg1, msg2 = assistant_messages
    assert msg1.id != msg2.id
    assert msg1.parent_message_id == user_msg.id
    assert msg2.parent_message_id == user_msg.id
    assert msg1.content == "RAG Version 1"
    assert msg2.content == "RAG Version 2"
