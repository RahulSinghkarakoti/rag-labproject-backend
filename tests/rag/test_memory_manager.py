from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.models.message import Message
from app.models.thread import Thread, ThreadSummary
from app.rag.memory.context import build_rag_context, estimate_context_tokens, get_context_metrics
from app.rag.memory.manager import MemoryManager


@pytest.mark.anyio
async def test_memory_manager_load_returns_most_recent_in_chronological_order():
    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    thread_id = uuid4()

    # Create 10 messages with sequence 1 to 10
    messages = [
        Message(thread_id=thread_id, sequence=i, role="human" if i % 2 == 1 else "assistant", content=f"msg_{i}")
        for i in range(1, 11)
    ]

    # Mock DB scalars returning the messages in descending order (as SQL query does: 10, 9, 8...)
    mock_result = MagicMock()
    mock_result.all.return_value = list(reversed(messages[:5]))  # top 5 most recent: 5, 4, 3, 2, 1
    mock_db.scalars.return_value = mock_result

    mgr = MemoryManager(db=mock_db, max_turns=5)
    result = await mgr.load(thread_id=thread_id, before_sequence=11)

    # Should reverse descending order back to chronological order (1, 2, 3, 4, 5)
    assert len(result) == 5
    assert result[0]["content"] == "msg_1"
    assert result[-1]["content"] == "msg_5"


@pytest.mark.anyio
async def test_memory_manager_sliding_window_under_threshold():
    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    thread = Thread(id=uuid4(), metadata_={})

    # 4 messages (under window threshold 8)
    messages = [
        Message(thread_id=thread.id, sequence=i, role="human" if i % 2 == 1 else "assistant", content=f"turn_{i}")
        for i in range(1, 5)
    ]
    mock_result = MagicMock()
    mock_result.all.return_value = messages
    mock_db.scalars.return_value = mock_result

    mgr = MemoryManager(db=mock_db, max_turns=50)
    recent_turns, summary, is_summarized = await mgr.get_conversation_context(
        thread=thread,
        before_sequence=5,
        recent_window_turns=8,
    )

    assert len(recent_turns) == 4
    assert summary is None
    assert is_summarized is False


@pytest.mark.anyio
async def test_memory_manager_triggers_compaction_for_older_turns(monkeypatch):
    monkeypatch.setattr("app.rag.memory.manager.resolve_summarizer_provider", lambda p: p)
    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    thread = Thread(id=uuid4(), metadata_={})

    # 18 messages (older 10 unsummarized + recent 8 window turns)
    messages = [
        Message(thread_id=thread.id, sequence=i, role="human" if i % 2 == 1 else "assistant", content=f"turn_{i}")
        for i in range(1, 19)
    ]
    mock_result = MagicMock()
    mock_result.all.return_value = messages
    mock_db.scalars.return_value = mock_result

    mock_provider = AsyncMock()
    mock_provider.ainvoke.return_value = "Summary of turns 1 to 10."

    mgr = MemoryManager(db=mock_db, max_turns=50)
    recent_turns, summary, is_summarized = await mgr.get_conversation_context(
        thread=thread,
        before_sequence=19,
        provider=mock_provider,
        recent_window_turns=8,
    )

    assert len(recent_turns) == 8
    assert recent_turns[0]["content"] == "turn_11"
    assert recent_turns[-1]["content"] == "turn_18"
    assert summary == "Summary of turns 1 to 10."
    assert is_summarized is True
    # Verify ThreadSummary model was created and added to db, not thread.metadata_
    assert mock_db.add.called
    added_summary = mock_db.add.call_args[0][0]
    assert isinstance(added_summary, ThreadSummary)
    assert added_summary.thread_id == thread.id
    assert added_summary.summary == "Summary of turns 1 to 10."
    assert added_summary.covered_until_message_id == messages[9].id
    assert added_summary.token_count is not None


@pytest.mark.anyio
async def test_memory_manager_reads_existing_thread_summary_model():
    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    thread = Thread(id=uuid4(), metadata_={})

    covered_msg_id = uuid4()
    messages = [
        Message(id=covered_msg_id, thread_id=thread.id, sequence=1, role="human", content="msg_1"),
        Message(id=uuid4(), thread_id=thread.id, sequence=2, role="assistant", content="msg_2"),
    ]
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = messages
    mock_db.scalars.return_value = mock_scalars

    existing_summary_obj = ThreadSummary(
        id=uuid4(),
        thread_id=thread.id,
        summary="Prior summary from ThreadSummary model.",
        covered_until_message_id=covered_msg_id,
        token_count=12,
    )
    mock_exec = MagicMock()
    mock_exec.first.return_value = (existing_summary_obj, 1)
    mock_db.execute.return_value = mock_exec

    mgr = MemoryManager(db=mock_db, max_turns=50)
    recent_turns, summary, is_summarized = await mgr.get_conversation_context(
        thread=thread,
        before_sequence=3,
        recent_window_turns=8,
    )

    assert len(recent_turns) == 2
    assert summary == "Prior summary from ThreadSummary model."
    assert is_summarized is True


def test_build_rag_context_with_summary():
    queries = ["What is RAG?"]
    documents = [MagicMock(content="RAG stands for Retrieval-Augmented Generation.", document=MagicMock(name="rag.txt"))]
    memory = [{"role": "user", "content": "Tell me more."}]
    summary = "User asked about artificial intelligence previously."

    context = build_rag_context(queries, documents, memory, summary=summary)
    assert "Conversation summary (older turns):" in context
    assert "User asked about artificial intelligence previously." in context
    assert "Recent conversation memory:" in context
    assert "Retrieved context:" in context


def test_context_metrics_calculation():
    text = "a" * 400
    assert estimate_context_tokens(text) == 100

    metrics = get_context_metrics(text, is_summarized=True, model_limit=1000)
    assert metrics["tokens"] == 100
    assert metrics["percentage"] == 10
    assert metrics["summarized"] is True
    assert metrics["warning"] is None

    # Test near limit warning (>= 80%)
    high_text = "a" * 3600
    high_metrics = get_context_metrics(high_text, is_summarized=True, model_limit=1000)
    assert high_metrics["percentage"] == 90
    assert high_metrics["warning"] == "near_limit"
