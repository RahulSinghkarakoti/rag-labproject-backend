from uuid import uuid4

from app.models.message import Message
from app.schemas.thread import MessageResponse


def test_message_response_serializes_assistant_ui_fields():
    message = Message(
        thread_id=uuid4(),
        role="assistant",
        content="Retrieved answer",
        sequence=2,
        status="completed",
        metadata_={"run_id": str(uuid4())},
    )
    message.id = uuid4()

    response = MessageResponse.model_validate(message)

    assert response.role == "assistant"
    assert response.content == "Retrieved answer"
    assert response.metadata["run_id"]


def test_message_create_supports_new_and_edit_payloads():
    from app.schemas.thread import MessageCreate

    # Regular new message
    new_msg = MessageCreate(content="Hello world")
    assert new_msg.content == "Hello world"
    assert new_msg.message_id is None

    # Edited message payload with message_id
    mid = str(uuid4())
    edit_msg = MessageCreate(content="Updated question", message_id=mid)
    assert edit_msg.content == "Updated question"
    assert edit_msg.message_id == mid

    # Retry message payload with no new content
    retry_msg = MessageCreate(message_id=mid)
    assert retry_msg.content is None
    assert retry_msg.message_id == mid

