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
