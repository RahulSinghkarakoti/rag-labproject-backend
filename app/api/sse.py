from typing import Any

from fastapi import APIRouter, Depends, Request

from app.api.dependencies import get_current_user
from app.core.sse import create_sse_response, publisher
from app.models.user import User

router = APIRouter()


@router.options("/stream")
async def stream_events_options() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/stream")
async def stream_events(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    channel = f"ingestion_channel:{current_user.id}"

    async def event_generator():
        async for event in publisher.subscribe(channel=channel):
            yield event

    return create_sse_response(event_generator())


@router.post("/notify")
async def send_notification(
    message: str,
    current_user: User = Depends(get_current_user),
) -> dict[str, str]:
    """Helper endpoint to publish a generic notification event to the authenticated user's clients."""
    channel = f"ingestion_channel:{current_user.id}"
    await publisher.publish(data={"message": message}, event="notification", channel=channel)
    return {"status": "event published"}