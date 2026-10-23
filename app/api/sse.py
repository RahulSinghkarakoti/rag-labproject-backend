from typing import Any

from fastapi import APIRouter, Request

from app.core.sse import create_sse_response, publisher

router = APIRouter()


@router.options("/stream")
async def stream_events_options() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/stream")
async def stream_events(request: Request):
    async def event_generator():
        async for event in publisher.subscribe():
            yield event

    return create_sse_response(event_generator())


@router.post("/notify")
async def send_notification(message: str) -> dict[str, str]:
    """Helper endpoint to publish a generic notification event to connected clients."""
    await publisher.publish(data={"message": message}, event="notification")
    return {"status": "event published"}