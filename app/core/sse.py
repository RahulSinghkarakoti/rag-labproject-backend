import json
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi.responses import StreamingResponse
import redis.asyncio as aioredis

redis = aioredis.from_url("redis://localhost:6379")

class RedisEventPublisher:
    async def subscribe(self, channel: str = "ingestion_channel"):
        pubsub = redis.pubsub()
        await pubsub.subscribe(channel)
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    payload = json.loads(message["data"].decode("utf-8"))
                    yield payload
        finally:
            await pubsub.unsubscribe(channel)

    async def publish(self, data: dict, event: str = "ingestion_status", channel: str = "ingestion_channel"):
        payload = {"event": event, "data": json.dumps(data)}
        await redis.publish(channel, json.dumps(payload))

publisher = RedisEventPublisher()

def create_sse_response(generator: AsyncGenerator) -> StreamingResponse:
    """Helper wrapper for creating a push-based SSE response."""
    async def format_sse():
        async for item in generator:
            if isinstance(item, dict):
                event = item.get("event", "message")
                data = item.get("data", "")
                if not isinstance(data, str):
                    data = json.dumps(data)
                yield f"event: {event}\ndata: {data}\n\n"
            else:
                yield f"data: {item}\n\n"
    
    return StreamingResponse(
        format_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )