import asyncio
import logging
from uuid import UUID

from app.core.redis import get_redis_pool

logger = logging.getLogger(__name__)


class CancellationManager:
    """Coordinates in-process and distributed cancellation signals for RAG runs."""

    def __init__(self) -> None:
        self._events: dict[str, asyncio.Event] = {}

    def register(self, run_id: str | UUID) -> asyncio.Event:
        """Register an active run and return its cancellation Event."""
        key = str(run_id)
        event = asyncio.Event()
        self._events[key] = event
        logger.debug("[CancellationManager] Registered run %s", key)
        return event

    def unregister(self, run_id: str | UUID) -> None:
        """Unregister a finished run."""
        key = str(run_id)
        self._events.pop(key, None)
        logger.debug("[CancellationManager] Unregistered run %s", key)

    def is_cancelled(self, run_id: str | UUID) -> bool:
        """Check if cancellation was requested for this run."""
        key = str(run_id)
        event = self._events.get(key)
        return event.is_set() if event else False

    async def cancel(self, run_id: str | UUID) -> bool:
        """Signal cancellation to any local or distributed workers running this run."""
        key = str(run_id)
        signalled = False
        event = self._events.get(key)
        if event:
            event.set()
            signalled = True
            logger.info("[CancellationManager] Local cancellation event set for run %s", key)

        redis_pool = get_redis_pool()
        if redis_pool:
            try:
                await redis_pool.publish(f"run_cancellation:{key}", "cancel")
                logger.debug("[CancellationManager] Published cancellation to Redis for run %s", key)
            except Exception as e:
                logger.warning("[CancellationManager] Failed to publish cancellation to Redis: %s", e)

        return signalled


cancellation_manager = CancellationManager()
