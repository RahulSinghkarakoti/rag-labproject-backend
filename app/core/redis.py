from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

redis_pool: ArqRedis | None = None


async def init_redis_pool(
    *,
    host: str = "localhost",
    port: int = 6379,
) -> ArqRedis:
    """Create the shared ARQ Redis pool used to enqueue background jobs."""
    global redis_pool
    redis_pool = await create_pool(RedisSettings(host=host, port=port))
    return redis_pool


async def close_redis_pool() -> None:
    """Close the shared ARQ Redis pool on application shutdown."""
    global redis_pool
    if redis_pool is not None:
        await redis_pool.close()
        redis_pool = None


def get_redis_pool() -> ArqRedis | None:
    return redis_pool
