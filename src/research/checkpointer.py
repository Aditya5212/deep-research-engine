import asyncio
import os

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

_checkpointer: AsyncPostgresSaver | None = None
_pool: AsyncConnectionPool | None = None
_checkpointer_lock = asyncio.Lock()


def _build_dsn() -> str:
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    db = os.getenv("POSTGRES_DB", "deep_research")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


async def get_checkpointer() -> AsyncPostgresSaver:
    """
    Return the singleton AsyncPostgresSaver, initializing it once.
    Sets up the DB schema on first call. Thread-safe via async lock.
    """
    global _checkpointer, _pool
    if _checkpointer is None:
        async with _checkpointer_lock:
            if _checkpointer is None:
                dsn = _build_dsn()
                pool = AsyncConnectionPool(
                    conninfo=dsn,
                    max_size=10,
                    kwargs={"autocommit": True},
                    open=False,
                )
                await pool.open()
                saver = AsyncPostgresSaver(pool)
                await saver.setup()  # creates checkpointer tables if not present
                _pool = pool
                _checkpointer = saver
    return _checkpointer


async def close_checkpointer() -> None:
    """Close the connection pool. Call this on app shutdown."""
    global _checkpointer, _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        _checkpointer = None
