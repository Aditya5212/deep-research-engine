import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from src.research.checkpointer import get_checkpointer

load_dotenv()

_base_agent = None
_agent_lock = asyncio.Lock()


def _build_agent(checkpointer):
    chat_model = init_chat_model(
        model="gemini-3-flash-preview",
        model_provider="google_genai",
        streaming=True,
        temperature=0.5,
        timeout=300,
        max_tokens=8000,
        configurable_fields=(
            "model",
            "model_provider",
            "streaming",
            "temperature",
            "timeout",
            "max_tokens",
            "base_url",
            "api_key",
        ),
    )
    return create_deep_agent(model=chat_model, checkpointer=checkpointer)


async def get_base_agent():
    """Return the singleton deep agent, initializing it once with an async lock."""
    global _base_agent
    if _base_agent is None:
        async with _agent_lock:
            if _base_agent is None:
                checkpointer = await get_checkpointer()
                _base_agent = _build_agent(checkpointer)
    return _base_agent


@asynccontextmanager
async def agent_context() -> AsyncGenerator:
    """Async context manager that yields the singleton agent."""
    agent = await get_base_agent()
    try:
        yield agent
    finally:
        pass  # teardown logic can go here if needed
