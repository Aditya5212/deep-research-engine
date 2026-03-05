import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.research.checkpointer import get_checkpointer

load_dotenv()

_base_agent = None
_mcp_client: MultiServerMCPClient | None = None
_agent_lock = asyncio.Lock()

TAVILY_MCP_URL = os.getenv("TAVILY_MCP_URL", "http://localhost:8000/tavily/mcp")


async def _build_agent(checkpointer):
    global _mcp_client

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

    _mcp_client = MultiServerMCPClient(
        {
            "tavily": {
                "url": TAVILY_MCP_URL,
                "transport": "streamable_http",
            }
        }
    )
    tools = await _mcp_client.get_tools()

    return create_deep_agent(model=chat_model, tools=tools, checkpointer=checkpointer)


async def get_base_agent():
    """Return the singleton deep agent, initializing it once with an async lock."""
    global _base_agent
    if _base_agent is None:
        async with _agent_lock:
            if _base_agent is None:
                checkpointer = await get_checkpointer()
                _base_agent = await _build_agent(checkpointer)
    return _base_agent


async def close_agent():
    """Reset the agent and MCP client singletons."""
    global _base_agent, _mcp_client
    _mcp_client = None
    _base_agent = None


@asynccontextmanager
async def agent_context() -> AsyncGenerator:
    """Async context manager that yields the singleton agent."""
    agent = await get_base_agent()
    try:
        yield agent
    finally:
        pass  # teardown logic can go here if needed
