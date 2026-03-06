import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from deepagents import create_deep_agent
from dotenv import load_dotenv
from fastmcp import Client as FastMCPClient
from langchain.agents.middleware import InterruptOnConfig
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.tools import load_mcp_tools

from src.research.checkpointer import get_checkpointer
from src.tavily_mcp_server import mcp as tavily_mcp

load_dotenv()

_base_agent = None
_fastmcp_client: FastMCPClient | None = None
_agent_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Human-in-the-Loop configuration
# ---------------------------------------------------------------------------
# Tools that require human approval before execution.
# - tavily_research : comprehensive multi-source research (expensive, high-impact)
# - tavily_extract  : fetches content from explicit URLs (privacy/cost concern)
# - tavily_crawl    : broad website crawl (expensive, may touch sensitive paths)
# tavily_search and tavily_map are auto-approved (low-risk, cheap).
HITL_INTERRUPT_ON: dict[str, bool | InterruptOnConfig] = {
    "tavily_research": InterruptOnConfig(
        allowed_decisions=["approve", "edit", "reject"],
        description="Comprehensive research task requested. Review the query before proceeding.",
    ),
    "tavily_extract": InterruptOnConfig(
        allowed_decisions=["approve", "edit", "reject"],
        description="🌐 URL extraction requested. Review the target URLs before proceeding.",
    ),
    "tavily_crawl": InterruptOnConfig(
        allowed_decisions=["approve", "edit", "reject"],
        description="🕷️ Website crawl requested. Review the target URL and scope before proceeding.",
    ),
}


async def _build_agent(checkpointer):
    global _fastmcp_client

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

    # In-process FastMCP client — connects directly to the FastMCP instance
    # without going through HTTP. The client is reentrant (reference-counted),
    # so the single open session is reused across all agent invocations.
    _fastmcp_client = FastMCPClient(tavily_mcp)
    await _fastmcp_client.__aenter__()
    tools = await load_mcp_tools(_fastmcp_client.session)

    return create_deep_agent(
        model=chat_model,
        tools=tools,
        checkpointer=checkpointer,
        interrupt_on=HITL_INTERRUPT_ON,
    )


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
    global _base_agent, _fastmcp_client
    if _fastmcp_client is not None:
        await _fastmcp_client.__aexit__(None, None, None)
        _fastmcp_client = None
    _base_agent = None


@asynccontextmanager
async def agent_context() -> AsyncGenerator:
    """Async context manager that yields the singleton agent."""
    agent = await get_base_agent()
    try:
        yield agent
    finally:
        pass  # teardown logic can go here if needed
