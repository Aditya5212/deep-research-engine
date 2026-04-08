import os
import logging
import traceback
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from datetime import datetime, UTC
from config import settings
from src.app_logger import get_logger
from fastmcp import Client as FastMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from src.research.checkpointer import get_checkpointer

load_dotenv()

# Standalone MCP server URL — set MCP_SERVER_URL in .env to override.
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001/mcp")


logger = get_logger("a2a.agent")

# Module start timestamp — log for debugging/deployment audits
_MODULE_START_TIME = datetime.now(UTC).isoformat()
logger.info("[AGENT START] module_start_time=%s", _MODULE_START_TIME)

# --- Domain-specific system prompts ---
DOMAIN_PROMPTS = {
    "finance": "You are a financial research analyst. Prioritize SEC filings, earnings reports, and analyst notes. Always cite source authority.",
    "medical": "You are a biomedical researcher. Prioritize peer-reviewed papers, PubMed, and clinical trial data. Flag any non-peer-reviewed sources.",
    "tech": "You are a technical research analyst. Prioritize arXiv, GitHub, official docs, and credible tech publications.",
    "legal": "You are a legal research specialist. Prioritize statutes, case law, and regulatory filings. Note jurisdiction.",
    "general": "You are a deep research analyst. Gather information from multiple authoritative sources, cross-validate claims, and identify gaps.",
}

# Optional FastMCP initialization
_fastmcp_client = None


# --- AgentMiddleware for logging tool calls (module-level) ---
class ToolLoggingMiddleware(AgentMiddleware):
    async def awrap_tool_call(self, request, handler):
        # tool_call is a dict in LangChain Core ≥ 0.3 (TypedDict, not Pydantic model)
        tool_call = request.tool_call
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("name", "?")
            tool_args = tool_call.get("args", {})
        else:
            tool_name = getattr(tool_call, "name", "?")
            tool_args = getattr(tool_call, "args", {})
        logger.info("[TOOL CALL] %s args=%s", tool_name, tool_args)
        return await handler(request)


async def get_researcher_agent(domain: str = "general"):
    """Build a LangChain v1 researcher agent for a given domain."""
    model = ChatOpenAI(
        model="moonshotai/kimi-k2-instruct",
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url="https://integrate.api.nvidia.com/v1",
        temperature=0.6,
        top_p=0.9,
        max_completion_tokens=16384,
    )

    system_prompt = f"""{DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["general"])}

Your job is to research a single sub-question deeply.
For each search iteration:
1. Generate targeted queries based on the sub-question and any identified gaps
2. Search and analyze retrieved sources
3. Extract key claims with confidence (high/medium/low)
4. Identify remaining gaps
5. Stop when: 3+ independent sources agree OR max iterations reached

Always return structured output:
{{
  "findings": "...",
  "citations": [{{"url": "...", "title": "...", "claim": "..."}}],
  "confidence": "high|medium|low",
  "identified_gaps": ["..."]
}}
"""

    global _fastmcp_client
    if _fastmcp_client is None:
        logger.info("[MCP] No active client — connecting to %s", MCP_SERVER_URL)
        try:
            _fastmcp_client = FastMCPClient(MCP_SERVER_URL)
            await _fastmcp_client.__aenter__()
            logger.info("[MCP] FastMCPClient connected to standalone MCP server")
        except Exception as exc:
            _fastmcp_client = None
            logger.error(
                "[MCP] FastMCPClient initialization failed: %s: %s\n%s",
                type(exc).__name__, exc, traceback.format_exc(),
            )
            raise RuntimeError(f"MCP server initialization failed: {exc}") from exc
    else:
        logger.debug("[MCP] Reusing existing FastMCPClient session")

    # Pass the underlying MCP ClientSession — load_mcp_tools expects ClientSession,
    # not the FastMCP Client wrapper. FastMCP Client.session is the raw MCP session.
    try:
        tools = await load_mcp_tools(_fastmcp_client.session)
        logger.info(
            "[MCP] Loaded %d tools from Tavily MCP server: %s",
            len(tools),
            [t.name for t in tools],
        )
    except Exception as exc:
        logger.error(
            "[MCP] load_mcp_tools failed: %s: %s\n%s",
            type(exc).__name__, exc, traceback.format_exc(),
        )
        raise RuntimeError(f"Failed to load MCP tools: {exc}") from exc

    # Initialize checkpointer (shared singleton). Deployment note: when deploying,
    # use a different checkpointer instance/config for production environments.
    # Logged timestamp below records when this agent initialized the checkpointer.
    try:
        checkpointer = await get_checkpointer()
        _CHECKPOINT_INIT_TIME = datetime.now(UTC).isoformat()
        logger.info("[CHECKPOINTER] initialized_at=%s", _CHECKPOINT_INIT_TIME)
    except Exception as exc:
        logger.error(
            "[CHECKPOINTER] Initialization failed: %s: %s\n%s",
            type(exc).__name__, exc, traceback.format_exc(),
        )
        raise RuntimeError(f"Checkpointer initialization failed: {exc}") from exc

    try:
        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            middleware=[ToolLoggingMiddleware()],
            checkpointer=checkpointer,
        )
        logger.info(
            "[AGENT] create_agent complete for domain=%s | tools=%d", domain, len(tools)
        )
    except Exception as exc:
        logger.error(
            "[AGENT] create_agent failed for domain=%s: %s: %s\n%s",
            domain, type(exc).__name__, exc, traceback.format_exc(),
        )
        raise RuntimeError(f"Agent creation failed for domain={domain}: {exc}") from exc

    return agent


@asynccontextmanager
async def researcher_agent_context(domain: str = "general") -> AsyncGenerator:
    """Async context manager that yields the domain researcher agent."""
    logger.info("[CTX] Entering researcher_agent_context for domain=%s", domain)
    agent = await get_researcher_agent(domain)
    try:
        yield agent
    finally:
        logger.info("[CTX] Exiting researcher_agent_context for domain=%s", domain)
