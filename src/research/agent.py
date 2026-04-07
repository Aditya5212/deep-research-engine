import asyncio
import os
from contextlib import asynccontextmanager
from typing import Annotated, AsyncGenerator

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from dotenv import load_dotenv
from fastmcp import Client as FastMCPClient
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools

from src.research.checkpointer import get_checkpointer
from src.research.state import ResearchAgentState
from src.tavily_mcp_server import mcp as tavily_mcp
from src.research.delegate_tool import delegate_research_task

load_dotenv()

_base_agent = None
_fastmcp_client: FastMCPClient | None = None
_agent_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Delegated-tasks inspection tool
# ---------------------------------------------------------------------------

@tool("list_delegated_tasks")
def list_delegated_tasks(
    state: Annotated[ResearchAgentState, InjectedState],
) -> dict:
    """
    Return all research tasks delegated to A2A sub-agents in this session.

    Output is a dict keyed by A2A task_id:
        {
          "<task_id>": {
            "sub_question":   str,   # the question sent to the sub-agent
            "domain":         str,   # finance | medical | tech | legal | general
            "context_id":     str,   # A2A thread_id reused for this domain
            "confidence":     str,   # high | medium | low
            "delegated_at":   str,   # ISO-8601 UTC timestamp
          },
          ...
        }

    Use this to check what sub-research has already been done before deciding
    whether to delegate again or synthesize from existing findings.
    """
    return state.get("delegated_tasks") or {}


# ---------------------------------------------------------------------------
# Orchestrator system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a deep research orchestrator. Your role is to answer complex questions by:
1. Decomposing the question into focused sub-questions per domain (finance, medical, tech, legal, general)
2. Delegating each sub-question to a specialized domain researcher via delegate_research_task
3. Synthesizing all returned findings into a comprehensive, cited answer

## Managing research threads

Each call to delegate_research_task runs on an A2A sub-agent thread identified by a context_id.
You control which thread handles each request. Always call list_delegated_tasks first to see
all existing threads and their context_ids before deciding.

Decision rules for context_id:

- **New topic / isolated research** → omit context_id (or pass None).
  The tool creates a fresh thread. Use this when the sub-question is genuinely new and
  should not be influenced by any prior search history.

- **Dig deeper into existing research** → pass the context_id of an existing task entry.
  The same sub-agent resumes with full memory of its prior searches. Use this when you
  need more detail on something already partially researched.

- **Same domain, different angle** → create a new thread (omit context_id) if the new
  question is distinct enough that prior context would confuse the sub-agent; reuse the
  existing thread if the prior context is relevant background.

## Other guidelines
- Always cite sources from the citations list in your final answer
- Cross-validate claims that appear across multiple domain findings
- Flag remaining gaps or contradictions explicitly in your synthesis
"""

# ---------------------------------------------------------------------------
# Human-in-the-Loop policy
# ---------------------------------------------------------------------------
# delegate_research_task  — spawns an A2A sub-agent (high-impact, irreversible)
# tavily_extract          — fetches explicit URLs (privacy / cost concern)
# tavily_crawl            — broad crawl (expensive, may touch sensitive paths)
# tavily_search / map     — cheap read-only lookups; auto-approved (not listed)
HITL_INTERRUPT_ON: dict[str, bool | InterruptOnConfig] = {
    "delegate_research_task": InterruptOnConfig(
        allowed_decisions=["approve", "edit", "reject"],
        description="Domain research sub-task requested. Review the sub-question and domain before spawning the A2A agent.",
    ),
    "tavily_extract": InterruptOnConfig(
        allowed_decisions=["approve", "edit", "reject"],
        description="URL extraction requested. Review the target URLs before proceeding.",
    ),
    "tavily_crawl": InterruptOnConfig(
        allowed_decisions=["approve", "edit", "reject"],
        description="Website crawl requested. Review the target URL and scope before proceeding.",
    ),
}


async def _build_agent(checkpointer):
    global _fastmcp_client

    # Previous Google Gemini config (commented out for NVIDIA switch)
    # chat_model = init_chat_model(
    #     model="gemini-3-flash-preview",
    #     model_provider="google_genai",
    #     streaming=True,
    #     temperature=0.5,
    #     timeout=300,
    #     max_tokens=8000,
    #     configurable_fields=(
    #         "model",
    #         "model_provider",
    #         "streaming",
    #         "temperature",
    #         "timeout",
    #         "max_tokens",
    #         "base_url",
    #         "api_key",
    #     ),
    # )

    chat_model = ChatOpenAI(
        model="moonshotai/kimi-k2-instruct",
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url="https://integrate.api.nvidia.com/v1",
        temperature=0.6,
        top_p=0.9,
        max_completion_tokens=16384,
    )

    # In-process FastMCP client — connects directly to the FastMCP instance
    # without going through HTTP. The client is reentrant (reference-counted),
    # so the single open session is reused across all agent invocations.
    _fastmcp_client = FastMCPClient(tavily_mcp)
    await _fastmcp_client.__aenter__()
    tools = await load_mcp_tools(_fastmcp_client.session)
    # tavily_research is a high-level Tavily AI report — excluded because:
    # 1. Its schema (input/model/stream) conflicts with how the LLM expects search tools to work
    # 2. The agent does its own multi-step research using tavily_search directly
    tools = [t for t in tools if t.name != "tavily_research"]
    tools.append(delegate_research_task)
    tools.append(list_delegated_tasks)

    return create_agent(
        model=chat_model,
        tools=tools,
        checkpointer=checkpointer,
        state_schema=ResearchAgentState,
        system_prompt=SYSTEM_PROMPT,
        middleware=[HumanInTheLoopMiddleware(interrupt_on=HITL_INTERRUPT_ON)],
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
