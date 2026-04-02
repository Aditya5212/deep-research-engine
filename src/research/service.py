import asyncio
import logging
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langgraph.types import Command

from src.research.agent import agent_context

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run metadata collection via LangChain callbacks
# ---------------------------------------------------------------------------

class RunMetadataCollector(BaseCallbackHandler):
    """
    Attached to every ainvoke() call via config["callbacks"].

    Fires on every LLM call and tool execution inside the agent run,
    giving cumulative token counts across all reasoning steps and a
    structured log of every tool call with timing.
    """

    def __init__(self) -> None:
        super().__init__()
        self._t0 = time.perf_counter()
        self._tool_starts: dict[str, float] = {}
        self.tool_calls: list[dict[str, Any]] = []
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.llm_calls: int = 0

    # ── LLM hooks ─────────────────────────────────────────────────────────────

    def on_llm_start(self, serialized: dict, prompts: list[str], *, run_id: UUID, **kwargs: Any) -> None:
        self.llm_calls += 1

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        # Path 1 — llm_output dict (OpenAI-compat style used by ChatNVIDIA)
        if response.llm_output:
            raw = response.llm_output.get("token_usage") or response.llm_output.get("usage") or {}
            self.input_tokens += raw.get("prompt_tokens", raw.get("input_tokens", 0))
            self.output_tokens += raw.get("completion_tokens", raw.get("output_tokens", 0))
            return
        # Path 2 — usage_metadata on ChatGeneration (newer LangChain style)
        for gen_list in response.generations:
            for gen in gen_list:
                um = getattr(getattr(gen, "message", None), "usage_metadata", None)
                if um:
                    self.input_tokens += um.get("input_tokens", 0)
                    self.output_tokens += um.get("output_tokens", 0)

    # ── Tool hooks ─────────────────────────────────────────────────────────────

    def on_tool_start(self, serialized: dict, input_str: str, *, run_id: UUID, **kwargs: Any) -> None:
        key = str(run_id)
        self._tool_starts[key] = time.perf_counter()
        self.tool_calls.append({
            "name": serialized.get("name", "unknown"),
            "input": str(input_str)[:500],
            "elapsed_s": None,
            "error": None,
        })

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        key = str(run_id)
        t0 = self._tool_starts.pop(key, None)
        if self.tool_calls:
            self.tool_calls[-1]["elapsed_s"] = round(time.perf_counter() - t0, 3) if t0 else None

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        key = str(run_id)
        t0 = self._tool_starts.pop(key, None)
        if self.tool_calls:
            self.tool_calls[-1]["error"] = str(error)
            self.tool_calls[-1]["elapsed_s"] = round(time.perf_counter() - t0, 3) if t0 else None

    # ── Export ─────────────────────────────────────────────────────────────────

    def to_metadata(self) -> dict[str, Any]:
        total = self.input_tokens + self.output_tokens
        return {
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": total,
            },
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "elapsed_s": round(time.perf_counter() - self._t0, 3),
        }

# ---------------------------------------------------------------------------
# Transient error handling
# ---------------------------------------------------------------------------

_TRANSIENT_MARKERS = ("503", "429", "UNAVAILABLE", "Resource has been exhausted", "rate limit")
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0  # seconds


class TransientAPIError(Exception):
    """Raised when the underlying LLM API returns a transient/retryable error."""


def _is_transient(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(marker.lower() in msg for marker in _TRANSIENT_MARKERS)


async def _ainvoke_with_retry(agent, input_: Any, config: dict[str, Any]) -> RunMetadataCollector:
    """Call agent.ainvoke with exponential-backoff retry for transient errors.

    Injects a RunMetadataCollector into ``config["callbacks"]`` and returns it
    so the caller can read accumulated token usage and tool call logs.
    """
    collector = RunMetadataCollector()
    patched = {**config, "callbacks": [*config.get("callbacks", []), collector]}
    last_exc: BaseException | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            await agent.ainvoke(input_, config=patched)
            return collector
        except Exception as exc:
            if _is_transient(exc):
                last_exc = exc
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                log.warning(
                    "Transient API error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, _MAX_RETRIES, delay, exc,
                )
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(delay)
                    continue
            raise
    raise TransientAPIError(str(last_exc)) from last_exc


def _extract_content(content: Any) -> str:
    """Normalize message content to a plain string."""
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def _collect_interrupts(snapshot) -> list[dict[str, Any]]:
    """Extract pending HITL interrupts from a state snapshot."""
    interrupts: list[dict[str, Any]] = []
    for task in snapshot.tasks or []:
        for intr in task.interrupts or []:
            interrupts.append({"id": intr.id, "value": intr.value})
    return interrupts


async def chat_with_agent(message: str, thread_id: str = "default") -> dict[str, Any]:
    """
    Send a message to the base agent and return a structured response.

    If the agent pauses to wait for human approval (HITL), the response will
    have ``interrupted=True`` and a populated ``interrupts`` list.  The caller
    should present the interrupt details to the user, collect a decision, and
    then call :func:`resume_agent`.
    """
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    async with agent_context() as agent:
        collector = await _ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": message}]},
            config,
        )
        snapshot = await agent.aget_state(config)

    interrupts = _collect_interrupts(snapshot)
    if interrupts:
        return {"reply": None, "interrupted": True, "interrupts": interrupts, "metadata": collector.to_metadata()}

    messages = snapshot.values.get("messages", [])
    reply = _extract_content(messages[-1].content) if messages else ""
    return {"reply": reply, "interrupted": False, "interrupts": [], "metadata": collector.to_metadata()}


async def get_pending_interrupts(thread_id: str) -> list[dict[str, Any]]:
    """Return any pending HITL interrupts for *thread_id* without invoking the agent."""
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    async with agent_context() as agent:
        snapshot = await agent.aget_state(config)
    return _collect_interrupts(snapshot)


async def resume_agent(
    thread_id: str,
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Resume an interrupted agent with human decisions.

    Each entry in *decisions* must be one of::

        {"type": "approve"}
        {"type": "edit", "edited_action": {"name": "<tool>", "args": {...}}}
        {"type": "reject", "message": "<optional reason>"}

    The number of decisions must match the number of pending action_requests
    reported in the interrupt value.
    """
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    async with agent_context() as agent:
        collector = await _ainvoke_with_retry(
            agent,
            Command(resume={"decisions": decisions}),
            config,
        )
        snapshot = await agent.aget_state(config)

    # Agent may pause at another HITL checkpoint
    interrupts = _collect_interrupts(snapshot)
    if interrupts:
        return {"reply": None, "interrupted": True, "interrupts": interrupts, "metadata": collector.to_metadata()}

    messages = snapshot.values.get("messages", [])
    reply = _extract_content(messages[-1].content) if messages else ""
    return {"reply": reply, "interrupted": False, "interrupts": [], "metadata": collector.to_metadata()}


def _strip_msg_metadata(msg) -> dict:
    """Serialize a LangChain message via model_dump(), dropping provider metadata."""
    d = msg.model_dump()
    d.pop("response_metadata", None)
    d.pop("usage_metadata", None)
    return d


def _snapshot_to_dict(snapshot) -> dict[str, Any]:
    """Serialize a LangGraph StateSnapshot to a plain dict."""
    values = dict(snapshot.values)
    if "messages" in values:
        values["messages"] = [_strip_msg_metadata(m) for m in values["messages"]]
    return {
        "values": values,
        "next": list(snapshot.next),
        "config": snapshot.config,
        "metadata": snapshot.metadata,
        "created_at": snapshot.created_at,
        "parent_config": snapshot.parent_config,
        "tasks": [
            {"id": t.id, "name": t.name, "error": str(t.error) if t.error else None}
            for t in (snapshot.tasks or [])
        ],
    }


async def get_thread_state(thread_id: str, checkpoint_id: str | None = None) -> dict[str, Any]:
    """Return the latest state snapshot for a thread (or a specific checkpoint)."""
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if checkpoint_id:
        config["configurable"]["checkpoint_id"] = checkpoint_id
    async with agent_context() as agent:
        snapshot = await agent.aget_state(config)
    return _snapshot_to_dict(snapshot)


async def get_thread_history(thread_id: str) -> list[dict[str, Any]]:
    """Return the full chronological state history for a thread."""
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    async with agent_context() as agent:
        history = []
        async for snapshot in agent.aget_state_history(config):
            history.append(_snapshot_to_dict(snapshot))
    return history
