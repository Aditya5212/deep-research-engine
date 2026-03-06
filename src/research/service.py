import asyncio
import logging
from typing import Any

from langgraph.types import Command

from src.research.agent import agent_context

log = logging.getLogger(__name__)

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


async def _ainvoke_with_retry(agent, input_: Any, config: dict[str, Any]) -> None:
    """Call agent.ainvoke with exponential-backoff retry for transient errors."""
    last_exc: BaseException | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            await agent.ainvoke(input_, config=config)
            return
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
        await _ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": message}]},
            config,
        )
        snapshot = await agent.aget_state(config)

    interrupts = _collect_interrupts(snapshot)
    if interrupts:
        return {"reply": None, "interrupted": True, "interrupts": interrupts}

    messages = snapshot.values.get("messages", [])
    reply = _extract_content(messages[-1].content) if messages else ""
    return {"reply": reply, "interrupted": False, "interrupts": []}


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
        await _ainvoke_with_retry(
            agent,
            Command(resume={"decisions": decisions}),
            config,
        )
        snapshot = await agent.aget_state(config)

    # Agent may pause at another HITL checkpoint
    interrupts = _collect_interrupts(snapshot)
    if interrupts:
        return {"reply": None, "interrupted": True, "interrupts": interrupts}

    messages = snapshot.values.get("messages", [])
    reply = _extract_content(messages[-1].content) if messages else ""
    return {"reply": reply, "interrupted": False, "interrupts": []}


def _snapshot_to_dict(snapshot) -> dict[str, Any]:
    """Serialize a LangGraph StateSnapshot to a plain dict."""
    return {
        "values": snapshot.values,
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
