from typing import Any

from src.research.agent import agent_context


async def chat_with_agent(message: str, thread_id: str = "default") -> str:
    """
    Send a message to the base agent and return its response.
    Uses the singleton agent via the async context manager.
    """
    async with agent_context() as agent:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
    content = result["messages"][-1].content
    # Some models return a list of content blocks e.g. [{'type': 'text', 'text': '...'}]
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return content


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
