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
