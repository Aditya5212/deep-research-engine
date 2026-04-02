# Issues & Solutions — A2A Agent (Deep Research Engine)

| # | Error | File | Root Cause | Fix Applied |
|---|-------|------|-----------|-------------|
| 1 | `TypeError: object of type 'coroutine' has no len()` | `a2a-server/src/agent.py` | `load_mcp_tools()` is async — called without `await` | `tools = await load_mcp_tools(_fastmcp_client.session)` |
| 2 | `TypeError: async_generator can't be used in 'await'` | `src/research/delegate_tool.py` | `client.send_message()` is an async generator, not a coroutine | `async for event in client.send_message(message, ...):` |
| 3 | `TypeError: list_tools() got unexpected keyword argument 'cursor'` | `langchain_mcp_adapters` → FastMCP | `load_mcp_tools` expects a raw MCP `ClientSession`, not a FastMCP `Client` wrapper | Pass `_fastmcp_client.session` instead of `_fastmcp_client` |
| 4 | `ValueError: Checkpointer requires thread_id / checkpoint_ns / checkpoint_id` | `a2a-server/src/agent_executor.py` | LangGraph checkpointer requires `configurable` keys in `ainvoke` config | `config={"configurable": {"thread_id": context_id}}` |
| 5 | `AttributeError: 'dict' object has no attribute 'name'` | `a2a-server/src/agent.py` — `ToolLoggingMiddleware` | LangChain Core ≥ 0.3 changed `tool_call` from Pydantic model to `TypedDict`/`dict` | `isinstance(tool_call, dict)` guard with `tool_call.get("name")` fallback |
| 6 | `datetime.utcnow()` deprecation warning | `a2a-server/src/agent.py` | Python 3.12+ deprecated `datetime.utcnow()` | `from datetime import UTC; datetime.now(UTC)` |
| 7 | Deprecated `A2AClient` usage | `src/research/delegate_tool.py` | `A2AClient` removed in newer A2A SDK | Replaced with `ClientFactory.connect()` + `ClientConfig` |

## Quick Reference: Async Patterns

```python
# Single async result
result = await async_function()

# Stream of async results
async for item in async_generator():
    process(item)

# Dict vs Object access (LangChain tool_call)
if isinstance(tool_call, dict):
    name = tool_call.get("name")
else:
    name = getattr(tool_call, "name", None)

# LangGraph ainvoke with checkpointer
await agent.ainvoke(input, config={"configurable": {"thread_id": context_id}})
```

## Status

```
✅ Server starts
✅ Agent initializes with MCP tools (Tavily)
✅ LLM (NVIDIA Kimi) responds and calls tools
✅ Checkpointer persists state per conversation context
✅ Tool call logging via AgentMiddleware
✅ A2A delegate tool uses correct client API
⚠️  JSON parse warning — graceful fallback to raw string (non-critical)
```
