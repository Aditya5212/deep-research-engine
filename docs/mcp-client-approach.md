# MCP Client Approach: Old vs New

## Summary

The agent originally connected to the Tavily MCP server using an HTTP-based client
(`MultiServerMCPClient`). Even though the MCP server runs **inside the same process**
as the FastAPI app, the old client still went through the full HTTP stack on every tool
call — opening a new session, performing a 5-request handshake, executing the tool, and
then closing the session.

The new approach uses FastMCP's in-process `Client`, which calls the tool functions
directly in memory — no sockets, no handshakes, no per-call overhead.

---

## Old Approach — `MultiServerMCPClient` over HTTP

**File:** `src/research/agent.py`

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

_mcp_client = MultiServerMCPClient(
    {
        "tavily": {
            "url": "http://localhost:8000/tavily/mcp",  # HTTP to localhost
            "transport": "streamable_http",
        }
    }
)
tools = await _mcp_client.get_tools()
```

### What happened on every tool call

```
Agent  →  HTTP POST /tavily/mcp  →  FastAPI  →  FastMCP server  →  FastAPI  →  Agent
```

5 HTTP requests per tool execution:
```
POST /tavily/mcp  200 OK      # MCP initialize
POST /tavily/mcp  202 Accepted # SSE session open
GET  /tavily/mcp  200 OK      # SSE stream open
POST /tavily/mcp  200 OK      # actual tool call
DELETE /tavily/mcp 200 OK     # session close
```

### Problems

| Problem | Detail |
|---|---|
| **Per-call handshake** | A new HTTP session was opened and closed for every single tool invocation |
| **Network overhead** | Full TCP stack even though server is in the same process |
| **Deprecation warning** | `DeprecationWarning: Use streamable_http_client instead` logged on every request |
| **Wrong tool for the job** | `MultiServerMCPClient` is designed for **remote/external** MCP servers |

---

## New Approach — `FastMCPClient` in-process

**File:** `src/research/agent.py`

```python
from fastmcp import Client as FastMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from src.tavily_mcp_server import mcp as tavily_mcp

# Open one persistent in-process session at agent startup
_fastmcp_client = FastMCPClient(tavily_mcp)   # passes the FastMCP instance directly
await _fastmcp_client.__aenter__()             # session opened once

tools = await load_mcp_tools(_fastmcp_client.session)  # tools loaded from live session
```

### What happens on every tool call

```
Agent  →  direct function call  →  Tavily tool function  →  Agent
```

0 HTTP requests per tool execution. The `FastMCPClient` uses an **in-memory transport**
when given a `FastMCP` instance — it bypasses HTTP entirely and calls tool functions
directly in the same Python process.

### Session lifetime

The `FastMCPClient` is **reentrant and reference-counted**. One session is opened at
agent startup and kept alive for the lifetime of the application. Multiple concurrent
tool calls share the same session safely without reconnecting.

Cleanup is done on app shutdown via `close_agent()`:

```python
async def close_agent():
    global _base_agent, _fastmcp_client
    if _fastmcp_client is not None:
        await _fastmcp_client.__aexit__(None, None, None)
        _fastmcp_client = None
    _base_agent = None
```

---

## Comparison

| | Old (`MultiServerMCPClient`) | New (`FastMCPClient`) |
|---|---|---|
| **Package** | `langchain-mcp-adapters` | `fastmcp` |
| **Transport** | HTTP to `localhost:8000` | In-process (in-memory) |
| **Session lifetime** | New session per tool call | One session, app lifetime |
| **HTTP requests per tool call** | 5 | 0 |
| **Deprecation warning** | Yes | No |
| **Designed for** | Remote/external MCP servers | Same-process FastMCP servers |
| **Concurrent safety** | Separate sessions (no sharing) | Reference-counted, safe |

---

## When to use each

| Scenario | Client to use |
|---|---|
| MCP server is in the **same process** (our case) | `FastMCPClient(mcp_instance)` |
| MCP server is on a **remote host** or separate process | `MultiServerMCPClient` with `streamable_http` transport |
| Testing/development against a local FastMCP server | `FastMCPClient(mcp_instance)` |

---

## Verified output

Running the in-process approach in isolation:

```
session type: <class 'mcp.client.session.ClientSession'>
is ClientSession: True
tools loaded: ['tavily_search', 'tavily_extract', 'tavily_crawl', 'tavily_map', 'tavily_research']
```

Server logs before fix (5 requests per tool call):
```
POST /tavily/mcp  200 OK
POST /tavily/mcp  202 Accepted
GET  /tavily/mcp  200 OK
POST /tavily/mcp  200 OK
DELETE /tavily/mcp 200 OK
```

Server logs after fix (no MCP HTTP traffic at all):
```
POST /research/chat  200 OK
POST /research/session/t03/resume  200 OK
GET  /research/session/t03/state   200 OK
```
