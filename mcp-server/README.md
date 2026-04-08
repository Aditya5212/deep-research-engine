# mcp-server

Standalone **Tavily MCP Server** built with [FastMCP](https://github.com/jlowin/fastmcp).  
Exposes all Tavily API capabilities as MCP tools over the **Streamable HTTP transport** — runs independently and can be connected to by any MCP client (e.g. the A2A research agent via `langchain-mcp-adapters`).

---

## Requirements

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- A Tavily API key — get one free at https://app.tavily.com

---

## Setup

```bash
cd mcp-server

# 1. Copy env template and fill in your key
cp .env.example .env

# 2. Install dependencies (creates isolated .venv)
uv sync
```

`.env` file:
```env
TAVILY_API_KEY=tvly-your-key-here
LOG_LEVEL=INFO   # optional: DEBUG for verbose output
```

---

## Running the server

```bash
uv run start
```

The server starts on **`http://0.0.0.0:8001`**.  
The MCP endpoint is available at **`http://localhost:8001/mcp`**.

To run on a different port:
```bash
uv run uvicorn main:asgi_app --host 0.0.0.0 --port 9000
```

---

## Connecting from an MCP client

Using `fastmcp` Python client:
```python
from fastmcp import Client

async with Client("http://localhost:8001/mcp") as client:
    tools = await client.list_tools()
    result = await client.call_tool("tavily_search", {"query": "AI news"})
```

Using `langchain-mcp-adapters` (as used by the A2A research agent):
```python
from fastmcp import Client as FastMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

client = FastMCPClient("http://localhost:8001/mcp")
await client.__aenter__()
tools = await load_mcp_tools(client.session)
```

Set `MCP_SERVER_URL=http://localhost:8001/mcp` in the A2A server's `.env` to point it at this instance.

---

## Available tools

### `tavily_search`
Execute a web search query.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | str | required | Search query |
| `search_depth` | str | `basic` | `basic` / `advanced` / `fast` / `ultra-fast` |
| `topic` | str | `general` | `general` / `news` / `finance` |
| `max_results` | int | `5` | Number of results (0–20) |
| `include_answer` | str | `false` | `false` / `basic` / `advanced` |
| `include_raw_content` | str | `false` | `false` / `markdown` / `text` |
| `time_range` | str | `None` | `day` / `week` / `month` / `year` |
| `start_date` / `end_date` | str | `None` | Date filter (`YYYY-MM-DD`) |
| `include_images` | bool | `False` | Include image search results |
| `include_domains` / `exclude_domains` | list | `None` | Domain allow/block lists |
| `country` | str | `None` | Boost results from a country |
| `auto_parameters` | bool | `False` | Let Tavily auto-tune parameters |
| `exact_match` | bool | `False` | Exact phrase matching |

---

### `tavily_extract`
Extract content from one or more URLs.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `urls` | list[str] | required | URLs to extract (max 20) |
| `extract_depth` | str | `basic` | `basic` / `advanced` |
| `output_format` | str | `markdown` | `markdown` / `text` |
| `query` | str | `None` | Rerank chunks by relevance to this query |
| `include_images` | bool | `False` | Include extracted images |
| `timeout` | float | `None` | Per-URL timeout in seconds (1–60) |

---

### `tavily_crawl`
Graph-based website crawler starting from a root URL.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `url` | str | required | Root URL to crawl |
| `max_depth` | int | `1` | Link-hop depth (1–5) |
| `max_breadth` | int | `20` | Max links per page (1–500) |
| `limit` | int | `50` | Total pages to process |
| `instructions` | str | `None` | Natural language focus instructions |
| `select_paths` / `exclude_paths` | list | `None` | Regex path filters |
| `select_domains` / `exclude_domains` | list | `None` | Regex domain filters |
| `extract_depth` | str | `basic` | `basic` / `advanced` |
| `timeout` | float | `150.0` | Max crawl time (10–150 s) |

---

### `tavily_map`
Generate a URL structure map of a website.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `url` | str | required | Root URL to map |
| `max_depth` | int | `1` | Link-hop depth (1–5) |
| `max_breadth` | int | `20` | Max links per page |
| `limit` | int | `50` | Total links to process |
| `instructions` | str | `None` | Natural language link selection guidance |
| `select_paths` / `exclude_paths` | list | `None` | Regex path filters |
| `timeout` | float | `150.0` | Max mapping time (10–150 s) |

---

### `tavily_research`
Comprehensive multi-source research report with automatic search planning.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input` | str | required | Research question or task |
| `model` | str | `auto` | `mini` / `pro` / `auto` |
| `stream` | bool | `False` | Stream SSE progress events |
| `output_schema` | dict | `None` | JSON Schema for structured output |
| `citation_format` | str | `numbered` | `numbered` / `mla` / `apa` / `chicago` |

---

## Logging

All tool calls are logged to stdout with:
- **Entry**: tool name, key input parameters
- **Exit**: result count / status, response time
- **Error**: exception type and message

Control verbosity via `LOG_LEVEL` in `.env`:
```env
LOG_LEVEL=DEBUG   # full detail
LOG_LEVEL=INFO    # normal (default)
LOG_LEVEL=WARNING # errors only
```

---

## Project structure

```
mcp-server/
  main.py          ← entry point; builds ASGI app, runs uvicorn on :8001
  pyproject.toml   ← standalone UV project (isolated from root workspace)
  uv.lock          ← pinned lockfile
  .env             ← secrets (not committed)
  .env.example     ← template
  src/
    server.py      ← FastMCP instance + all 5 Tavily tools + logger
```

