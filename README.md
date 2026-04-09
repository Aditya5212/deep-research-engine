# Deep Research Engine

Multi-service research system built around **three servers**:

1. **MCP Server** — Tavily tools exposed over Streamable HTTP
2. **A2A Server** — remote domain research agent (cloud-ready)
3. **Main Orchestrator** — FastAPI app that handles user queries, HITL, and delegates work

---

## Architecture

```
User / UI
  ↓
Main Orchestrator (FastAPI, :8000)
  ├── direct MCP tools (tavily_search/extract/crawl/map)
  └── delegate_research_task → A2A Server (domain agent)
                                   └── uses MCP Server for Tavily tools
MCP Server (FastMCP, :8001)
```

## Why this architecture

- **Isolation**: MCP and A2A run in their own processes with separate deps and configs.
- **Scalability**: A2A can scale independently for heavy research workloads.
- **Cost control**: MCP traffic is centralized and observable; A2A can be throttled.
- **Failure containment**: MCP outages do not crash the orchestrator; failures are localized.
- **Deployment flexibility**: A2A can run on remote/cloud hosts, MCP can run close to Tavily.
- **Faster iteration**: Each service can be updated/redeployed without restarting the whole system.

### Server responsibilities

**1) MCP Server (mcp-server)**
- Standalone Tavily MCP server
- Exposes tools over Streamable HTTP
- Runs on `http://localhost:8001/mcp`

**2) A2A Server (a2a-server)**
- Hosts the domain researcher agent
- Uses MCP tools over HTTP (`MCP_SERVER_URL`)
- Exposes A2A endpoint at `/agent`
- Intended to run as a separate service in cloud or on another machine

**3) Main Orchestrator (root main.py)**
- FastAPI app for `/research/*` endpoints and HITL flows
- Delegates sub-questions to A2A via `AGENT_URL`
- Also calls MCP tools directly for quick lookups

---

## Requirements

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (for Postgres via `docker compose`)

---

## Setup

### 1) Main orchestrator
```bash
uv sync
cp .env.example .env
```

Set in `.env`:
```
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=deep_research
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

AGENT_URL=http://localhost:8000/agent/
MCP_SERVER_URL=http://localhost:8001/mcp
```

### 2) MCP server
```bash
cd mcp-server
cp .env.example .env
uv sync
```

### 3) A2A server
```bash
cd a2a-server
cp .env.example .env
uv sync
```

---

## Running the stack (local)

### Start Postgres
```bash
docker compose up -d
```

### Start MCP server
```bash
cd mcp-server
uv run start
```

### Start main orchestrator
```bash
cd ..
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### A2A server
For local dev, the A2A endpoint is mounted by the main orchestrator at:

```
http://localhost:8000/agent
```

If running A2A separately (recommended for production), start it on its own host/port
and set `AGENT_URL` in `.env` to that URL. The A2A server should expose `/agent`.

---

## MCP tools (Tavily)

These tools are provided by the standalone MCP server and are available to both
the A2A server and the orchestrator:

- `tavily_search`
- `tavily_extract`
- `tavily_crawl`
- `tavily_map`
- `tavily_research`

See the full tool reference and examples in [mcp-server/README.md](mcp-server/README.md).

---

## API endpoints (main orchestrator)

- `GET /` — welcome message
- `GET /health` — health check
- `POST /research/chat` — main chat endpoint
- `POST /research/session/{thread_id}/resume` — HITL resume
- `GET /research/session/{thread_id}/history` — state history

Full API docs at `http://localhost:8000/docs`.

---

## Project structure

```
deep-research-engine/
├── main.py                    # Orchestrator FastAPI app
├── src/
│   └── research/               # Orchestrator agent + HITL + delegation
│       ├── agent.py
│       ├── delegate_tool.py
│       ├── controller.py
│       └── service.py
├── a2a-server/                 # Remote domain research agent (A2A)
├── mcp-server/                 # Standalone Tavily MCP server
├── pyproject.toml
├── docker-compose.yml
└── .env.example
```

---

## License

MIT
