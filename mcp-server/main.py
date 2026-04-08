"""Standalone Tavily MCP Server.

Exposes all Tavily tools (search, extract, crawl, map, research) via the
MCP Streamable HTTP transport.

Run:
    uv run start
    # or directly:
    uv run uvicorn main:asgi_app --host 0.0.0.0 --port 8001 --reload
"""

import uvicorn

from src.server import mcp

# ASGI app — mountable or directly served by uvicorn
asgi_app = mcp.http_app(path="/mcp")


def run_server() -> None:
    """Entry point invoked by `uv run start`."""
    uvicorn.run(
        "main:asgi_app",
        host="0.0.0.0",
        port=8001,
        reload=False,
    )


if __name__ == "__main__":
    run_server()
