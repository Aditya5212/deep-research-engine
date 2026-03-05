from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.tavily_mcp_server import mcp as tavily_mcp
from src.research.agent import close_agent
from src.research.checkpointer import close_checkpointer
from src.research.controller import router as research_router

# Build the MCP ASGI app — exposes Streamable HTTP transport at /tavily/mcp
tavily_mcp_app = tavily_mcp.http_app(path="/mcp")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the MCP server lifespan alongside FastAPI
    async with tavily_mcp_app.lifespan(app):
        yield
        await close_agent()
        await close_checkpointer()


app = FastAPI(lifespan=lifespan)

app.include_router(research_router)

# Mount MCP server — reachable at /tavily/mcp
app.mount("/tavily", tavily_mcp_app)


@app.get("/")
def root():
    return {"message": "Hello from deep-research-engine!"}


@app.get("/health")
def health():
    return {"status": "ok"}
