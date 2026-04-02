from contextlib import asynccontextmanager
import sys
from pathlib import Path

# Provide access to a2a-server code
a2a_src = str(Path(__file__).parent / "a2a-server" / "src")
if a2a_src not in sys.path:
    sys.path.append(a2a_src)

from fastapi import FastAPI
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill
from agent_executor import ResearchAgentExecutor

from src.tavily_mcp_server import mcp as tavily_mcp
from src.research.agent import close_agent
from src.research.checkpointer import close_checkpointer
from src.research.controller import router as research_router

# Build the MCP ASGI app — exposes Streamable HTTP transport at /tavily/mcp
tavily_mcp_app = tavily_mcp.http_app(path="/mcp")

# --- Initialize A2A Agent Server ---
research_skill = AgentSkill(
    id="deep_research",
    name="Deep Domain Research",
    description="Researches a sub-question with iterative search and structured findings.",
    tags=["research", "search", "analysis"],
    examples=["What are latest EU AI regulations?"],
    input_modes=["text"],
    output_modes=["text", "data"],
)

agent_card = AgentCard(
    name="Deep Research Domain Agent",
    description="Domain-aware deep research agent with streaming.",
    url="http://localhost:8000/agent",
    version="1.0.0",
    default_input_modes=["text"],
    default_output_modes=["text", "data"],
    capabilities=AgentCapabilities(streaming=True),
    skills=[research_skill],
)

a2a_handler = DefaultRequestHandler(
    agent_executor=ResearchAgentExecutor(),
    task_store=InMemoryTaskStore(),
)

a2a_app = A2AStarletteApplication(
    agent_card=agent_card,
    http_handler=a2a_handler,
).build()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start all inner lifespans alongside FastAPI
    async with a2a_app.router.lifespan_context(app):
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


app.mount("/agent", a2a_app)
