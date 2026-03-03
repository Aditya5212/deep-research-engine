from fastapi import APIRouter
from pydantic import BaseModel

from src.research.service import chat_with_agent, get_thread_state, get_thread_history

router = APIRouter(prefix="/research", tags=["research"])


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Send a message to the deep research agent and get a response."""
    reply = await chat_with_agent(
        message=request.message,
        thread_id=request.thread_id,
    )
    return ChatResponse(reply=reply, thread_id=request.thread_id)


@router.get("/session/{thread_id}/state")
async def session_state(thread_id: str, checkpoint_id: str | None = None):
    """Get the current (or a specific checkpoint) state snapshot for a session/thread."""
    return await get_thread_state(thread_id, checkpoint_id)


@router.get("/session/{thread_id}/history")
async def session_history(thread_id: str):
    """Get the full state history (all checkpoints) for a session/thread."""
    return await get_thread_history(thread_id)
