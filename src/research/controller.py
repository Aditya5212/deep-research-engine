from fastapi import APIRouter
from pydantic import BaseModel

from src.research.service import chat_with_agent

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
