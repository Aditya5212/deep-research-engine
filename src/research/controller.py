from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.research.service import (
    TransientAPIError,
    chat_with_agent,
    get_pending_interrupts,
    get_thread_history,
    get_thread_state,
    resume_agent,
)

router = APIRouter(prefix="/research", tags=["research"])


# ---------------------------------------------------------------------------
# Shared DTOs
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class ActionRequest(BaseModel):
    """A single tool call awaiting human review."""
    name: str
    args: dict[str, Any]
    description: str | None = None


class ReviewConfig(BaseModel):
    """Describes which decisions are allowed for this tool call."""
    action_name: str
    allowed_decisions: list[Literal["approve", "edit", "reject"]]


class HITLRequestValue(BaseModel):
    """The structured payload inside each interrupt's ``value`` field."""
    action_requests: list[ActionRequest]
    review_configs: list[ReviewConfig]


class InterruptValue(BaseModel):
    """A single pending HITL interrupt."""
    id: str
    value: HITLRequestValue


class ChatResponse(BaseModel):
    reply: str | None = None
    thread_id: str
    interrupted: bool = False
    interrupts: list[InterruptValue] = []


# ---------------------------------------------------------------------------
# Decision models for HITL resume
# ---------------------------------------------------------------------------

class ApproveDecision(BaseModel):
    type: Literal["approve"] = "approve"


class EditedAction(BaseModel):
    name: str
    args: dict[str, Any]


class EditDecision(BaseModel):
    type: Literal["edit"] = "edit"
    edited_action: EditedAction


class RejectDecision(BaseModel):
    type: Literal["reject"] = "reject"
    message: str | None = None


class ResumeRequest(BaseModel):
    """Human decisions for all pending action_requests in the current interrupt.

    Provide one decision per ``action_request`` listed in the interrupt value.
    Each decision must be one of ``ApproveDecision``, ``EditDecision``, or
    ``RejectDecision``.
    """
    decisions: list[ApproveDecision | EditDecision | RejectDecision]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Send a message to the deep research agent.

    If the agent triggers a Human-in-the-Loop checkpoint (e.g., for
    ``tavily_research``, ``tavily_extract``, or ``tavily_crawl``), the response
    will have ``interrupted=True`` and ``interrupts`` populated with the pending
    action requests.  Use ``POST /research/session/{thread_id}/resume`` to
    provide decisions and continue execution.
    """
    try:
        result = await chat_with_agent(
            message=request.message,
            thread_id=request.thread_id,
        )
    except TransientAPIError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"The AI model is temporarily unavailable. Please try again in a moment. ({exc})",
        ) from exc
    return ChatResponse(
        reply=result["reply"],
        thread_id=request.thread_id,
        interrupted=result["interrupted"],
        interrupts=[InterruptValue(**i) for i in result["interrupts"]],
    )


@router.get("/session/{thread_id}/interrupts", response_model=list[InterruptValue])
async def session_interrupts(thread_id: str) -> list[InterruptValue]:
    """Get any pending HITL interrupts for a session without invoking the agent.

    Returns an empty list if the agent is not currently paused.
    Each entry exposes the full ``HITLRequest`` (``action_requests`` +
    ``review_configs``) so the caller knows exactly which tool calls are pending
    and which decisions are allowed.
    """
    raw = await get_pending_interrupts(thread_id)
    return [InterruptValue(**i) for i in raw]


@router.post("/session/{thread_id}/resume", response_model=ChatResponse)
async def session_resume(thread_id: str, body: ResumeRequest) -> ChatResponse:
    """Resume an interrupted agent session with human decisions.

    Provide one decision per pending ``action_request`` in the interrupt.
    The agent will continue executing after the approved/edited/rejected tool
    calls and may pause again if another HITL checkpoint is reached.
    """
    try:
        result = await resume_agent(
            thread_id=thread_id,
            decisions=[d.model_dump() for d in body.decisions],
        )
    except TransientAPIError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"The AI model is temporarily unavailable. Please try again in a moment. ({exc})",
        ) from exc
    return ChatResponse(
        reply=result["reply"],
        thread_id=thread_id,
        interrupted=result["interrupted"],
        interrupts=[InterruptValue(**i) for i in result["interrupts"]],
    )


@router.get("/session/{thread_id}/state")
async def session_state(thread_id: str, checkpoint_id: str | None = None):
    """Get the current (or a specific checkpoint) state snapshot for a session/thread."""
    return await get_thread_state(thread_id, checkpoint_id)


@router.get("/session/{thread_id}/history")
async def session_history(thread_id: str):
    """Get the full state history (all checkpoints) for a session/thread."""
    return await get_thread_history(thread_id)
