import traceback
import uuid
import httpx
from a2a.client.client_factory import ClientFactory
from a2a.client.client import ClientConfig
from a2a.types import (
    Message,
    Role,
    Part,
    TextPart,
    TaskArtifactUpdateEvent,
)
from langchain_core.tools import tool
from pydantic import BaseModel
from src.app_logger import get_logger

logger = get_logger("research.delegate_tool")

class DelegateResearchInput(BaseModel):
    sub_question: str
    domain: str                  # finance | medical | tech | legal | general
    context_id: str | None = None  # pass existing contextId for multi-turn
    task_id: str | None = None

RESEARCH_AGENT_URL = "http://localhost:8000/agent/"  # your mounted A2A server

@tool("delegate_research_task", args_schema=DelegateResearchInput, return_direct=False)
async def delegate_research_task(
    sub_question: str,
    domain: str,
    context_id: str | None = None,
    task_id: str | None = None,
) -> dict:
    """
    Delegate a sub-question to the domain research A2A agent.
    Pass context_id and task_id to continue an existing research thread (multi-turn).
    Omit both to start a fresh thread.
    Returns structured findings, citations, confidence, and gaps.
    """
    try:
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as httpx_client:
            # Use ClientFactory (replaces deprecated A2AClient)
            try:
                client = await ClientFactory.connect(
                    agent=RESEARCH_AGENT_URL,
                    client_config=ClientConfig(httpx_client=httpx_client),
                )
            except Exception as exc:
                logger.error(
                    "[DELEGATE] ClientFactory.connect failed | domain=%s | %s: %s\n%s",
                    domain, type(exc).__name__, exc, traceback.format_exc(),
                )
                return {
                    "error": type(exc).__name__,
                    "message": f"Failed to connect to A2A research agent: {exc}",
                    "findings": "",
                    "citations": [],
                    "confidence": "low",
                    "identified_gaps": [f"Agent connection error: {exc}"],
                    "context_id": context_id or "",
                    "task_id": task_id or "",
                }

            # Build the Message directly — new Client.send_message() takes a Message, not a SendMessageRequest
            message = Message(
                role=Role.user,
                message_id=uuid.uuid4().hex,
                task_id=task_id,
                context_id=context_id,
                parts=[Part(root=TextPart(text=sub_question))],
            )

            returned_context_id = context_id
            returned_task_id = task_id
            artifacts = []

            # Client.send_message() is an async generator yielding (Task, UpdateEvent) or Message
            try:
                async for event in client.send_message(
                    message,
                    request_metadata={"domain": domain},
                ):
                    # event is either (Task, UpdateEvent) tuple or a Message
                    if isinstance(event, tuple):
                        task_obj, update = event
                        # Capture task/context ids from the Task object
                        if getattr(task_obj, "context_id", None):
                            returned_context_id = task_obj.context_id
                        if getattr(task_obj, "id", None):
                            returned_task_id = task_obj.id
                        # Collect artifact data from artifact update events
                        if isinstance(update, TaskArtifactUpdateEvent):
                            for part in update.artifact.parts or []:
                                p = part.root if hasattr(part, "root") else part
                                if hasattr(p, "data"):
                                    artifacts.append(p.data)
            except Exception as exc:
                logger.error(
                    "[DELEGATE] send_message failed | domain=%s | task_id=%s | %s: %s\n%s",
                    domain, returned_task_id, type(exc).__name__, exc, traceback.format_exc(),
                )
                return {
                    "error": type(exc).__name__,
                    "message": f"A2A agent communication error: {exc}",
                    "findings": "",
                    "citations": [],
                    "confidence": "low",
                    "identified_gaps": [f"Communication error: {exc}"],
                    "context_id": returned_context_id or "",
                    "task_id": returned_task_id or "",
                }

            # -- STREAMING variant (same call, no behavior change needed — send_message auto-streams) --
            # The client already handles streaming internally; no separate send_message_streaming needed.

            # Take last artifact as final structured result
            merged = artifacts[-1] if artifacts else {}
            if not merged:
                logger.warning(
                    "[DELEGATE] No artifacts received | domain=%s | task_id=%s",
                    domain, returned_task_id,
                )
            return {
                "findings": merged.get("findings", ""),
                "citations": merged.get("citations", []),
                "confidence": merged.get("confidence", "low"),
                "identified_gaps": merged.get("identified_gaps", []),
                "context_id": returned_context_id or "",
                "task_id": returned_task_id or "",
            }

    except Exception as exc:
        logger.error(
            "[DELEGATE] Unexpected error | domain=%s | %s: %s\n%s",
            domain, type(exc).__name__, exc, traceback.format_exc(),
        )
        return {
            "error": type(exc).__name__,
            "message": str(exc),
            "findings": "",
            "citations": [],
            "confidence": "low",
            "identified_gaps": [f"Unexpected error: {exc}"],
            "context_id": context_id or "",
            "task_id": task_id or "",
        }
