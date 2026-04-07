import traceback
import uuid
from datetime import datetime, UTC
from typing import Annotated

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
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel
from src.app_logger import get_logger
from src.research.state import ResearchAgentState

logger = get_logger("research.delegate_tool")


class DelegateResearchInput(BaseModel):
    sub_question: str
    domain: str  # finance | medical | tech | legal | general
    context_id: str | None = None  # None = fresh thread; existing id = continue that thread


RESEARCH_AGENT_URL = "http://localhost:8000/agent/"  # your mounted A2A server

@tool("delegate_research_task", args_schema=DelegateResearchInput, return_direct=False)
async def delegate_research_task(
    sub_question: str,
    domain: str,
    state: Annotated[ResearchAgentState, InjectedState],
    context_id: str | None = None,
) -> Command:
    """
    Delegate a sub-question to a domain research A2A sub-agent.

    The orchestrator controls which agent thread handles the request:

    - Pass context_id=None (or omit it) to spawn a FRESH agent thread.
      Use this when researching a new topic that needs its own isolated context.

    - Pass an existing context_id from list_delegated_tasks to CONTINUE that
      agent thread. The sub-agent will have full memory of its prior searches
      and can build on them without re-searching from scratch.

    Always call list_delegated_tasks first to see what threads already exist
    before deciding whether to reuse or create a new one.

    Writes the delegation record into ResearchAgentState.delegated_tasks keyed
    by the A2A task_id. Returns structured findings, citations, confidence,
    and identified_gaps.
    """
    # ── Resolve context_id (= A2A thread_id for the sub-agent) ────────────────
    # None  → orchestrator wants a fresh isolated thread for this sub-question.
    # value → orchestrator chose to continue an existing thread (keeps memory).
    resolved_context_id: str = context_id or uuid.uuid4().hex
    is_new = context_id is None
    logger.info(
        "[DELEGATE] domain=%s | context_id=%s | new_thread=%s | question=%r",
        domain, resolved_context_id, is_new, sub_question,
    )

    try:
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as httpx_client:
            # ── Connect to A2A server ──────────────────────────────────────────
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
                return Command(update={}, result={
                    "error": type(exc).__name__,
                    "message": f"Failed to connect to A2A research agent: {exc}",
                    "findings": "",
                    "citations": [],
                    "confidence": "low",
                    "identified_gaps": [f"Agent connection error: {exc}"],
                    "context_id": resolved_context_id,
                    "task_id": "",
                })

            message = Message(
                role=Role.user,
                message_id=uuid.uuid4().hex,
                # task_id intentionally omitted — A2A SDK generates a fresh one per request.
                # context_id is the stable thread identifier chosen by the orchestrator.
                context_id=resolved_context_id,
                parts=[Part(root=TextPart(text=sub_question))],
            )

            returned_task_id: str = ""
            artifacts: list[dict] = []

            # ── Stream events from A2A server ──────────────────────────────────
            try:
                async for event in client.send_message(
                    message,
                    request_metadata={"domain": domain},
                ):
                    if isinstance(event, tuple):
                        task_obj, update = event
                        if getattr(task_obj, "id", None):
                            returned_task_id = task_obj.id
                        if isinstance(update, TaskArtifactUpdateEvent):
                            for part in update.artifact.parts or []:
                                p = part.root if hasattr(part, "root") else part
                                if hasattr(p, "data"):
                                    artifacts.append(p.data)
            except Exception as exc:
                logger.error(
                    "[DELEGATE] send_message failed | domain=%s | context_id=%s | %s: %s\n%s",
                    domain, resolved_context_id, type(exc).__name__, exc, traceback.format_exc(),
                )
                return Command(update={}, result={
                    "error": type(exc).__name__,
                    "message": f"A2A agent communication error: {exc}",
                    "findings": "",
                    "citations": [],
                    "confidence": "low",
                    "identified_gaps": [f"Communication error: {exc}"],
                    "context_id": resolved_context_id,
                    "task_id": returned_task_id,
                })

            merged = artifacts[-1] if artifacts else {}
            if not merged:
                logger.warning(
                    "[DELEGATE] No artifacts received | domain=%s | context_id=%s",
                    domain, resolved_context_id,
                )

            task_key = returned_task_id or uuid.uuid4().hex
            result_payload = {
                "findings":        merged.get("findings", ""),
                "citations":       merged.get("citations", []),
                "confidence":      merged.get("confidence", "low"),
                "identified_gaps": merged.get("identified_gaps", []),
                "context_id":      resolved_context_id,
                "task_id":         task_key,
            }

            # ── Persist delegation record into graph state ──────────────────────
            # Command(update=...) is applied atomically by LangGraph's ToolNode
            # before the ToolMessage is added to messages — the orchestrator sees
            # the updated delegated_tasks on its very next step.
            state_update = {
                "delegated_tasks": {
                    task_key: {
                        "sub_question":  sub_question,
                        "domain":        domain,
                        "context_id":    resolved_context_id,
                        "confidence":    merged.get("confidence", "low"),
                        "delegated_at":  datetime.now(UTC).isoformat(),
                    }
                }
            }
            logger.info(
                "[DELEGATE] Persisting task | task_id=%s | domain=%s | context_id=%s | confidence=%s",
                task_key, domain, resolved_context_id, merged.get("confidence", "low"),
            )
            return Command(update=state_update, result=result_payload)

    except Exception as exc:
        logger.error(
            "[DELEGATE] Unexpected error | domain=%s | %s: %s\n%s",
            domain, type(exc).__name__, exc, traceback.format_exc(),
        )
        return Command(update={}, result={
            "error": type(exc).__name__,
            "message": str(exc),
            "findings": "",
            "citations": [],
            "confidence": "low",
            "identified_gaps": [f"Unexpected error: {exc}"],
            "context_id": resolved_context_id,
            "task_id": "",
        })
