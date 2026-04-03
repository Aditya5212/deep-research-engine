import json
import logging
import time
import traceback
import uuid
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
    TaskStatus,
    TaskState,
    Artifact,
    DataPart,
    Part,
    Message,
    Role,
    TextPart,
)
from agent import researcher_agent_context
from src.app_logger import get_logger

logger = get_logger("a2a.executor")


def _status_event(
    task_id: str,
    context_id: str,
    state: TaskState,
    text: str,
    final: bool = False,
) -> TaskStatusUpdateEvent:
    """Build a correctly-typed TaskStatusUpdateEvent."""
    return TaskStatusUpdateEvent(
        task_id=task_id,
        context_id=context_id,
        final=final,
        status=TaskStatus(
            state=state,
            message=Message(
                role=Role.agent,
                message_id=uuid.uuid4().hex,
                parts=[Part(root=TextPart(text=text))],
            ),
        ),
    )


class ResearchAgentExecutor(AgentExecutor):
    """
    A2A AgentExecutor for a deep research domain agent.
    Delegates all agent logic to agent.py — no agent is built here.
    """

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or str(uuid.uuid4())
        context_id = context.context_id or str(uuid.uuid4())
        user_message = context.get_user_input()
        metadata = context.metadata or {}
        domain = metadata.get("domain", "general")
        sub_question = user_message

        logger.info(
            "[TASK:%s] >>> Execute called | domain=%s | question=%r",
            task_id, domain, sub_question,
        )

        # ── Enqueue: task started ──────────────────────────────────────────────
        logger.debug("[TASK:%s] Enqueuing TaskState.working (started)", task_id)
        await event_queue.enqueue_event(
            _status_event(task_id, context_id, TaskState.working,
                          f"[{domain}] Starting research: '{sub_question}'")
        )

        findings = None
        t_start = time.perf_counter()

        # ── Agent: non-streaming ainvoke (JSON-RPC, message/send) ─────────────
        # Streaming is intentionally disabled here. The agent runs to completion
        # and returns a single response. Enable streaming later via astream()
        # when the client uses message/stream (SSE).
        #
        # -- STREAMING (commented out, re-enable for message/stream support) --
        # async for chunk in agent.astream(
        #     {"messages": [{"role": "user", "content": sub_question}]},
        #     stream_mode="updates",
        # ):
        #     agent_msgs = chunk.get("agent", {}).get("messages", [])
        #     for msg in agent_msgs:
        #         if getattr(msg, "tool_calls", None):
        #             for tc in msg.tool_calls:
        #                 tool_name = tc.get("name", "?")
        #                 tool_query = tc.get("args", {}).get("query", "")
        #                 await event_queue.enqueue_event(
        #                     _status_event(task_id, context_id, TaskState.working,
        #                                   f"[{domain}] Tool: {tool_name} | {tool_query}")
        #                 )
        #         elif getattr(msg, "content", None):
        #             findings = msg.content
        # -----------------------------------------------------------------------

        try:
            logger.info("[TASK:%s] Opening researcher_agent_context domain=%s", task_id, domain)
            async with researcher_agent_context(domain=domain) as agent:
                logger.debug("[TASK:%s] Agent ready — calling ainvoke", task_id)
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": sub_question}]},
                    config={
                        "configurable": {
                            # thread_id scopes the checkpoint to this conversation context.
                            # Using context_id groups all turns in the same session;
                            # use task_id here instead if you want per-task isolation.
                            "thread_id": context_id,
                        }
                    },
                )
                # Extract final AI message content from the returned state
                messages = result.get("messages", [])
                for msg in reversed(messages):
                    if getattr(msg, "content", None) and not getattr(msg, "tool_calls", None):
                        findings = msg.content
                        break

            elapsed = time.perf_counter() - t_start
            logger.info(
                "[TASK:%s] ainvoke complete | elapsed=%.2fs | findings=%s",
                task_id, elapsed, findings is not None,
            )

            # ── Parse findings ─────────────────────────────────────────────────
            try:
                parsed = json.loads(findings) if isinstance(findings, str) else {}
                logger.debug("[TASK:%s] Parsed findings as JSON successfully", task_id)
            except (json.JSONDecodeError, TypeError) as parse_exc:
                logger.warning(
                    "[TASK:%s] JSON parse failed (%s) — using raw string", task_id, parse_exc
                )
                parsed = {
                    "findings": str(findings),
                    "citations": [],
                    "confidence": "low",
                    "identified_gaps": [],
                }

            # ── Enqueue: artifact (final structured result) ────────────────────
            artifact_id = uuid.uuid4().hex
            logger.info("[TASK:%s] Enqueuing TaskArtifactUpdateEvent (artifact_id=%s)", task_id, artifact_id)
            await event_queue.enqueue_event(
                TaskArtifactUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    artifact=Artifact(
                        artifact_id=artifact_id,
                        name="research_findings",
                        parts=[Part(root=DataPart(data=parsed))],
                    ),
                    last_chunk=True,
                )
            )

            # ── Enqueue: completed ─────────────────────────────────────────────
            logger.info("[TASK:%s] Enqueuing TaskState.completed", task_id)
            await event_queue.enqueue_event(
                _status_event(
                    task_id, context_id, TaskState.completed,
                    f"[{domain}] Complete | {elapsed:.2f}s",
                    final=True,
                )
            )
            logger.info("[TASK:%s] <<< Execute finished", task_id)

        except Exception as exc:
            elapsed = time.perf_counter() - t_start
            logger.error(
                "[TASK:%s] Unhandled exception | domain=%s | elapsed=%.2fs | %s: %s\n%s",
                task_id, domain, elapsed,
                type(exc).__name__, exc,
                traceback.format_exc(),
            )
            try:
                await event_queue.enqueue_event(
                    _status_event(
                        task_id, context_id, TaskState.failed,
                        json.dumps({
                            "error": type(exc).__name__,
                            "message": str(exc),
                            "domain": domain,
                            "elapsed_s": round(elapsed, 2),
                        }),
                        final=True,
                    )
                )
            except Exception as eq_exc:
                logger.error(
                    "[TASK:%s] Failed to enqueue error event: %s", task_id, eq_exc
                )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or "unknown"
        context_id = context.context_id or "unknown"
        logger.warning("[TASK:%s] Cancel requested", task_id)
        await event_queue.enqueue_event(
            _status_event(task_id, context_id, TaskState.canceled,
                          "Research task canceled", final=True)
        )
        logger.info("[TASK:%s] Enqueued TaskState.canceled", task_id)
