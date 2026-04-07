from typing import Annotated, NotRequired

from langchain.agents.middleware.types import AgentState


def _merge_dicts(a: dict, b: dict) -> dict:
    """Reducer for delegated_tasks — merges new entries, never overwrites existing ones."""
    return {**a, **b}


class ResearchAgentState(AgentState):
    """
    Extended state schema for the main research orchestrator agent.

    Extra fields (on top of AgentState.messages):

        delegated_tasks — dict[task_id, DelegatedTaskEntry] persisted across turns.

        Schema per entry:
            {
              "sub_question": str,      # the question sent to the A2A agent
              "domain":       str,      # finance | medical | tech | legal | general
              "context_id":   str,      # A2A context_id == A2A thread_id (stable per domain)
              "confidence":   str,      # high | medium | low (from last result)
              "delegated_at": str,      # ISO-8601 UTC timestamp
            }

    The _merge_dicts reducer means state updates always ADD new task entries;
    existing entries are never overwritten by a merge.
    """
    delegated_tasks: NotRequired[Annotated[dict, _merge_dicts]]
