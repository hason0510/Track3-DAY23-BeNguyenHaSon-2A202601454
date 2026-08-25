"""State schema for the Day 08 LangGraph lab.

Students should extend the schema only when needed. Keep state lean and serializable.
"""

from __future__ import annotations

from enum import StrEnum
from operator import add
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field, field_validator


class Route(StrEnum):
    SIMPLE = "simple"
    TOOL = "tool"
    MISSING_INFO = "missing_info"
    RISKY = "risky"
    ERROR = "error"
    DEAD_LETTER = "dead_letter"
    DONE = "done"


class LabEvent(BaseModel):
    """Append-only audit event for grading and debugging."""

    node: str
    event_type: str
    message: str
    latency_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalDecision(BaseModel):
    approved: bool = False
    reviewer: str = "mock-reviewer"
    comment: str = ""


class AgentState(TypedDict, total=False):
    """LangGraph state.

    Reducer policy:

    * Append-only (``Annotated[list, add]``): ``messages``, ``tool_results``, ``errors``,
      ``events``. These are chronological audit trails - a node returns only the *new*
      entries and LangGraph merges them, so no node may mutate the incoming list.
    * Overwrite (plain annotation): every scalar / "current value" field. ``route``,
      ``attempt``, ``evaluation_result``, ``approval`` and friends describe the state
      *right now*, so the latest write wins.

    All values stay JSON-serializable so any checkpointer backend can persist them.
    """

    thread_id: str
    scenario_id: str
    query: str
    route: str
    risk_level: str
    attempt: int
    max_attempts: int
    final_answer: str | None
    # --- student-added fields -------------------------------------------------
    # Retry-loop gate read by route_after_evaluate ("success" | "needs_retry").
    evaluation_result: str
    # Clarification flow: the question we send back to the user.
    pending_question: str | None
    # Risky flow: the side effect proposed *before* any tool runs.
    proposed_action: str | None
    # HITL flow: plain serializable mapping shaped like ApprovalDecision.
    approval: dict[str, Any] | None
    # --- append-only audit trails ---------------------------------------------
    messages: Annotated[list[str], add]
    tool_results: Annotated[list[str], add]
    errors: Annotated[list[str], add]
    events: Annotated[list[dict[str, Any]], add]


class Scenario(BaseModel):
    id: str
    query: str
    expected_route: Route
    requires_approval: bool = False
    should_retry: bool = False
    max_attempts: int = 3
    tags: list[str] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")
        return value


def initial_state(scenario: Scenario) -> AgentState:
    """Create a serializable initial state for one scenario."""
    return {
        "thread_id": f"thread-{scenario.id}",
        "scenario_id": scenario.id,
        "query": scenario.query,
        "route": "",
        "risk_level": "unknown",
        "attempt": 0,
        "max_attempts": scenario.max_attempts,
        "final_answer": None,
        "evaluation_result": "",
        "pending_question": None,
        "proposed_action": None,
        "approval": None,
        "messages": [],
        "tool_results": [],
        "errors": [],
        "events": [],
    }


def make_event(node: str, event_type: str, message: str, **metadata: Any) -> dict[str, Any]:
    """Create a normalized event payload.

    ``latency_ms`` is promoted out of the metadata into the typed field so metrics can
    aggregate it without knowing each node's private metadata keys.
    """
    latency_ms = int(metadata.pop("latency_ms", 0) or 0)
    event = LabEvent(
        node=node,
        event_type=event_type,
        message=message,
        latency_ms=latency_ms,
        metadata=metadata,
    )
    return event.model_dump()
