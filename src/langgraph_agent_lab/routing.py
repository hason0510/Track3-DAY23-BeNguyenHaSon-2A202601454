"""Routing functions for conditional edges.

Each function takes AgentState and returns a string — the name of the next node.
These strings MUST match node names registered in graph.py.

Routing is pure: read state, return a node name. No LLM calls, no mutation, no side
effects. Every function fails *closed* (towards a terminating branch) when the state
it reads is missing or malformed.
"""

from __future__ import annotations

from .state import AgentState, Route

#: Decision table for the post-classification fan-out.
_CLASSIFY_ROUTES: dict[str, str] = {
    Route.SIMPLE.value: "answer",
    Route.TOOL.value: "tool",
    Route.MISSING_INFO.value: "clarify",
    Route.RISKY.value: "risky_action",
    Route.ERROR.value: "retry",
}


def route_after_classify(state: AgentState) -> str:
    """Map classified route to the next graph node.

    Unknown or missing routes default to ``answer`` so a bad classification still
    terminates with a grounded response instead of dead-ending the graph.
    """
    route = str(state.get("route") or "")
    return _CLASSIFY_ROUTES.get(route, "answer")


def route_after_evaluate(state: AgentState) -> str:
    """Decide if the tool result is satisfactory or needs retry.

    This is the 'done?' check that creates the retry loop — a key LangGraph
    advantage over linear LCEL chains. Only an explicit ``needs_retry`` verdict
    loops; anything else moves forward to the answer.
    """
    if state.get("evaluation_result") == "needs_retry":
        return "retry"
    return "answer"


def route_after_retry(state: AgentState) -> str:
    """Decide whether to retry the tool or give up.

    Bounded by construction: ``retry_or_fallback_node`` owns the counter and has
    already incremented it, so this reads the post-increment value. ``>=`` (not
    ``==``) means an out-of-range counter also fails closed to the dead letter.
    """
    attempt = int(state.get("attempt", 0) or 0)
    max_attempts = int(state.get("max_attempts", 3) or 0)
    return "tool" if attempt < max_attempts else "dead_letter"


def route_after_approval(state: AgentState) -> str:
    """Route based on the human approval decision.

    Approval is a *gate*: only an explicit ``approved is True`` lets the risky side
    effect reach the tool node. Missing/malformed decisions are treated as rejected.
    """
    approval = state.get("approval") or {}
    approved = approval.get("approved") if isinstance(approval, dict) else None
    return "tool" if approved is True else "clarify"
