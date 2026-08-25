"""Graph construction.

This module is intentionally import-safe. It imports LangGraph only inside the builder so unit tests
that check schema/metrics can run even if students are still debugging graph wiring.
"""

from __future__ import annotations

from typing import Any

from .state import AgentState

#: Registered graph-node name -> node function attribute in ``nodes.py``.
#: The keys are the public contract: routing functions return exactly these strings.
NODE_NAMES: dict[str, str] = {
    "intake": "intake_node",
    "classify": "classify_node",
    "tool": "tool_node",
    "evaluate": "evaluate_node",
    "answer": "answer_node",
    "clarify": "ask_clarification_node",
    "risky_action": "risky_action_node",
    "approval": "approval_node",
    "retry": "retry_or_fallback_node",
    "dead_letter": "dead_letter_node",
    "finalize": "finalize_node",
}


def build_graph(checkpointer: Any | None = None) -> Any:
    """Build and compile the LangGraph workflow.

    Architecture::

        START → intake → classify → [route_after_classify]
          simple       → answer → finalize → END
          tool         → tool → evaluate → [route_after_evaluate]
                                              success     → answer → finalize → END
                                              needs_retry → retry → [route_after_retry]
                                                                      tool (retry)
                                                                      dead_letter → finalize → END
          missing_info → clarify → finalize → END
          risky        → risky_action → approval → [route_after_approval]
                                                      approved → tool → evaluate → ...
                                                      rejected → clarify → finalize → END
          error        → retry → [route_after_retry] → ...

    The ``checkpointer`` argument is passed straight through to ``compile()``; the caller
    (CLI/tests) owns its lifecycle and backend choice, so this builder never creates one.
    """
    from langgraph.graph import END, START, StateGraph

    from . import nodes
    from .routing import (
        route_after_approval,
        route_after_classify,
        route_after_evaluate,
        route_after_retry,
    )

    builder = StateGraph(AgentState)

    # 1) Register the 11 nodes under their public names.
    for node_name, function_name in NODE_NAMES.items():
        builder.add_node(node_name, getattr(nodes, function_name))

    # 2) Fixed edges — the parts of the flow that never branch.
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "classify")
    builder.add_edge("tool", "evaluate")
    builder.add_edge("risky_action", "approval")
    builder.add_edge("answer", "finalize")
    builder.add_edge("clarify", "finalize")
    builder.add_edge("dead_letter", "finalize")
    builder.add_edge("finalize", END)

    # 3) Conditional edges — path maps mirror the routing decision tables exactly.
    builder.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "answer": "answer",
            "tool": "tool",
            "clarify": "clarify",
            "risky_action": "risky_action",
            "retry": "retry",
        },
    )
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"retry": "retry", "answer": "answer"},
    )
    builder.add_conditional_edges(
        "retry",
        route_after_retry,
        {"tool": "tool", "dead_letter": "dead_letter"},
    )
    builder.add_conditional_edges(
        "approval",
        route_after_approval,
        {"tool": "tool", "clarify": "clarify"},
    )

    return builder.compile(checkpointer=checkpointer)
