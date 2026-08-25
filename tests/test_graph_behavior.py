"""Offline behaviour tests for graph invariants.

These never call a provider: ``nodes.get_llm`` is stubbed, so the whole graph runs
deterministically. They assert the properties that must hold for *any* ticket — no
scenario ids, no exact query strings:

* a risky action is approved BEFORE the tool runs;
* a rejected approval never reaches the tool;
* the retry loop is bounded and dead-ends at ``dead_letter``;
* every route terminates at exactly one ``finalize`` event.
"""

import importlib.util
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("langgraph") is None, reason="langgraph not installed"
)

from langgraph_agent_lab import nodes
from langgraph_agent_lab.graph import NODE_NAMES, build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import ApprovalDecision, Route, Scenario, initial_state


class _StubStructured:
    def __init__(self, schema, route, verdict):
        self._schema = schema
        self._route = route
        self._verdict = verdict

    def invoke(self, _prompt):
        if self._schema is nodes.RouteDecision:
            return nodes.RouteDecision(
                route=self._route,
                risk_level="high" if self._route == "risky" else "low",
                reason="stubbed",
            )
        return nodes.JudgeVerdict(verdict=self._verdict, reason="stubbed")


class _StubLLM:
    def __init__(self, route, verdict):
        self._route = route
        self._verdict = verdict

    def with_structured_output(self, schema):
        return _StubStructured(schema, self._route, self._verdict)

    def invoke(self, _prompt):
        return SimpleNamespace(content="stubbed generated text")


@pytest.fixture
def stub_llm(monkeypatch):
    def _install(route: str, verdict: str = "success") -> None:
        monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: _StubLLM(route, verdict))

    return _install


def _run(route: str, query: str = "generic ticket text", max_attempts: int = 3) -> dict:
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    scenario = Scenario(
        id="behaviour",
        query=query,
        expected_route=Route(route),
        max_attempts=max_attempts,
    )
    state = initial_state(scenario)
    return graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})


def _nodes_visited(result: dict) -> list[str]:
    return [event.get("node") for event in result.get("events", [])]


def test_all_eleven_nodes_registered():
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    registered = set(graph.get_graph().nodes) - {"__start__", "__end__"}
    assert registered == set(NODE_NAMES)
    assert len(NODE_NAMES) == 11


@pytest.mark.parametrize("route", ["simple", "tool", "missing_info", "risky", "error"])
def test_every_route_terminates_at_exactly_one_finalize(stub_llm, route):
    stub_llm(route)
    result = _run(route)
    visited = _nodes_visited(result)
    assert visited.count("finalize") == 1
    assert visited[-1] == "finalize"
    assert result.get("final_answer") or result.get("pending_question")


def test_approval_is_observed_before_the_tool_runs(stub_llm):
    stub_llm("risky")
    result = _run("risky", query="refund this customer and email them")
    visited = _nodes_visited(result)
    assert "approval" in visited and "tool" in visited
    assert visited.index("approval") < visited.index("tool")
    assert result["approval"]["approved"] is True
    assert result["proposed_action"]


def test_rejected_approval_goes_to_clarification_and_never_touches_the_tool(
    stub_llm, monkeypatch
):
    stub_llm("risky")
    rejected = ApprovalDecision(
        approved=False, reviewer="test-reviewer", comment="not permitted by policy"
    )

    def _reject(state):
        return {
            "approval": rejected.model_dump(),
            "events": [nodes.make_event("approval", "rejected", "rejected by reviewer")],
        }

    monkeypatch.setattr(nodes, "approval_node", _reject)
    result = _run("risky", query="delete this account right now")
    visited = _nodes_visited(result)
    assert "tool" not in visited
    assert visited.index("approval") < visited.index("clarify")
    assert result.get("pending_question")


def test_tool_node_refuses_a_risky_action_without_approval():
    update = nodes.tool_node({"route": "risky", "attempt": 0, "query": "refund order 1"})
    assert "ERROR" in update["tool_results"][0]
    assert update["events"][0]["event_type"] == "blocked"


def test_retry_loop_is_bounded_and_increments_once_per_visit(stub_llm):
    stub_llm("error")
    result = _run("error", query="timeout failure in the billing service", max_attempts=3)
    visited = _nodes_visited(result)
    assert visited.count("retry") == result["attempt"] == 2
    assert "dead_letter" not in visited
    assert visited.count("tool") == 2


def test_retry_budget_of_one_dead_letters_without_calling_the_tool(stub_llm):
    stub_llm("error")
    result = _run("error", query="system failure cannot recover", max_attempts=1)
    visited = _nodes_visited(result)
    assert visited.count("retry") == 1
    assert "tool" not in visited
    assert visited.index("dead_letter") < visited.index("finalize")
    assert "escalated" in result["final_answer"].lower()


def test_nodes_do_not_mutate_incoming_lists():
    state = {"route": "error", "attempt": 1, "max_attempts": 3, "errors": [], "tool_results": []}
    errors_before = state["errors"]
    nodes.retry_or_fallback_node(state)
    nodes.tool_node(state)
    assert errors_before == [] and state["errors"] == [] and state["attempt"] == 1
