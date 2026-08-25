"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

Contract used by every node below::

    read current state
      -> compute new values in local variables
      -> return a partial update dict
      -> let the LangGraph reducers merge it

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)

Failure policy (explicit, never silent): an LLM/provider error is recorded in ``errors``
and emitted as a ``failed`` event. Where a controlled fallback exists it is marked in the
event metadata (``fallback=True``); a provider error is never turned into a response that
pretends to have succeeded.
"""

from __future__ import annotations

import os
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, ApprovalDecision, Route, make_event

_MAX_CONTEXT_CHARS = 600
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _elapsed_ms(start: float) -> int:
    """Milliseconds since ``start`` (``time.perf_counter`` reference)."""
    return int((time.perf_counter() - start) * 1000)


def _text_of(response: Any) -> str:
    """Extract plain text from a LangChain message across provider shapes."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        ]
        return "".join(parts).strip()
    return str(content).strip()


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict[str, Any]:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── classify ────────────────────────────────────────────────────────


class RouteDecision(BaseModel):
    """Structured-output contract for the classifier."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"] = Field(
        description="Route bucket for this support ticket."
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        description="How damaging an unreviewed action on this ticket would be."
    )
    reason: str = Field(default="", description="One short sentence justifying the route.")


_CLASSIFIER_PROMPT = """You are the intent classifier of a support-ticket agent.

Classify the ticket into exactly one route:
- risky: the user asks for an action with a real side effect (refund, delete, cancel,
  send email, modify data on their behalf, charge, deploy).
- tool: the user asks for information that requires a lookup (order status, tracking,
  account records, search) but no state change.
- missing_info: the ticket is too vague or incomplete to act on (no object, no id,
  no describable problem).
- error: the ticket reports a system/technical failure (timeout, crash, service
  unavailable, cannot recover, 5xx).
- simple: a general question answerable from knowledge, with no lookup and no action.

Priority when several buckets seem to fit: risky > tool > missing_info > error > simple.
Set risk_level=high for risky tickets, otherwise low or medium.

Ticket:
{query}
"""

_RISKY_MARKERS = (
    "refund",
    "delete",
    "remove",
    "cancel",
    "send ",
    "email",
    "charge",
    "deploy",
    "close account",
)
_TOOL_MARKERS = ("lookup", "look up", "status", "track", "find", "search", "check", "order")
_ERROR_MARKERS = ("timeout", "failure", "failed", "crash", "cannot recover", "unavailable", "5xx")


def _fallback_route(query: str) -> tuple[str, str]:
    """Keyword fallback used ONLY when the LLM call fails. Never the primary path."""
    text = query.lower()
    if any(marker in text for marker in _RISKY_MARKERS):
        return Route.RISKY.value, "high"
    if any(marker in text for marker in _TOOL_MARKERS):
        return Route.TOOL.value, "low"
    if len(text.split()) <= 4:
        return Route.MISSING_INFO.value, "low"
    if any(marker in text for marker in _ERROR_MARKERS):
        return Route.ERROR.value, "medium"
    return Route.SIMPLE.value, "low"


def classify_node(state: AgentState) -> dict[str, Any]:
    """Classify the query into a route using an LLM with structured output.

    Primary path: ``get_llm().with_structured_output(RouteDecision)``. Only when the
    provider call or validation fails do we drop to a keyword fallback, and that
    degradation is written to ``errors`` and flagged in the event metadata.
    """
    start = time.perf_counter()
    query = (state.get("query") or "").strip()

    if not query:
        return {
            "route": Route.MISSING_INFO.value,
            "risk_level": "low",
            "errors": ["classify: empty query, cannot classify"],
            "events": [
                make_event(
                    "classify",
                    "completed",
                    "empty query classified as missing_info",
                    latency_ms=_elapsed_ms(start),
                    route=Route.MISSING_INFO.value,
                    structured_output=False,
                )
            ],
        }

    try:
        classifier = get_llm().with_structured_output(RouteDecision)
        raw = classifier.invoke(_CLASSIFIER_PROMPT.format(query=query))
        decision = raw if isinstance(raw, RouteDecision) else RouteDecision.model_validate(raw)
    except Exception as exc:  # provider error, schema violation, network failure
        route, risk_level = _fallback_route(query)
        return {
            "route": route,
            "risk_level": risk_level,
            "errors": [f"classify: LLM classification failed ({type(exc).__name__}): {exc}"],
            "events": [
                make_event(
                    "classify",
                    "failed",
                    "structured classification failed, keyword fallback applied",
                    latency_ms=_elapsed_ms(start),
                    route=route,
                    risk_level=risk_level,
                    structured_output=False,
                    fallback=True,
                )
            ],
        }

    route = decision.route
    risk_level = "high" if route == Route.RISKY.value else decision.risk_level
    return {
        "route": route,
        "risk_level": risk_level,
        "messages": [f"classify:{route}"],
        "events": [
            make_event(
                "classify",
                "completed",
                f"classified as {route}",
                latency_ms=_elapsed_ms(start),
                route=route,
                risk_level=risk_level,
                reason=decision.reason[:200],
                structured_output=True,
            )
        ],
    }


# ─── tool ────────────────────────────────────────────────────────────


def tool_node(state: AgentState) -> dict[str, Any]:
    """Execute a mock tool call.

    Two guarantees matter more than the mock payload itself:

    1. A risky action only executes when an approval decision is on record — the node
       fails closed otherwise, so the approval gate cannot be bypassed by an edge.
    2. Error-route tickets fail transiently for the first two attempts so the retry
       loop is exercised without any scenario-id knowledge.
    """
    start = time.perf_counter()
    route = str(state.get("route") or "")
    attempt = int(state.get("attempt", 0) or 0)
    query = (state.get("query") or "")[:120]
    approval = state.get("approval") or {}
    approved = approval.get("approved") if isinstance(approval, dict) else None
    proposed_action = state.get("proposed_action") or "unspecified action"

    if route == Route.ERROR.value and attempt < 2:
        result = f"ERROR | tool=mock_support_lookup | attempt={attempt} | transient backend timeout"
        return {
            "tool_results": [result],
            "errors": [result],
            "events": [
                make_event(
                    "tool",
                    "failed",
                    "mock tool returned a transient failure",
                    latency_ms=_elapsed_ms(start),
                    attempt=attempt,
                    route=route,
                )
            ],
        }

    if route == Route.RISKY.value and approved is not True:
        result = "ERROR | tool=mock_action_executor | blocked: no approval decision on record"
        return {
            "tool_results": [result],
            "errors": [result],
            "events": [
                make_event(
                    "tool",
                    "blocked",
                    "risky action refused because approval was missing or rejected",
                    latency_ms=_elapsed_ms(start),
                    attempt=attempt,
                    route=route,
                )
            ],
        }

    if route == Route.RISKY.value:
        reviewer = approval.get("reviewer", "unknown") if isinstance(approval, dict) else "unknown"
        result = (
            f"OK | tool=mock_action_executor | attempt={attempt} | executed={proposed_action} "
            f"| approved_by={reviewer} | request={query}"
        )
        tool_name = "mock_action_executor"
    else:
        result = (
            f"OK | tool=mock_support_lookup | attempt={attempt} | request={query} "
            f"| payload=1 matching record (status=open, owner=support-tier-1, sla=24h)"
        )
        tool_name = "mock_support_lookup"

    return {
        "tool_results": [result],
        "messages": [f"tool:{tool_name}"],
        "events": [
            make_event(
                "tool",
                "completed",
                f"{tool_name} returned a result",
                latency_ms=_elapsed_ms(start),
                attempt=attempt,
                route=route,
                tool=tool_name,
            )
        ],
    }


# ─── evaluate ────────────────────────────────────────────────────────


class JudgeVerdict(BaseModel):
    """Structured-output contract for the LLM-as-judge evaluator."""

    verdict: Literal["success", "needs_retry"] = Field(
        description="success if the tool result answers the request, else needs_retry."
    )
    reason: str = Field(default="", description="One short sentence justifying the verdict.")


_JUDGE_PROMPT = """You judge whether a tool result is usable for answering a support ticket.

Ticket: {query}

Latest tool result: {result}

Answer needs_retry ONLY if the result is an error, empty, or clearly unusable.
If it carries usable information, answer success.
"""


def evaluate_node(state: AgentState) -> dict[str, Any]:
    """Evaluate tool results — the retry-loop gate.

    Deterministic fast path for explicit failures (an ``ERROR`` marker never needs a
    model to interpret), LLM-as-judge for everything else, heuristic fallback if the
    judge call fails.
    """
    start = time.perf_counter()
    results = state.get("tool_results") or []
    latest = results[-1] if results else ""
    query = (state.get("query") or "")[:_MAX_CONTEXT_CHARS]

    if not latest:
        return {
            "evaluation_result": "needs_retry",
            "errors": ["evaluate: no tool result to evaluate"],
            "events": [
                make_event(
                    "evaluate",
                    "completed",
                    "no tool evidence available, requesting retry",
                    latency_ms=_elapsed_ms(start),
                    verdict="needs_retry",
                    judge="heuristic",
                )
            ],
        }

    if "ERROR" in latest.upper():
        return {
            "evaluation_result": "needs_retry",
            "events": [
                make_event(
                    "evaluate",
                    "completed",
                    "tool result contains an explicit error marker",
                    latency_ms=_elapsed_ms(start),
                    verdict="needs_retry",
                    judge="heuristic",
                )
            ],
        }

    try:
        judge = get_llm().with_structured_output(JudgeVerdict)
        raw = judge.invoke(_JUDGE_PROMPT.format(query=query, result=latest[:_MAX_CONTEXT_CHARS]))
        verdict = raw if isinstance(raw, JudgeVerdict) else JudgeVerdict.model_validate(raw)
    except Exception as exc:
        return {
            "evaluation_result": "success",
            "errors": [f"evaluate: LLM judge failed ({type(exc).__name__}): {exc}"],
            "events": [
                make_event(
                    "evaluate",
                    "failed",
                    "LLM judge unavailable, heuristic accepted the non-error result",
                    latency_ms=_elapsed_ms(start),
                    verdict="success",
                    judge="heuristic",
                    fallback=True,
                )
            ],
        }

    return {
        "evaluation_result": verdict.verdict,
        "events": [
            make_event(
                "evaluate",
                "completed",
                f"LLM judge verdict: {verdict.verdict}",
                latency_ms=_elapsed_ms(start),
                verdict=verdict.verdict,
                judge="llm",
                reason=verdict.reason[:200],
            )
        ],
    }


# ─── answer ──────────────────────────────────────────────────────────


_ANSWER_PROMPT = """You are a support agent writing the final reply to a customer.

Ticket: {query}

Tool evidence (may be empty):
{tool_context}

Action / approval context (may be empty):
{approval_context}

Rules:
- Ground every factual claim in the evidence above. Do not invent order data.
- If an action was rejected or never approved, do NOT claim it was performed.
- If the evidence is empty, answer from general support knowledge and say plainly
  that no lookup was performed.
- Be concrete and under 120 words. Plain text, no markdown headers.
"""


def _tool_context(state: AgentState) -> str:
    results = state.get("tool_results") or []
    usable = [item for item in results if "ERROR" not in item.upper()]
    chosen = usable[-3:] if usable else results[-2:]
    if not chosen:
        return "(no tool was called)"
    return "\n".join(f"- {item[:_MAX_CONTEXT_CHARS]}" for item in chosen)


def _approval_context(state: AgentState) -> str:
    proposed_action = state.get("proposed_action")
    approval = state.get("approval")
    if not proposed_action and not approval:
        return "(no action required approval)"
    lines = []
    if proposed_action:
        lines.append(f"- proposed action: {str(proposed_action)[:_MAX_CONTEXT_CHARS]}")
    if isinstance(approval, dict):
        status = "APPROVED" if approval.get("approved") is True else "REJECTED"
        lines.append(
            f"- review decision: {status} by {approval.get('reviewer', 'unknown')} "
            f"({approval.get('comment', '')})"
        )
    return "\n".join(lines)


def answer_node(state: AgentState) -> dict[str, Any]:
    """Generate the final response with an LLM, grounded in state context."""
    start = time.perf_counter()
    query = (state.get("query") or "")[:_MAX_CONTEXT_CHARS]
    prompt = _ANSWER_PROMPT.format(
        query=query,
        tool_context=_tool_context(state),
        approval_context=_approval_context(state),
    )

    try:
        answer = _text_of(get_llm().invoke(prompt))
    except Exception as exc:
        message = (
            "We could not generate a reply automatically because the language model was "
            "unavailable. This ticket has been kept open and escalated to a human agent."
        )
        return {
            "final_answer": message,
            "errors": [f"answer: LLM generation failed ({type(exc).__name__}): {exc}"],
            "events": [
                make_event(
                    "answer",
                    "failed",
                    "grounded generation failed, escalation message returned",
                    latency_ms=_elapsed_ms(start),
                    grounded=False,
                )
            ],
        }

    return {
        "final_answer": answer,
        "messages": [f"answer:{answer[:60]}"],
        "events": [
            make_event(
                "answer",
                "completed",
                "grounded answer generated",
                latency_ms=_elapsed_ms(start),
                grounded=True,
                tool_results_used=len(state.get("tool_results") or []),
                answer_chars=len(answer),
            )
        ],
    }


# ─── clarify ─────────────────────────────────────────────────────────


_CLARIFY_PROMPT = """A support ticket cannot be actioned yet. Ask the customer ONE
specific follow-up question that unblocks it.

Ticket: {query}

Additional context (may be empty): {context}

Name the exact missing detail (id, product, error message, desired outcome).
Return only the question, one sentence, no preamble.
"""


def ask_clarification_node(state: AgentState) -> dict[str, Any]:
    """Ask for missing information instead of hallucinating.

    Reached from two places: the ``missing_info`` route, and a rejected approval —
    in which case the rejection comment becomes part of the context.
    """
    start = time.perf_counter()
    query = (state.get("query") or "")[:_MAX_CONTEXT_CHARS]
    approval = state.get("approval")
    approval_map: dict[str, Any] = approval if isinstance(approval, dict) else {}
    rejected = bool(approval_map) and approval_map.get("approved") is not True
    reason = "rejected_approval" if rejected else "missing_info"

    if rejected:
        proposed = str(state.get("proposed_action") or "unspecified")[:200]
        comment = approval_map.get("comment", "")
        context = (
            f"A reviewer rejected the proposed action '{proposed}' with the comment: "
            f"'{comment}'. Ask what alternative the customer wants."
        )
    else:
        context = "(none)"

    try:
        question = _text_of(get_llm().invoke(_CLARIFY_PROMPT.format(query=query, context=context)))
        degraded = False
        error_update: list[str] = []
    except Exception as exc:
        question = (
            "Could you share more detail so we can help — which account, order or product "
            "is affected, and what outcome are you expecting?"
        )
        degraded = True
        error_update = [f"clarify: LLM question generation failed ({type(exc).__name__}): {exc}"]

    update: dict[str, Any] = {
        "pending_question": question,
        "final_answer": question,
        "messages": [f"clarify:{question[:60]}"],
        "events": [
            make_event(
                "clarify",
                "failed" if degraded else "completed",
                "clarification requested",
                latency_ms=_elapsed_ms(start),
                reason=reason,
                fallback=degraded,
            )
        ],
    }
    if error_update:
        update["errors"] = error_update
    return update


# ─── risky action ────────────────────────────────────────────────────


_RISKY_PROMPT = """A support ticket requests an action with a real side effect.

Ticket: {query}

Write ONE sentence describing the concrete action to execute and the irreversible
effect a reviewer must weigh. Do not execute anything. Return only that sentence.
"""


def risky_action_node(state: AgentState) -> dict[str, Any]:
    """Prepare a risky action for human approval. Executes nothing."""
    start = time.perf_counter()
    query = (state.get("query") or "")[:_MAX_CONTEXT_CHARS]
    risk_level = state.get("risk_level", "high")

    try:
        proposed = _text_of(get_llm().invoke(_RISKY_PROMPT.format(query=query)))
        degraded = False
        error_update: list[str] = []
    except Exception as exc:
        proposed = f"Execute the customer-requested side effect described in: {query}"
        degraded = True
        error_update = [f"risky_action: LLM proposal failed ({type(exc).__name__}): {exc}"]

    update: dict[str, Any] = {
        "proposed_action": proposed,
        "messages": [f"risky_action:{proposed[:60]}"],
        "events": [
            make_event(
                "risky_action",
                "failed" if degraded else "proposed",
                "action prepared for human review, not executed",
                latency_ms=_elapsed_ms(start),
                risk_level=risk_level,
                requires_approval=True,
                fallback=degraded,
            )
        ],
    }
    if error_update:
        update["errors"] = error_update
    return update


# ─── approval ────────────────────────────────────────────────────────


def approval_node(state: AgentState) -> dict[str, Any]:
    """Human-in-the-loop approval step — the gate in front of every side effect.

    Default: a mock decision so tests and CI never block on input. Extension: set
    ``LANGGRAPH_INTERRUPT=true`` to suspend the graph with ``interrupt()`` and resume
    with a real reviewer payload.
    """
    start = time.perf_counter()
    proposed_action = state.get("proposed_action")

    if not proposed_action:
        blocked = ApprovalDecision(
            approved=False, reviewer="policy-guard", comment="nothing concrete to review"
        )
        return {
            "approval": blocked.model_dump(),
            "errors": ["approval: no proposed_action to review, failing closed"],
            "events": [
                make_event(
                    "approval",
                    "rejected",
                    "approval refused: no proposed action on record",
                    latency_ms=_elapsed_ms(start),
                    approved=False,
                    mode="policy-guard",
                )
            ],
        }

    mode = "mock"
    if os.getenv("LANGGRAPH_INTERRUPT", "").strip().lower() in _TRUE_VALUES:
        from langgraph.types import interrupt

        payload = interrupt(
            {"proposed_action": proposed_action, "question": "Approve this action?"}
        )
        mode = "interrupt"
        if isinstance(payload, dict):
            decision = ApprovalDecision(
                approved=bool(payload.get("approved", False)),
                reviewer=str(payload.get("reviewer", "human-reviewer")),
                comment=str(payload.get("comment", "")),
            )
        else:
            decision = ApprovalDecision(
                approved=bool(payload), reviewer="human-reviewer", comment="resume payload"
            )
    else:
        decision = ApprovalDecision(
            approved=True,
            reviewer="mock-reviewer",
            comment="auto-approved by mock reviewer (set LANGGRAPH_INTERRUPT=true for real HITL)",
        )

    return {
        "approval": decision.model_dump(),
        "messages": [f"approval:{'approved' if decision.approved else 'rejected'}"],
        "events": [
            make_event(
                "approval",
                "approved" if decision.approved else "rejected",
                f"approval observed from {decision.reviewer}",
                latency_ms=_elapsed_ms(start),
                approved=decision.approved,
                reviewer=decision.reviewer,
                mode=mode,
            )
        ],
    }


# ─── retry / dead letter / finalize ──────────────────────────────────


def retry_or_fallback_node(state: AgentState) -> dict[str, Any]:
    """Record a retry attempt. This node is the ONLY owner of the attempt counter."""
    start = time.perf_counter()
    attempt = int(state.get("attempt", 0) or 0) + 1
    max_attempts = int(state.get("max_attempts", 3) or 0)
    results = state.get("tool_results") or []
    latest_failure = results[-1] if results else "no tool result yet (error route entry)"

    return {
        "attempt": attempt,
        "errors": [f"retry: attempt {attempt}/{max_attempts} after: {latest_failure[:200]}"],
        "messages": [f"retry:{attempt}/{max_attempts}"],
        "events": [
            make_event(
                "retry",
                "recorded",
                f"retry attempt {attempt} of {max_attempts}",
                latency_ms=_elapsed_ms(start),
                attempt=attempt,
                max_attempts=max_attempts,
            )
        ],
    }


def dead_letter_node(state: AgentState) -> dict[str, Any]:
    """Handle unresolvable failures after max retries: retry → fallback → dead letter."""
    start = time.perf_counter()
    attempt = int(state.get("attempt", 0) or 0)
    max_attempts = int(state.get("max_attempts", 3) or 0)
    errors = state.get("errors") or []
    last_error = errors[-1] if errors else "no error detail recorded"

    message = (
        f"We could not complete this request automatically after {attempt} of "
        f"{max_attempts} allowed attempts. The ticket has been escalated to a human "
        f"support engineer with the failure history attached. Last failure: {last_error[:200]}"
    )
    return {
        "final_answer": message,
        "messages": ["dead_letter:escalated"],
        "events": [
            make_event(
                "dead_letter",
                "exhausted",
                "retry budget exhausted, escalated to human support",
                latency_ms=_elapsed_ms(start),
                attempt=attempt,
                max_attempts=max_attempts,
                error_count=len(errors),
            )
        ],
    }


def finalize_node(state: AgentState) -> dict[str, Any]:
    """Emit a final audit event. All routes must pass through here before END.

    The classified ``route`` is deliberately left untouched: metrics compare it with the
    expected route, so overwriting it with ``done``/``dead_letter`` would corrupt them.
    """
    start = time.perf_counter()
    return {
        "events": [
            make_event(
                "finalize",
                "completed",
                "workflow finished",
                latency_ms=_elapsed_ms(start),
                route=state.get("route"),
                has_answer=bool(state.get("final_answer")),
                has_pending_question=bool(state.get("pending_question")),
                attempts=int(state.get("attempt", 0) or 0),
            )
        ],
    }
