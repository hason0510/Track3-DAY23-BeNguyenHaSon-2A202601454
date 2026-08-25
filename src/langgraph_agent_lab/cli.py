"""CLI for the lab."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from .graph import build_graph
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer
from .report import write_report
from .scenarios import load_scenarios
from .state import Route, Scenario, initial_state

app = typer.Typer(no_args_is_help=True)


def _state_history_ok(graph: Any, run_config: dict[str, Any]) -> bool:
    """Return True when the checkpointer can replay this thread's state history."""
    try:
        snapshot = graph.get_state(run_config)
        history = list(graph.get_state_history(run_config))
    except Exception:
        return False
    return bool(history) and snapshot is not None and bool(snapshot.values)


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run all grading scenarios and write metrics JSON."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer = build_checkpointer(cfg.get("checkpointer", "memory"), cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)
    metrics = []
    replayable = 0
    for scenario in scenarios:
        state = initial_state(scenario)
        run_config = {"configurable": {"thread_id": state["thread_id"]}}
        final_state = graph.invoke(state, config=run_config)
        if _state_history_ok(graph, run_config):
            replayable += 1
        metrics.append(
            metric_from_state(
                final_state,
                scenario.expected_route.value,
                scenario.requires_approval,
            )
        )
    resume_success = bool(scenarios) and replayable == len(scenarios)
    report = summarize_metrics(metrics, resume_success=resume_success)
    write_metrics(report, output)
    if cfg.get("report_path"):
        write_report(report, cfg["report_path"])
    typer.echo(
        f"Wrote metrics to {output} "
        f"(state history replayable for {replayable}/{len(scenarios)} threads)"
    )


@app.command("validate-metrics")
def validate_metrics(metrics: Annotated[Path, typer.Option("--metrics")]) -> None:
    """Validate metrics JSON schema for grading."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    typer.echo(f"Metrics valid. success_rate={report.success_rate:.2%}")


@app.command("recovery-demo")
def recovery_demo(
    query: Annotated[str, typer.Option("--query")] = "Please lookup order status for order 12345",
    expected_route: Annotated[str, typer.Option("--expected-route")] = "tool",
    thread_id: Annotated[str, typer.Option("--thread-id")] = "recovery-demo",
    db: Annotated[Path, typer.Option("--db")] = Path("checkpoints.db"),
    inspect_only: Annotated[
        bool,
        typer.Option("--inspect-only", help="Skip execution; only replay an existing thread."),
    ] = False,
    output: Annotated[Path, typer.Option("--output")] = Path("outputs/recovery_evidence.json"),
) -> None:
    """Persistence evidence: run a thread on SQLite, then replay it from the database.

    Run once to execute and persist, then run again with ``--inspect-only`` from a NEW
    process: the second process re-opens the same database file and replays the state
    history, which is the crash-resume evidence for the report.
    """
    checkpointer = build_checkpointer("sqlite", str(db))
    graph = build_graph(checkpointer=checkpointer)
    run_config = {"configurable": {"thread_id": thread_id}}

    if not inspect_only:
        scenario = Scenario(id=thread_id, query=query, expected_route=Route(expected_route))
        state = initial_state(scenario)
        state["thread_id"] = thread_id
        graph.invoke(state, config=run_config)

    snapshot = graph.get_state(run_config)
    history = list(graph.get_state_history(run_config))
    values = snapshot.values if snapshot else {}
    evidence = {
        "mode": "inspect-only (fresh process)" if inspect_only else "execute+persist",
        "pid": os.getpid(),
        "database": str(db),
        "thread_id": thread_id,
        "checkpoint_id": (snapshot.config or {}).get("configurable", {}).get("checkpoint_id")
        if snapshot
        else None,
        "checkpoints_in_history": len(history),
        "next_nodes": list(snapshot.next) if snapshot else [],
        "route": values.get("route"),
        "attempt": values.get("attempt"),
        "events_recorded": len(values.get("events", []) or []),
        "nodes_replayed": [
            event.get("node") for event in (values.get("events", []) or [])
        ],
        "final_answer_preview": (values.get("final_answer") or "")[:160],
        "resume_success": bool(history) and bool(values),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(json.dumps(evidence, indent=2, ensure_ascii=False))


@app.command("show-graph")
def show_graph(
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Print (and optionally write) the Mermaid diagram of the compiled graph."""
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    diagram = graph.get_graph().draw_mermaid()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(diagram, encoding="utf-8")
        typer.echo(f"Wrote diagram to {output}")
    typer.echo(diagram)


if __name__ == "__main__":
    app()
