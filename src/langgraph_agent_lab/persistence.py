"""Checkpointer adapter.

The caller owns the checkpointer lifecycle and passes it into ``build_graph()``; the
graph builder never creates one itself. ``memory`` is the default used by tests and CI,
``sqlite`` is the durable backend used for the persistence/recovery evidence, and
``postgres`` is the optional multi-process backend.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_SQLITE_PATH = "checkpoints.db"


def _sqlite_path(database_url: str | None) -> str:
    """Accept a bare path or a ``sqlite:///path`` URL and return a filesystem path."""
    if not database_url:
        return DEFAULT_SQLITE_PATH
    for prefix in ("sqlite:///", "sqlite://", "sqlite:"):
        if database_url.startswith(prefix):
            return database_url[len(prefix) :] or DEFAULT_SQLITE_PATH
    return database_url


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer for the requested backend."""
    if kind == "none":
        return None

    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                'SQLite checkpointer requires: pip install -e ".[sqlite]"'
            ) from exc

        path = _sqlite_path(database_url)
        parent = Path(path).expanduser().parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: LangGraph may touch the connection from worker threads.
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        saver = SqliteSaver(conn=conn)
        saver.setup()
        return saver

    if kind == "postgres":
        if not database_url:
            raise ValueError("Postgres checkpointer requires database_url (DATABASE_URL)")
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                'Postgres checkpointer requires: pip install -e ".[postgres]"'
            ) from exc

        saver = PostgresSaver.from_conn_string(database_url).__enter__()
        saver.setup()
        return saver

    raise ValueError(f"Unknown checkpointer kind: {kind}")
