#!/usr/bin/env python3
"""
GARVIS App Ledger Viewer CLI.

Read-only cockpit instrument for inspecting Stage 1 decision ledger metadata.

This tool is a window, not a hand.

Default behavior:
- opens SQLite using mode=ro
- prints safe metadata only
- hides raw JSON payloads
- writes no files
- creates no exports
- performs no network calls
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_LIMIT = 10
MAX_LIMIT = 100

REQUIRED_COLUMNS = {
    "id",
    "timestamp",
    "input_hash",
    "candidates_json",
    "gate_scores_json",
    "decision",
    "approved_by_human",
}


class ViewerError(RuntimeError):
    """Raised for safe user-facing viewer failures."""


@dataclass(frozen=True)
class LedgerRow:
    row_id: int
    timestamp: str
    decision: str
    approved_by_human: int
    input_hash: str
    candidates_summary: str
    gates_summary: str


def build_readonly_uri(db_path: Path) -> str:
    resolved = db_path.expanduser().resolve()
    return f"file:{resolved.as_posix()}?mode=ro"


def connect_read_only(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise ViewerError(f"ledger database not found: {db_path}")
    if not db_path.is_file():
        raise ViewerError(f"ledger path is not a file: {db_path}")

    uri = build_readonly_uri(db_path)
    return sqlite3.connect(uri, uri=True)


def validate_schema(conn: sqlite3.Connection) -> None:
    table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='decision_ledger'"
    ).fetchone()

    if table is None:
        raise ViewerError("decision_ledger table not found")

    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(decision_ledger)").fetchall()
    }

    missing = sorted(REQUIRED_COLUMNS - columns)
    if missing:
        raise ViewerError(f"decision_ledger missing required columns: {', '.join(missing)}")


def clamp_limit(limit: int) -> int:
    if limit < 1:
        raise ViewerError("--limit must be at least 1")
    return min(limit, MAX_LIMIT)


def summarize_json(raw: str | None, label: str) -> str:
    if raw is None or raw == "":
        return f"{label}:empty"

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return f"{label}:unreadable"

    if isinstance(parsed, list):
        return f"{label}:list[{len(parsed)}]"

    if isinstance(parsed, dict):
        return f"{label}:dict[{len(parsed)} keys]"

    return f"{label}:{type(parsed).__name__}"


def short_hash(value: str | None) -> str:
    if not value:
        return "-"
    value = str(value)
    if len(value) <= 12:
        return value
    return value[:12] + "…"


def fetch_rows(conn: sqlite3.Connection, limit: int) -> list[LedgerRow]:
    rows = conn.execute(
        """
        SELECT
            id,
            timestamp,
            decision,
            approved_by_human,
            input_hash,
            candidates_json,
            gate_scores_json
        FROM decision_ledger
        ORDER BY id DESC
        LIMIT ?
        """,
        (clamp_limit(limit),),
    ).fetchall()

    result: list[LedgerRow] = []
    for row in rows:
        result.append(
            LedgerRow(
                row_id=int(row[0]),
                timestamp=str(row[1]),
                decision=str(row[2]),
                approved_by_human=int(row[3]),
                input_hash=short_hash(row[4]),
                candidates_summary=summarize_json(row[5], "candidates"),
                gates_summary=summarize_json(row[6], "gates"),
            )
        )

    return result


def trim(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return value[: max(0, width - 1)] + "…"


def render_table(rows: Sequence[LedgerRow], db_path: Path) -> str:
    lines: list[str] = []
    lines.append("GARVIS App Ledger Viewer")
    lines.append("mode: read-only")
    lines.append(f"db: {db_path}")
    lines.append("raw_payloads: hidden")
    lines.append("")

    if not rows:
        lines.append("No ledger rows found.")
        return "\n".join(lines)

    header = (
        f"{'ID':>5}  "
        f"{'Timestamp':<24}  "
        f"{'Approved':<8}  "
        f"{'Decision':<28}  "
        f"{'Input Hash':<14}  "
        f"{'Payload Summary'}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for row in rows:
        payload_summary = f"{row.candidates_summary}; {row.gates_summary}"
        lines.append(
            f"{row.row_id:>5}  "
            f"{trim(row.timestamp, 24):<24}  "
            f"{row.approved_by_human:<8}  "
            f"{trim(row.decision, 28):<28}  "
            f"{row.input_hash:<14}  "
            f"{payload_summary}"
        )

    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only GARVIS Stage 1 decision ledger viewer."
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Path to the Stage 1 SQLite decision ledger.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of most recent rows to show. Default: {DEFAULT_LIMIT}. Max: {MAX_LIMIT}.",
    )
    return parser.parse_args(argv)


def run(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    db_path = Path(args.db)

    conn: sqlite3.Connection | None = None
    try:
        conn = connect_read_only(db_path)
        validate_schema(conn)
        rows = fetch_rows(conn, args.limit)
        print(render_table(rows, db_path))
        return 0
    except ViewerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except sqlite3.Error as exc:
        print(f"ERROR: ledger read failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if conn is not None:
            conn.close()


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
