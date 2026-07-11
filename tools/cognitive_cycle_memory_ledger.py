#!/usr/bin/env python3
"""
GARVIS Cognitive Cycle Memory Ledger CLI.

DIRECTIVE-008F.

Stage 2 local memory-vessel initialization instrument.

This tool can initialize a local SQLite database for future cognitive-cycle
memory records.

It does not:
- append cognitive cycles
- read raw cognitive cycle JSON
- execute actions
- run candidates
- call a network
- call an LLM
- contact the outside world
- commit
- push
- upgrade claims
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


DEFAULT_DB = "data/cognitive_cycles/cognitive_cycle_memory.sqlite3"
SCHEMA_VERSION = "1.0"


class MemoryLedgerError(RuntimeError):
    """Safe user-facing memory ledger failure."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def initialize_database(db_path: Path) -> None:
    with connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ledger_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at_utc TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cognitive_cycle_records (
                record_id TEXT PRIMARY KEY,
                record_version TEXT NOT NULL,
                cycle_id TEXT NOT NULL,
                cycle_version TEXT NOT NULL,
                cycle_timestamp_utc TEXT NOT NULL,
                cycle_hash TEXT NOT NULL,
                cycle_schema_version TEXT NOT NULL,
                json_path TEXT,
                markdown_path TEXT,
                json_committed INTEGER NOT NULL DEFAULT 0,
                markdown_committed INTEGER NOT NULL DEFAULT 0,
                artifact_policy TEXT NOT NULL,
                previous_cycle_id TEXT,
                next_cycle_id TEXT,
                parent_record_id TEXT,
                supersedes_record_id TEXT,
                selected_candidate_id TEXT,
                decision TEXT NOT NULL,
                confidence TEXT NOT NULL,
                next_smallest_step TEXT NOT NULL,
                case_against_selected TEXT NOT NULL,
                risk_of_doing_selected TEXT NOT NULL,
                risk_of_not_doing_selected TEXT NOT NULL,
                power_requested INTEGER NOT NULL,
                requested_stage TEXT NOT NULL,
                requested_permissions_json TEXT NOT NULL,
                why_power_is_needed TEXT NOT NULL,
                why_power_should_be_refused TEXT NOT NULL,
                approval_required INTEGER NOT NULL DEFAULT 1,
                ledger_required INTEGER NOT NULL DEFAULT 1,
                permission_granted_by_this_record INTEGER NOT NULL DEFAULT 0,
                review_status TEXT NOT NULL DEFAULT 'unreviewed',
                operator TEXT NOT NULL DEFAULT 'Adrien D Thomas',
                operator_decision TEXT NOT NULL DEFAULT 'none',
                review_timestamp_utc TEXT,
                notes TEXT NOT NULL DEFAULT '',
                approval_ledger_id INTEGER,
                record_is_memory_not_action INTEGER NOT NULL DEFAULT 1,
                record_is_not_approval INTEGER NOT NULL DEFAULT 1,
                record_is_not_empirical_result INTEGER NOT NULL DEFAULT 1,
                record_is_not_claim_upgrade INTEGER NOT NULL DEFAULT 1,
                created_at_utc TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS open_problem_placeholders (
                problem_id TEXT PRIMARY KEY,
                problem_title TEXT NOT NULL,
                problem_status TEXT NOT NULL DEFAULT 'placeholder_only',
                claim_status TEXT NOT NULL DEFAULT 'forbidden_to_claim',
                allowed_activity TEXT NOT NULL DEFAULT 'define, question, design tests, preserve uncertainty',
                forbidden_activity TEXT NOT NULL DEFAULT 'claim solved, claim proof, claim consciousness, claim AGI',
                created_at_utc TEXT NOT NULL
            )
            """
        )

        now = utc_now()
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "stage": "Stage 2 memory vessel initialization",
            "append_behavior": "not_implemented",
            "open_problem_box": "placeholder_table_only",
            "network_allowed": "false",
            "llm_calls_allowed": "false",
            "external_contact_allowed": "false",
            "claim_upgrade_allowed": "false",
        }

        for key, value in metadata.items():
            conn.execute(
                """
                INSERT OR IGNORE INTO ledger_metadata (key, value, created_at_utc)
                VALUES (?, ?, ?)
                """,
                (key, value, now),
            )


def verify_database(db_path: Path) -> list[str]:
    if not db_path.exists():
        raise MemoryLedgerError(f"database does not exist: {db_path}")

    required_tables = {
        "ledger_metadata",
        "cognitive_cycle_records",
        "open_problem_placeholders",
    }

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        tables = {row[0] for row in rows}

        missing = sorted(required_tables - tables)
        if missing:
            raise MemoryLedgerError(f"missing required tables: {', '.join(missing)}")

        metadata_rows = conn.execute(
            "SELECT key, value FROM ledger_metadata ORDER BY key"
        ).fetchall()

    return [f"{key}={value}" for key, value in metadata_rows]


def render_status(db_path: Path, metadata: list[str]) -> str:
    lines = []
    lines.append("# GARVIS Cognitive Cycle Memory Ledger")
    lines.append("")
    lines.append("mode: local initialization / verification")
    lines.append("stage: Stage 2 memory vessel")
    lines.append("append_behavior: not implemented")
    lines.append("execution_behavior: no candidate execution")
    lines.append("network_calls: none")
    lines.append("llm_calls: none")
    lines.append("outside_contact: none")
    lines.append("claim_upgrades: none")
    lines.append(f"database: {db_path}")
    lines.append("")
    lines.append("## Tables")
    lines.append("")
    lines.append("- ledger_metadata")
    lines.append("- cognitive_cycle_records")
    lines.append("- open_problem_placeholders")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    for item in metadata:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Boundary")
    lines.append("")
    lines.append("- This tool initializes the memory vessel only.")
    lines.append("- It does not append cognitive cycles.")
    lines.append("- It does not store claims.")
    lines.append("- It does not solve open problems.")
    lines.append("- It reserves a future placeholder for no-claim/open-problem work.")
    lines.append("- Adrien decides what becomes memory.")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize or verify the GARVIS cognitive-cycle memory ledger."
    )
    parser.add_argument("--db", default=DEFAULT_DB, help=f"SQLite path. Default: {DEFAULT_DB}")
    parser.add_argument("--init-db", action="store_true", help="Initialize the local SQLite memory vessel.")
    parser.add_argument("--verify", action="store_true", help="Verify the memory vessel schema.")
    return parser.parse_args(argv)


def run(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    db_path = Path(args.db).expanduser()

    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path

    try:
        if not args.init_db and not args.verify:
            raise MemoryLedgerError("choose --init-db and/or --verify")

        if args.init_db:
            initialize_database(db_path)

        metadata = verify_database(db_path)
        print(render_status(db_path, metadata))
        return 0
    except (sqlite3.Error, OSError, MemoryLedgerError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
