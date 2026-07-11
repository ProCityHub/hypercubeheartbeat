#!/usr/bin/env python3
"""
GARVIS Scientific Cockpit Snapshot CLI.

Stage 0 view-only cockpit instrument.

This tool summarizes local repository and ledger state for Adrien.

It does not:
- call a network
- call an LLM
- run tests
- run guard checks
- modify files
- commit files
- push branches
- open pull requests
- execute actions
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Sequence


RISKY_UNTRACKED_PREFIXES = (
    "AGI/",
    "brain.py",
    "data/stage1_senses/",
    "tmp/self_design_proposals/",
)

DEFAULT_LEDGER_DB = "data/stage1_senses/decision_ledger.sqlite3"
DEFAULT_LIMIT = 8


class SnapshotError(RuntimeError):
    """Safe user-facing failure."""


def run_git(repo: Path, args: Sequence[str], allow_failure: bool = False) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 and not allow_failure:
        raise SnapshotError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def ensure_git_repo(repo: Path) -> None:
    inside = run_git(repo, ["rev-parse", "--is-inside-work-tree"])
    if inside != "true":
        raise SnapshotError(f"not a git repository: {repo}")


def git_branch(repo: Path) -> str:
    return run_git(repo, ["rev-parse", "--abbrev-ref", "HEAD"], allow_failure=True) or "unknown"


def git_commit(repo: Path) -> str:
    return run_git(repo, ["rev-parse", "--short", "HEAD"], allow_failure=True) or "unknown"


def git_status_lines(repo: Path) -> list[str]:
    raw = run_git(repo, ["status", "--short"], allow_failure=True)
    return [line for line in raw.splitlines() if line.strip()]


def parse_status(lines: Sequence[str]) -> dict[str, object]:
    staged = 0
    modified = 0
    untracked: list[str] = []
    risky: list[str] = []

    for line in lines:
        if line.startswith("?? "):
            path = line[3:]
            untracked.append(path)
            if any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in RISKY_UNTRACKED_PREFIXES):
                risky.append(path)
            continue

        if len(line) >= 2:
            if line[0] != " ":
                staged += 1
            if line[1] != " ":
                modified += 1

    return {
        "staged": staged,
        "modified": modified,
        "untracked": untracked,
        "risky": risky,
        "clean": staged == 0 and modified == 0 and not untracked,
    }


def list_matching(repo: Path, pattern: str, limit: int) -> list[str]:
    paths = []
    for path in sorted(repo.glob(pattern)):
        if path.is_file():
            paths.append(path.relative_to(repo).as_posix())
    return paths[:limit]


def list_recursive(repo: Path, base: str, pattern: str, limit: int) -> list[str]:
    root = repo / base
    if not root.exists():
        return []
    paths = []
    for path in sorted(root.rglob(pattern)):
        if path.is_file():
            paths.append(path.relative_to(repo).as_posix())
    return paths[:limit]


def readonly_sqlite_uri(path: Path) -> str:
    return f"file:{path.expanduser().resolve().as_posix()}?mode=ro"


def ledger_summary(db_path: Path) -> list[str]:
    if not db_path.exists():
        return [f"ledger: not found at {db_path}", "ledger_mode: unavailable"]

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(readonly_sqlite_uri(db_path), uri=True)
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='decision_ledger'"
        ).fetchone()
        if table is None:
            return ["ledger: decision_ledger table not found", "ledger_mode: read-only"]

        total = conn.execute("SELECT COUNT(*) FROM decision_ledger").fetchone()[0]
        approved = conn.execute(
            "SELECT COUNT(*) FROM decision_ledger WHERE approved_by_human = 1"
        ).fetchone()[0]
        latest = conn.execute(
            """
            SELECT id, timestamp, decision, approved_by_human, input_hash
            FROM decision_ledger
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

        lines = [
            "ledger: available",
            "ledger_mode: read-only",
            f"ledger_rows: {total}",
            f"ledger_approved_rows: {approved}",
        ]

        if latest:
            input_hash = str(latest[4] or "")
            short_hash = input_hash[:12] + "…" if len(input_hash) > 12 else input_hash
            lines.extend(
                [
                    f"latest_row_id: {latest[0]}",
                    f"latest_timestamp: {latest[1]}",
                    f"latest_decision: {latest[2]}",
                    f"latest_approved_by_human: {latest[3]}",
                    f"latest_input_hash: {short_hash or '-'}",
                ]
            )

        return lines
    except sqlite3.Error as exc:
        return [f"ledger: read failed safely: {exc}", "ledger_mode: read-only attempted"]
    finally:
        if conn is not None:
            conn.close()


def bullet_list(items: Sequence[str], empty: str) -> list[str]:
    if not items:
        return [f"- {empty}"]
    return [f"- {item}" for item in items]


def render_snapshot(repo: Path, ledger_db: Path, limit: int) -> str:
    ensure_git_repo(repo)

    status = parse_status(git_status_lines(repo))
    tools = list_recursive(repo, "tools", "*.py", limit)
    tests = list_recursive(repo, "tests", "test_*.py", limit)
    decisions = list_recursive(repo, "ai_infrastructure/decisions", "*.md", limit)
    app_interfaces = list_recursive(repo, "app_infrastructure/interfaces", "*.md", limit)

    lines: list[str] = []
    lines.append("# GARVIS Scientific Cockpit Snapshot")
    lines.append("")
    lines.append("mode: read-only")
    lines.append("stage: Stage 0 view-only")
    lines.append("actions_executed: none")
    lines.append("network_calls: none")
    lines.append("llm_calls: none")
    lines.append("")
    lines.append("## Git State")
    lines.append("")
    lines.append(f"- branch: {git_branch(repo)}")
    lines.append(f"- commit: {git_commit(repo)}")
    lines.append(f"- clean: {status['clean']}")
    lines.append(f"- staged_changes: {status['staged']}")
    lines.append(f"- modified_changes: {status['modified']}")
    lines.append(f"- untracked_paths: {len(status['untracked'])}")
    lines.append("")
    lines.append("## Do Not Commit Watch")
    lines.append("")
    lines.extend(bullet_list(status["risky"], "no risky untracked paths detected"))  # type: ignore[arg-type]
    lines.append("")
    lines.append("## Tools")
    lines.append("")
    lines.extend(bullet_list(tools, "no tools found"))
    lines.append("")
    lines.append("## Tests")
    lines.append("")
    lines.extend(bullet_list(tests, "no tests found"))
    lines.append("")
    lines.append("## Decision Records")
    lines.append("")
    lines.extend(bullet_list(decisions, "no decision records found"))
    lines.append("")
    lines.append("## App Interface Runbooks")
    lines.append("")
    lines.extend(bullet_list(app_interfaces, "no app interface runbooks found"))
    lines.append("")
    lines.append("## Local Ledger Memory")
    lines.append("")
    lines.extend(f"- {item}" for item in ledger_summary(ledger_db))
    lines.append("")
    lines.append("## Boundary")
    lines.append("")
    lines.append("- This snapshot is a cockpit instrument.")
    lines.append("- It does not approve anything.")
    lines.append("- It does not execute anything.")
    lines.append("- It does not turn proposals into action.")
    lines.append("- Adrien remains final authority.")
    lines.append("")

    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a read-only GARVIS scientific cockpit snapshot."
    )
    parser.add_argument(
        "--repo",
        default=".",
        help="Repository root. Default: current directory.",
    )
    parser.add_argument(
        "--ledger-db",
        default=DEFAULT_LEDGER_DB,
        help=f"Optional local Stage 1 ledger path. Default: {DEFAULT_LEDGER_DB}.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Maximum items per section. Default: {DEFAULT_LIMIT}.",
    )
    return parser.parse_args(argv)


def run(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    repo = Path(args.repo).expanduser().resolve()
    ledger_db = Path(args.ledger_db).expanduser()
    if not ledger_db.is_absolute():
        ledger_db = repo / ledger_db

    try:
        limit = max(1, min(int(args.limit), 50))
        print(render_snapshot(repo, ledger_db, limit))
        return 0
    except SnapshotError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
