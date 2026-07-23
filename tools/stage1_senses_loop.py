#!/usr/bin/env python3
"""
Stage 1 Senses Loop

Scope:
- Local Android/Termux sensing scaffold.
- SQLite append-only decision ledger.
- Camera/microphone/notification commands are optional and human-run.
- No autonomous outside action.
- No LLM call.
- No claim upgrade.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS decision_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    input_hash TEXT NOT NULL,
    candidates_json TEXT NOT NULL,
    gate_scores_json TEXT NOT NULL,
    decision TEXT NOT NULL,
    approved_by_human INTEGER NOT NULL CHECK (approved_by_human IN (0,1))
);

CREATE INDEX IF NOT EXISTS idx_decision_ledger_timestamp
ON decision_ledger(timestamp);

CREATE INDEX IF NOT EXISTS idx_decision_ledger_input_hash
ON decision_ledger(input_hash);

CREATE TRIGGER IF NOT EXISTS decision_ledger_no_update
BEFORE UPDATE ON decision_ledger
BEGIN
    SELECT RAISE(ABORT, 'append-only ledger: UPDATE forbidden');
END;

CREATE TRIGGER IF NOT EXISTS decision_ledger_no_delete
BEFORE DELETE ON decision_ledger
BEGIN
    SELECT RAISE(ABORT, 'append-only ledger: DELETE forbidden');
END;
"""


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.executescript(SCHEMA)


def append_decision(
    db_path: Path,
    input_text: str,
    candidates: list[dict[str, Any]],
    gate_scores: dict[str, Any],
    decision: str,
    approved_by_human: bool,
) -> int:
    init_db(db_path)
    with sqlite3.connect(db_path) as con:
        cur = con.execute(
            """
            INSERT INTO decision_ledger
            (timestamp, input_hash, candidates_json, gate_scores_json, decision, approved_by_human)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now(),
                sha256_text(input_text),
                json_dumps(candidates),
                json_dumps(gate_scores),
                decision,
                1 if approved_by_human else 0,
            ),
        )
        return int(cur.lastrowid)


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def run_command(cmd: list[str], timeout: int = 30) -> dict[str, Any]:
    started = utc_now()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "started_utc": started,
            "command": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {
            "started_utc": started,
            "command": cmd,
            "error": repr(exc),
        }


def sense_status() -> dict[str, Any]:
    return {
        "termux_camera_photo": command_exists("termux-camera-photo"),
        "termux_microphone_record": command_exists("termux-microphone-record"),
        "termux_notification": command_exists("termux-notification"),
        "termux_wake_lock": command_exists("termux-wake-lock"),
        "python": sys.executable,
        "cwd": os.getcwd(),
    }


def capture_camera(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"camera_{int(time.time())}.jpg"
    if not command_exists("termux-camera-photo"):
        return {"available": False, "reason": "termux-camera-photo not found"}
    return run_command(["termux-camera-photo", "-c", "0", str(target)], timeout=60)


def record_microphone(output_dir: Path, seconds: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"audio_{int(time.time())}.m4a"
    if not command_exists("termux-microphone-record"):
        return {"available": False, "reason": "termux-microphone-record not found"}
    return run_command(
        ["termux-microphone-record", "-f", str(target), "-l", str(seconds)],
        timeout=max(30, seconds + 20),
    )


def notify_alive() -> dict[str, Any]:
    if not command_exists("termux-notification"):
        return {"available": False, "reason": "termux-notification not found"}
    return run_command(
        [
            "termux-notification",
            "--id",
            "lattice-stage1",
            "--title",
            "Lattice Stage 1",
            "--content",
            "Senses loop alive",
        ],
        timeout=20,
    )


def self_test(db_path: Path) -> int:
    input_text = "stage1 self-test"
    candidates = [{"name": "ledger_init"}, {"name": "sensor_status"}]
    gates = {
        "coherence": "not_run_stage1",
        "grounding": "not_run_stage1",
        "survival": "not_run_stage1",
        "sensor_status": sense_status(),
    }
    row_id = append_decision(
        db_path=db_path,
        input_text=input_text,
        candidates=candidates,
        gate_scores=gates,
        decision="SELF_TEST_LEDGER_APPEND",
        approved_by_human=False,
    )
    print(f"SELF_TEST_OK row_id={row_id} db={db_path}")
    return 0


def run_once(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    outputs = Path(args.output_dir)
    results: dict[str, Any] = {"sensor_status": sense_status()}

    if args.notify:
        results["notification"] = notify_alive()

    if args.camera:
        results["camera"] = capture_camera(outputs / "camera")

    if args.microphone_seconds > 0:
        results["microphone"] = record_microphone(outputs / "audio", args.microphone_seconds)

    row_id = append_decision(
        db_path=db_path,
        input_text=json_dumps(results),
        candidates=[{"name": "stage1_senses_snapshot"}],
        gate_scores=results,
        decision="STAGE1_SENSES_SNAPSHOT",
        approved_by_human=bool(args.approved),
    )
    print(f"APPENDED row_id={row_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 1 senses loop")
    parser.add_argument("--db", default="data/stage1_senses/decision_ledger.sqlite3")
    parser.add_argument("--output-dir", default="data/stage1_senses")
    parser.add_argument("--init-db", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--microphone-seconds", type=int, default=0)
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--approved", action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args(argv)

    db_path = Path(args.db)

    if args.init_db:
        init_db(db_path)
        print(f"DB_READY {db_path}")
        return 0

    if args.self_test:
        return self_test(db_path)

    if args.once:
        return run_once(args)

    print("Stage 1 loop starting. Press Ctrl-C to stop.")
    while True:
        run_once(args)
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
