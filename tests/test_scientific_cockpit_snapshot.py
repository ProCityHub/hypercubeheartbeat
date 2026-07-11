import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "scientific_cockpit_snapshot.py"


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)


def create_temp_git_repo(path: Path) -> None:
    run(["git", "init"], cwd=path)
    run(["git", "config", "user.email", "test@example.invalid"], cwd=path)
    run(["git", "config", "user.name", "Test User"], cwd=path)

    (path / "tools").mkdir()
    (path / "tests").mkdir()
    (path / "ai_infrastructure" / "decisions").mkdir(parents=True)
    (path / "app_infrastructure" / "interfaces").mkdir(parents=True)

    (path / "tools" / "example.py").write_text("print('example')\n")
    (path / "tests" / "test_example.py").write_text("def test_example():\n    assert True\n")
    (path / "ai_infrastructure" / "decisions" / "EXAMPLE_DECISION.md").write_text("# Decision\n")
    (path / "app_infrastructure" / "interfaces" / "EXAMPLE_RUNBOOK.md").write_text("# Runbook\n")

    run(["git", "add", "."], cwd=path)
    run(["git", "commit", "-m", "init"], cwd=path)


def create_ledger(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE decision_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                candidates_json TEXT NOT NULL,
                gate_scores_json TEXT NOT NULL,
                decision TEXT NOT NULL,
                approved_by_human INTEGER NOT NULL CHECK (approved_by_human IN (0, 1))
            )
            """
        )
        conn.execute(
            """
            INSERT INTO decision_ledger (
                timestamp,
                input_hash,
                candidates_json,
                gate_scores_json,
                decision,
                approved_by_human
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "2026-07-11T21:10:00Z",
                "abcdef1234567890SECRET_HASH_TAIL",
                '{"secret":"RAW_JSON_SHOULD_NOT_PRINT"}',
                '{"secret":"RAW_GATE_SHOULD_NOT_PRINT"}',
                "STAGE1_SENSES_SNAPSHOT",
                1,
            ),
        )
        conn.commit()
    finally:
        conn.close()


class ScientificCockpitSnapshotTests(unittest.TestCase):
    def test_snapshot_runs_against_current_repo(self):
        result = run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--repo",
                str(REPO_ROOT),
                "--limit",
                "5",
            ]
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("# GARVIS Scientific Cockpit Snapshot", result.stdout)
        self.assertIn("mode: read-only", result.stdout)
        self.assertIn("stage: Stage 0 view-only", result.stdout)
        self.assertIn("actions_executed: none", result.stdout)
        self.assertIn("network_calls: none", result.stdout)
        self.assertIn("llm_calls: none", result.stdout)
        self.assertIn("## Do Not Commit Watch", result.stdout)
        self.assertIn("## Tools", result.stdout)
        self.assertIn("## Tests", result.stdout)

    def test_snapshot_reads_ledger_metadata_without_raw_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            create_temp_git_repo(repo)
            ledger = repo / "ledger.sqlite3"
            create_ledger(ledger)

            result = run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--repo",
                    str(repo),
                    "--ledger-db",
                    str(ledger),
                ]
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ledger: available", result.stdout)
            self.assertIn("ledger_mode: read-only", result.stdout)
            self.assertIn("ledger_rows: 1", result.stdout)
            self.assertIn("latest_decision: STAGE1_SENSES_SNAPSHOT", result.stdout)
            self.assertNotIn("RAW_JSON_SHOULD_NOT_PRINT", result.stdout)
            self.assertNotIn("RAW_GATE_SHOULD_NOT_PRINT", result.stdout)
            self.assertNotIn("SECRET_HASH_TAIL", result.stdout)

    def test_missing_ledger_fails_open_not_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            create_temp_git_repo(repo)

            result = run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--repo",
                    str(repo),
                    "--ledger-db",
                    str(repo / "missing.sqlite3"),
                ]
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ledger: not found", result.stdout)
            self.assertIn("ledger_mode: unavailable", result.stdout)

    def test_non_git_repo_fails_safely(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--repo",
                    tmp,
                ]
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("ERROR:", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_source_contains_no_network_imports(self):
        source = SCRIPT_PATH.read_text()
        forbidden = [
            "import requests",
            "from requests",
            "import urllib",
            "from urllib",
            "import socket",
            "from socket",
            "http://",
            "https://",
        ]
        for item in forbidden:
            self.assertNotIn(item, source)


if __name__ == "__main__":
    unittest.main()
