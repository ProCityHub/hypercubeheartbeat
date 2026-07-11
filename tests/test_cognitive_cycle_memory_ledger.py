import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "cognitive_cycle_memory_ledger.py"


def run(cmd):
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


class CognitiveCycleMemoryLedgerTests(unittest.TestCase):
    def test_init_db_creates_required_tables_and_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "memory.sqlite3"

            result = run([
                sys.executable,
                str(SCRIPT_PATH),
                "--db",
                str(db),
                "--init-db",
                "--verify",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("# GARVIS Cognitive Cycle Memory Ledger", result.stdout)
            self.assertIn("append_behavior: not implemented", result.stdout)
            self.assertIn("open_problem_box=placeholder_table_only", result.stdout)

            with sqlite3.connect(db) as conn:
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }

                self.assertIn("ledger_metadata", tables)
                self.assertIn("cognitive_cycle_records", tables)
                self.assertIn("open_problem_placeholders", tables)

                metadata = dict(conn.execute("SELECT key, value FROM ledger_metadata").fetchall())

            self.assertEqual(metadata["schema_version"], "1.0")
            self.assertEqual(metadata["append_behavior"], "not_implemented")
            self.assertEqual(metadata["network_allowed"], "false")
            self.assertEqual(metadata["claim_upgrade_allowed"], "false")

    def test_verify_without_existing_database_fails_safely(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "missing.sqlite3"

            result = run([
                sys.executable,
                str(SCRIPT_PATH),
                "--db",
                str(db),
                "--verify",
            ])

            self.assertEqual(result.returncode, 2)
            self.assertIn("ERROR:", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_no_arguments_fails_safely(self):
        result = run([sys.executable, str(SCRIPT_PATH)])

        self.assertEqual(result.returncode, 2)
        self.assertIn("choose --init-db and/or --verify", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_records_table_contains_claim_and_approval_boundaries(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "memory.sqlite3"

            result = run([
                sys.executable,
                str(SCRIPT_PATH),
                "--db",
                str(db),
                "--init-db",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)

            with sqlite3.connect(db) as conn:
                columns = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(cognitive_cycle_records)").fetchall()
                }

            for column in [
                "record_is_memory_not_action",
                "record_is_not_approval",
                "record_is_not_empirical_result",
                "record_is_not_claim_upgrade",
                "permission_granted_by_this_record",
                "why_power_should_be_refused",
                "approval_required",
                "ledger_required",
            ]:
                self.assertIn(column, columns)

    def test_open_problem_placeholder_table_is_no_claim_box_seed(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "memory.sqlite3"

            result = run([
                sys.executable,
                str(SCRIPT_PATH),
                "--db",
                str(db),
                "--init-db",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)

            with sqlite3.connect(db) as conn:
                columns = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(open_problem_placeholders)").fetchall()
                }

            for column in [
                "problem_id",
                "problem_title",
                "problem_status",
                "claim_status",
                "allowed_activity",
                "forbidden_activity",
                "created_at_utc",
            ]:
                self.assertIn(column, columns)

    def test_source_contains_no_network_llm_or_append_behavior(self):
        source = SCRIPT_PATH.read_text()

        forbidden = [
            "import requests",
            "from requests",
            "import urllib",
            "from urllib",
            "import socket",
            "from socket",
            "openai",
            "anthropic",
            "http://",
            "https://",
            "append_cycle",
            "insert_cycle_record",
            "subprocess",
            "os.system",
            "popen",
            "exec(",
            "eval(",
        ]

        for item in forbidden:
            self.assertNotIn(item, source)


if __name__ == "__main__":
    unittest.main()
