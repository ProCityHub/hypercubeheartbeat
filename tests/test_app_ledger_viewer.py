import importlib.util
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "app_ledger_viewer.py"


def load_viewer_module():
    module_name = "app_ledger_viewer_under_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def create_test_ledger(path: Path) -> None:
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
                "2026-07-11T20:20:00Z",
                "abcdef1234567890SECRET_HASH_TAIL",
                '{"raw_path":"SECRET_CAMERA_PATH_SHOULD_NOT_PRINT","kind":"camera"}',
                '{"private_detail":"SECRET_GATE_VALUE_SHOULD_NOT_PRINT"}',
                "STAGE1_SENSES_SNAPSHOT",
                1,
            ),
        )
        conn.commit()
    finally:
        conn.close()


class AppLedgerViewerTests(unittest.TestCase):
    def test_cli_displays_safe_metadata_without_raw_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "ledger.sqlite3"
            create_test_ledger(db_path)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--db",
                    str(db_path),
                    "--limit",
                    "5",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("GARVIS App Ledger Viewer", result.stdout)
            self.assertIn("mode: read-only", result.stdout)
            self.assertIn("raw_payloads: hidden", result.stdout)
            self.assertIn("STAGE1_SENSES_SNAPSHOT", result.stdout)
            self.assertIn("1", result.stdout)
            self.assertIn("candidates:dict", result.stdout)
            self.assertIn("gates:dict", result.stdout)

            self.assertNotIn("SECRET_CAMERA_PATH_SHOULD_NOT_PRINT", result.stdout)
            self.assertNotIn("SECRET_GATE_VALUE_SHOULD_NOT_PRINT", result.stdout)
            self.assertNotIn("SECRET_HASH_TAIL", result.stdout)

    def test_readonly_connection_rejects_write(self):
        viewer = load_viewer_module()

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "ledger.sqlite3"
            create_test_ledger(db_path)

            conn = viewer.connect_read_only(db_path)
            try:
                with self.assertRaises(sqlite3.OperationalError):
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
                            "2026-07-11T20:21:00Z",
                            "write-attempt",
                            "[]",
                            "{}",
                            "SHOULD_FAIL",
                            0,
                        ),
                    )
            finally:
                conn.close()

    def test_missing_database_fails_safely_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.sqlite3"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--db",
                    str(missing),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("ERROR:", result.stderr)
            self.assertIn("ledger database not found", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_limit_is_capped(self):
        viewer = load_viewer_module()
        self.assertEqual(viewer.clamp_limit(500), viewer.MAX_LIMIT)

    def test_limit_below_one_fails(self):
        viewer = load_viewer_module()
        with self.assertRaises(viewer.ViewerError):
            viewer.clamp_limit(0)


if __name__ == "__main__":
    unittest.main()
