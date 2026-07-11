import importlib.util
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "stage1_senses_loop.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage1_senses_loop", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Stage1SensesLoopTests(unittest.TestCase):
    def test_init_and_append_decision(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "ledger.sqlite3"
            row_id = mod.append_decision(
                db_path=db,
                input_text="hello",
                candidates=[{"name": "candidate"}],
                gate_scores={"coherence": "not_run"},
                decision="TEST_APPEND",
                approved_by_human=False,
            )
            self.assertEqual(row_id, 1)

            with sqlite3.connect(db) as con:
                rows = con.execute(
                    "SELECT input_hash, decision, approved_by_human FROM decision_ledger"
                ).fetchall()

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][1], "TEST_APPEND")
            self.assertEqual(rows[0][2], 0)

    def test_ledger_blocks_update_and_delete(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "ledger.sqlite3"
            mod.append_decision(
                db_path=db,
                input_text="hello",
                candidates=[],
                gate_scores={},
                decision="TEST_APPEND",
                approved_by_human=True,
            )

            with sqlite3.connect(db) as con:
                with self.assertRaises(sqlite3.DatabaseError):
                    con.execute("UPDATE decision_ledger SET decision='CHANGED' WHERE id=1")
                with self.assertRaises(sqlite3.DatabaseError):
                    con.execute("DELETE FROM decision_ledger WHERE id=1")

    def test_self_test_cli(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "ledger.sqlite3"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    "--self-test",
                    "--db",
                    str(db),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("SELF_TEST_OK", proc.stdout)

            with sqlite3.connect(db) as con:
                count = con.execute("SELECT COUNT(*) FROM decision_ledger").fetchone()[0]

            self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
