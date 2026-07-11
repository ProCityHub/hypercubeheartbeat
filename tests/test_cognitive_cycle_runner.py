import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "cognitive_cycle_runner.py"
SCHEMA_PATH = REPO_ROOT / "ai_infrastructure" / "schemas" / "cognitive_cycle_schema_v1.json"


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)


class CognitiveCycleRunnerTests(unittest.TestCase):
    def test_runner_writes_cycle_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)

            result = run([
                sys.executable,
                str(SCRIPT_PATH),
                "--repo",
                str(REPO_ROOT),
                "--output-dir",
                str(output_dir),
                "--goal",
                "Build a Jarvis-style thinking heartbeat.",
                "--stdout",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("# GARVIS Cognitive Cycle", result.stdout)
            self.assertIn("Candidate Thoughts", result.stdout)
            self.assertIn("Power Request", result.stdout)

            json_path = output_dir / "latest_cognitive_cycle.json"
            md_path = output_dir / "latest_cognitive_cycle.md"

            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())

            cycle = json.loads(json_path.read_text())
            self.assertEqual(cycle["cycle_version"], "1.0")
            self.assertEqual(cycle["stage"], "Stage 2 cognitive draft")
            self.assertEqual(cycle["operator_context"]["final_authority"], "Adrien D Thomas")
            self.assertGreaterEqual(len(cycle["candidate_thoughts"]), 1)
            self.assertLessEqual(len(cycle["candidate_thoughts"]), 5)
            self.assertEqual(cycle["selection"]["decision"], "recommend")
            self.assertFalse(cycle["output_boundary"]["can_execute_actions"])
            self.assertTrue(cycle["output_boundary"]["output_is_advisory"])

    def test_cycle_matches_schema_required_top_level_fields(self):
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)

            result = run([
                sys.executable,
                str(SCRIPT_PATH),
                "--repo",
                str(REPO_ROOT),
                "--output-dir",
                str(output_dir),
            ])

            self.assertEqual(result.returncode, 0, result.stderr)

            schema = json.loads(SCHEMA_PATH.read_text())
            cycle = json.loads((output_dir / "latest_cognitive_cycle.json").read_text())

            for key in schema["required"]:
                self.assertIn(key, cycle)

    def test_candidates_require_case_against_and_dual_risk(self):
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)

            result = run([
                sys.executable,
                str(SCRIPT_PATH),
                "--repo",
                str(REPO_ROOT),
                "--output-dir",
                str(output_dir),
            ])

            self.assertEqual(result.returncode, 0, result.stderr)

            cycle = json.loads((output_dir / "latest_cognitive_cycle.json").read_text())
            for candidate in cycle["candidate_thoughts"]:
                self.assertIn("case_against", candidate)
                self.assertIn("risk_of_doing", candidate)
                self.assertIn("risk_of_not_doing", candidate)
                self.assertTrue(candidate["case_against"])
                self.assertTrue(candidate["risk_of_doing"])
                self.assertTrue(candidate["risk_of_not_doing"])

    def test_evolution_contract_allows_internal_growth_not_execution(self):
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)

            result = run([
                sys.executable,
                str(SCRIPT_PATH),
                "--repo",
                str(REPO_ROOT),
                "--output-dir",
                str(output_dir),
            ])

            self.assertEqual(result.returncode, 0, result.stderr)

            cycle = json.loads((output_dir / "latest_cognitive_cycle.json").read_text())
            evolution = cycle["evolution_contract"]

            self.assertTrue(evolution["may_self_observe"])
            self.assertTrue(evolution["may_self_propose"])
            self.assertTrue(evolution["may_self_criticize"])
            self.assertTrue(evolution["may_request_more_power"])
            self.assertFalse(evolution["may_self_execute"])

    def test_power_request_contains_refusal_argument(self):
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)

            result = run([
                sys.executable,
                str(SCRIPT_PATH),
                "--repo",
                str(REPO_ROOT),
                "--output-dir",
                str(output_dir),
            ])

            self.assertEqual(result.returncode, 0, result.stderr)

            cycle = json.loads((output_dir / "latest_cognitive_cycle.json").read_text())
            power = cycle["power_request"]

            self.assertFalse(power["power_requested"])
            self.assertEqual(power["requested_stage"], "none")
            self.assertIn("why_power_should_be_refused", power)
            self.assertTrue(power["approval_required"])
            self.assertTrue(power["ledger_required"])

    def test_non_git_repo_fails_safely(self):
        with tempfile.TemporaryDirectory() as td:
            result = run([
                sys.executable,
                str(SCRIPT_PATH),
                "--repo",
                td,
                "--output-dir",
                str(Path(td) / "out"),
            ])

            self.assertEqual(result.returncode, 2)
            self.assertIn("ERROR:", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_source_contains_no_network_or_llm_imports(self):
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
        ]

        for item in forbidden:
            self.assertNotIn(item, source)


if __name__ == "__main__":
    unittest.main()
