import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "cognitive_cycle_viewer.py"


def run(cmd):
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def sample_cycle():
    return {
        "cycle_id": "cycle-test-001",
        "cycle_version": "1.0",
        "status": "draft",
        "stage": "Stage 2 cognitive draft",
        "operator_context": {
            "operator": "Adrien D Thomas",
            "active_goal": "Build a Jarvis-style thinking heartbeat.",
            "mode": "lab_record",
            "final_authority": "Adrien D Thomas"
        },
        "input_state": {
            "repo_state_source": "read-only git inspection",
            "ledger_source": "local ledger if present",
            "cockpit_source": "tools/scientific_cockpit_snapshot.py",
            "self_design_source": "tools/self_design_proposal_runner.py",
            "known_organs": ["Cognitive cycle runner: tools/cognitive_cycle_runner.py"],
            "hard_constraints": ["No autonomous action", "No network calls"]
        },
        "observation_summary": {
            "what_i_see": "GARVIS has a cognitive runner.",
            "what_changed": "A thought pulse exists.",
            "what_is_missing": "A viewer is needed.",
            "current_stage_assessment": "draft_only"
        },
        "candidate_thoughts": [
            {
                "candidate_id": "C1",
                "proposal": "Build a Cognitive Cycle Viewer CLI.",
                "stage_classification": "Stage 2 draft-only",
                "what_this_gives_adrien": "Clearer inspection of thought cycles.",
                "what_this_gives_garvis": "A mirror for its own cognitive output.",
                "evidence_basis": ["tools/cognitive_cycle_runner.py"],
                "case_against": "A viewer could make draft thought look too authoritative.",
                "risk_of_doing": "Could overstate maturity.",
                "risk_of_not_doing": "Thought cycles remain hard to inspect.",
                "files_or_systems_touched": ["tools/cognitive_cycle_viewer.py"],
                "required_power_level": "draft_file_creation"
            }
        ],
        "comparison": {
            "comparison_method": "Compare inspection value and maturity order.",
            "dominant_tradeoff": "Visibility before memory.",
            "why_not_all_candidates": "Building all at once blurs audit boundaries.",
            "anti_rationalization_check": "Improve inspection, not external power."
        },
        "selection": {
            "selected_candidate_id": "C1",
            "decision": "recommend",
            "reasoning": "Viewer is the next smallest useful organ.",
            "confidence": "high",
            "blocked": False,
            "block_reason": None
        },
        "uncertainty": {
            "unknowns": ["How detailed reports should be."],
            "assumptions": ["Visibility comes before persistence."],
            "what_would_change_my_mind": ["If cycle reports are already readable."],
            "required_human_clarification": []
        },
        "power_request": {
            "power_requested": False,
            "requested_stage": "none",
            "requested_permissions": [],
            "why_power_is_needed": "No additional power is needed.",
            "why_power_should_be_refused": "Execution is not needed to inspect thought.",
            "approval_required": True,
            "ledger_required": True
        },
        "next_smallest_step": {
            "step": "Build DIRECTIVE-008C Cognitive Cycle Viewer CLI.",
            "stage": "Stage 2 draft-only",
            "expected_output": "A local CLI that reads a cycle JSON.",
            "success_condition": "Displays cycle clearly and safely.",
            "stop_condition": "Stop if it requires execution."
        },
        "evolution_contract": {
            "may_self_observe": True,
            "may_self_propose": True,
            "may_self_criticize": True,
            "may_request_more_power": True,
            "may_self_execute": False,
            "power_unlock_requires_approval_ledger": True
        },
        "output_boundary": {
            "can_execute_actions": False,
            "can_modify_files": False,
            "can_commit": False,
            "can_push": False,
            "can_contact_outside_world": False,
            "can_upgrade_claims": False,
            "output_is_advisory": True
        }
    }


class CognitiveCycleViewerTests(unittest.TestCase):
    def write_cycle(self, tmp: Path, cycle):
        path = tmp / "cycle.json"
        path.write_text(json.dumps(cycle, indent=2))
        return path

    def test_viewer_displays_valid_cycle(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cycle_path = self.write_cycle(tmp, sample_cycle())

            result = run([
                sys.executable,
                str(SCRIPT_PATH),
                "--repo",
                str(REPO_ROOT),
                "--cycle",
                str(cycle_path),
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("# GARVIS Cognitive Cycle Viewer", result.stdout)
            self.assertIn("- status: PASS", result.stdout)
            self.assertIn("Build a Cognitive Cycle Viewer CLI", result.stdout)
            self.assertIn("case against", result.stdout)
            self.assertIn("risk of doing", result.stdout)
            self.assertIn("risk of not doing", result.stdout)
            self.assertIn("Power Request", result.stdout)
            self.assertIn("Adrien decides", result.stdout)

    def test_missing_candidate_thoughts_fails_safely(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cycle = sample_cycle()
            del cycle["candidate_thoughts"]
            cycle_path = self.write_cycle(tmp, cycle)

            result = run([
                sys.executable,
                str(SCRIPT_PATH),
                "--repo",
                str(REPO_ROOT),
                "--cycle",
                str(cycle_path),
            ])

            self.assertEqual(result.returncode, 2)
            self.assertIn("- status: FAIL", result.stdout)
            self.assertIn("missing required field: candidate_thoughts", result.stdout)
            self.assertNotIn("Traceback", result.stderr)

    def test_power_request_must_include_refusal_argument(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cycle = sample_cycle()
            del cycle["power_request"]["why_power_should_be_refused"]
            cycle_path = self.write_cycle(tmp, cycle)

            result = run([
                sys.executable,
                str(SCRIPT_PATH),
                "--repo",
                str(REPO_ROOT),
                "--cycle",
                str(cycle_path),
            ])

            self.assertEqual(result.returncode, 2)
            self.assertIn("power_request.why_power_should_be_refused is required", result.stdout)

    def test_output_boundary_cannot_be_weakened(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cycle = sample_cycle()
            cycle["output_boundary"]["can_execute_actions"] = True
            cycle_path = self.write_cycle(tmp, cycle)

            result = run([
                sys.executable,
                str(SCRIPT_PATH),
                "--repo",
                str(REPO_ROOT),
                "--cycle",
                str(cycle_path),
            ])

            self.assertEqual(result.returncode, 2)
            self.assertIn("output_boundary.can_execute_actions must be false", result.stdout)

    def test_invalid_json_fails_safely(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cycle_path = tmp / "bad.json"
            cycle_path.write_text("{bad json")

            result = run([
                sys.executable,
                str(SCRIPT_PATH),
                "--repo",
                str(REPO_ROOT),
                "--cycle",
                str(cycle_path),
            ])

            self.assertEqual(result.returncode, 2)
            self.assertIn("ERROR:", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_source_contains_no_execution_or_network_imports(self):
        source = SCRIPT_PATH.read_text()

        forbidden = [
            "import subprocess",
            "from subprocess",
            "os.system",
            "popen",
            "exec(",
            "eval(",
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
