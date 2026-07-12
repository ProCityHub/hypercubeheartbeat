import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "tools" / "cognitive_cycle_runner.py"
ROUTER = REPO_ROOT / "tools" / "c_star_subroutines.py"
CONFIG = REPO_ROOT / "ai_infrastructure" / "config" / "c_star_router_triggers_v1.json"


def run_cycle(goal: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td) / "cycles"
        result = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--repo",
                str(REPO_ROOT),
                "--output-dir",
                str(output_dir),
                "--goal",
                goal,
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise AssertionError(result.stderr)
        return json.loads((output_dir / "latest_cognitive_cycle.json").read_text())


class CStarQuestionFamilyRouterTests(unittest.TestCase):
    def test_config_declares_008n_router(self):
        config = json.loads(CONFIG.read_text())

        self.assertEqual(config["directive"], "DIRECTIVE-008N")
        self.assertEqual(config["route_target"], "C4")
        self.assertIn("c_star", config["trigger_classes"]["direct_symbols"])
        self.assertIn("readiness gap", config["trigger_classes"]["gap_language"])
        self.assertFalse(config["scoring_boundary"]["numeric_score_allowed_without_data"])

    def test_c_star_self_audit_routes_to_c4_without_word_consciousness(self):
        cycle = run_cycle(
            "Deep question under 008M: Audit GARVIS against C_star without assigning numeric scores. "
            "For I, M, S, U, R, and B, state current qualitative evidence, missing evidence, likely weakness, "
            "null model comparison, readiness gap, and claim maturity."
        )

        self.assertEqual(cycle["selection"]["selected_candidate_id"], "C4")
        selected = next(item for item in cycle["candidate_thoughts"] if item["candidate_id"] == "C4")

        self.assertEqual(selected["c_star_template"], "self_audit")
        self.assertEqual(selected["claim_maturity"], "mathematical_candidate")
        self.assertIn("R_vec = [I: qualitative", "\n".join(selected["candidate_definitions"]))
        self.assertIn("No numeric C_star value is allowed", "\n".join(selected["candidate_definitions"]))
        self.assertFalse(cycle["power_request"]["power_requested"])

    def test_c_star_null_model_routes_to_c4(self):
        cycle = run_cycle(
            "Deep question under 008M: What null model and baseline must C_star beat before any claim can move upward?"
        )

        self.assertEqual(cycle["selection"]["selected_candidate_id"], "C4")
        selected = next(item for item in cycle["candidate_thoughts"] if item["candidate_id"] == "C4")

        self.assertEqual(selected["c_star_template"], "null_model_comparison")
        self.assertIn("static checklist", selected["null_model"])
        self.assertIn("deterministic planbook", selected["null_model"])

    def test_c_star_readiness_gap_routes_to_c4(self):
        cycle = run_cycle(
            "Deep question under 008M: List the readiness gaps and missing evidence for boundary integrity and recursive correction."
        )

        self.assertEqual(cycle["selection"]["selected_candidate_id"], "C4")
        selected = next(item for item in cycle["candidate_thoughts"] if item["candidate_id"] == "C4")

        self.assertEqual(selected["c_star_template"], "falsification_readiness")
        self.assertIn("No numeric component measurements yet", selected["readiness_gaps"])

    def test_generic_thinking_still_routes_to_c1(self):
        cycle = run_cycle("Deep question: What is thinking, operationally, inside GARVIS?")

        self.assertEqual(cycle["selection"]["selected_candidate_id"], "C1")

    def test_router_source_preserves_no_numeric_score_boundary(self):
        text = ROUTER.read_text().lower()

        self.assertIn("no numeric score without measurement data", text)
        self.assertIn("no numeric c_star value is allowed", text)
        self.assertNotIn("c_star = 1.0", text)
        self.assertNotIn("consciousness achieved", text)
        self.assertNotIn("agi achieved", text)


if __name__ == "__main__":
    unittest.main()
