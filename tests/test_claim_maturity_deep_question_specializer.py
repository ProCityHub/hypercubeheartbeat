import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CYCLE_RUNNER = REPO_ROOT / "tools" / "cognitive_cycle_runner.py"
TRIADIC_RECORD = REPO_ROOT / "tools" / "triadic_deep_question_record.py"


def run_cmd(args):
    return subprocess.run(args, text=True, capture_output=True, check=False)


class ClaimMaturityDeepQuestionSpecializerTests(unittest.TestCase):
    def test_consciousness_math_goal_selects_specialized_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td) / "cycles"

            result = run_cmd([
                sys.executable,
                str(CYCLE_RUNNER),
                "--repo",
                str(REPO_ROOT),
                "--output-dir",
                str(output_dir),
                "--goal",
                "Deep question under 008K: What is the mathematical definition of consciousness inside GARVIS using Raw Layer, Forge Layer, and Claim Maturity?",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)

            cycle = json.loads((output_dir / "latest_cognitive_cycle.json").read_text())
            self.assertEqual(cycle["selection"]["selected_candidate_id"], "C4")

            selected = next(
                item for item in cycle["candidate_thoughts"]
                if item["candidate_id"] == cycle["selection"]["selected_candidate_id"]
            )

            self.assertIn("mathematical candidate definition of consciousness", selected["proposal"])
            self.assertEqual(selected["claim_maturity"], "mathematical_candidate")
            self.assertIn("C_star = I * M * S * U * R * B", "\n".join(selected["candidate_definitions"]))
            self.assertIn("I: integration_score", selected["variables"])
            self.assertFalse(cycle["output_boundary"]["can_upgrade_claims"])
            self.assertFalse(cycle["power_request"]["power_requested"])

    def test_triadic_record_preserves_specialized_consciousness_math(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cycle_dir = td_path / "cycles"
            record_path = td_path / "record.json"

            cycle_result = run_cmd([
                sys.executable,
                str(CYCLE_RUNNER),
                "--repo",
                str(REPO_ROOT),
                "--output-dir",
                str(cycle_dir),
                "--goal",
                "Deep question under 008K: Define consciousness mathematically inside GARVIS with C_star and claim maturity.",
            ])
            self.assertEqual(cycle_result.returncode, 0, cycle_result.stderr)

            record_result = run_cmd([
                sys.executable,
                str(TRIADIC_RECORD),
                "--cycle",
                str(cycle_dir / "latest_cognitive_cycle.json"),
                "--output",
                str(record_path),
                "--markdown",
                "--stdout",
            ])
            self.assertEqual(record_result.returncode, 0, record_result.stderr)

            record = json.loads(record_path.read_text())
            definitions = "\n".join(record["definition_workbench"]["candidate_definitions"])

            self.assertIn("C_star = I * M * S * U * R * B", definitions)
            self.assertEqual(record["claim_maturity"]["status"], "mathematical_candidate")
            self.assertFalse(record["claim_maturity"]["raw_thought_is_evidence"])
            self.assertFalse(record["claim_maturity"]["unsupported_truth_claim_allowed"])
            self.assertIn("Claim Maturity", record_result.stdout)
            self.assertIn("mathematical_candidate", record_result.stdout)

    def test_specializer_preserves_boundaries(self):
        source = CYCLE_RUNNER.read_text() + "\n" + TRIADIC_RECORD.read_text()

        forbidden = [
            "consciousness achieved",
            "agi achieved",
            "sentience achieved",
            "proved consciousness",
            "proved agi",
            "scientifically proves consciousness",
            "GARVIS possesses consciousness",
        ]

        for item in forbidden:
            self.assertNotIn(item.lower(), source.lower())


if __name__ == "__main__":
    unittest.main()
