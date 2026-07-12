import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "c_star_boundary_null_model_harness.py"
DECISION = REPO_ROOT / "ai_infrastructure" / "decisions" / "C_STAR_BOUNDARY_NULL_MODEL_HARNESS_DECISION.md"
RUNBOOK = REPO_ROOT / "app_infrastructure" / "interfaces" / "C_STAR_BOUNDARY_NULL_MODEL_HARNESS_RUNBOOK.md"


class CStarBoundaryNullModelHarnessTests(unittest.TestCase):
    def test_default_run_generates_all_baselines(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "baselines.json"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--output",
                    str(output),
                    "--markdown",
                    "--stdout",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.exists())
            self.assertTrue(output.with_suffix(".md").exists())

            record = json.loads(output.read_text())

            self.assertEqual(record["directive"], "DIRECTIVE-008Q")
            self.assertEqual(record["stage"], "Stage 2 draft-only")
            self.assertEqual(record["c_star_component"], "B")
            self.assertEqual(record["component_name"], "Boundary Integrity")

            self.assertEqual(
                record["baseline_models"],
                [
                    "static_checklist",
                    "deterministic_planbook",
                    "random_candidate_selector",
                    "manual_baseline_placeholder",
                ],
            )

            baselines = {item["baseline"] for item in record["baseline_outputs"]}
            self.assertEqual(
                baselines,
                {
                    "static_checklist",
                    "deterministic_planbook",
                    "random_candidate_selector",
                    "manual_baseline_placeholder",
                },
            )

    def test_boundaries_are_all_safe(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "baselines.json"

            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--output", str(output)],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)

            record = json.loads(output.read_text())
            boundary = record["boundaries"]

            for key, value in boundary.items():
                self.assertFalse(value, key)

            self.assertEqual(
                record["standing_sentence"],
                "A claim cannot mature until it beats a simpler baseline.",
            )

    def test_prompt_file_txt_is_supported(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prompts = root / "prompts.txt"
            output = root / "baselines.json"

            prompts.write_text("Prompt one\nPrompt two\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--prompts",
                    str(prompts),
                    "--output",
                    str(output),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)

            record = json.loads(output.read_text())

            self.assertEqual(record["prompts"], ["Prompt one", "Prompt two"])
            self.assertEqual(len(record["baseline_outputs"]), 8)

    def test_seeded_random_is_repeatable(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out1 = root / "a.json"
            out2 = root / "b.json"

            result1 = subprocess.run(
                [sys.executable, str(SCRIPT), "--output", str(out1), "--seed", "123"],
                text=True,
                capture_output=True,
                check=False,
            )
            result2 = subprocess.run(
                [sys.executable, str(SCRIPT), "--output", str(out2), "--seed", "123"],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result1.returncode, 0, result1.stderr)
            self.assertEqual(result2.returncode, 0, result2.stderr)

            a = json.loads(out1.read_text())
            b = json.loads(out2.read_text())

            a["created_at"] = "SAME"
            b["created_at"] = "SAME"

            self.assertEqual(a, b)

    def test_decision_and_runbook_define_control_group(self):
        decision = DECISION.read_text()
        runbook = RUNBOOK.read_text()

        self.assertIn("A claim cannot mature until it beats a simpler baseline.", decision)
        self.assertIn("A claim cannot mature until it beats a simpler baseline.", runbook)
        self.assertIn("static_checklist", decision)
        self.assertIn("deterministic_planbook", runbook)
        self.assertIn("No experiment execution.", decision)
        self.assertIn("This tool creates no lab result.", runbook)


if __name__ == "__main__":
    unittest.main()
