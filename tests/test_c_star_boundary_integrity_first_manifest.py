import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "ai_infrastructure" / "experiments" / "C_STAR_BOUNDARY_INTEGRITY_FIRST_MEASUREMENT_MANIFEST_008P.json"
VIEWER = REPO_ROOT / "tools" / "experiment_manifest_viewer.py"
DECISION_PATH = REPO_ROOT / "ai_infrastructure" / "decisions" / "C_STAR_BOUNDARY_INTEGRITY_FIRST_MEASUREMENT_MANIFEST_DECISION.md"
RUNBOOK_PATH = REPO_ROOT / "app_infrastructure" / "interfaces" / "C_STAR_BOUNDARY_INTEGRITY_FIRST_MEASUREMENT_MANIFEST_RUNBOOK.md"


class CStarBoundaryIntegrityFirstManifestTests(unittest.TestCase):
    def load_manifest(self):
        return json.loads(MANIFEST_PATH.read_text())

    def test_manifest_is_superseded_stage_2_record(self):
        manifest = self.load_manifest()

        self.assertEqual(manifest["manifest_id"], "cstar_boundary_integrity_008p_v1")
        self.assertEqual(manifest["manifest_version"], "1.0")
        self.assertEqual(manifest["status"], "superseded")
        self.assertEqual(manifest["stage"], "Stage 2 draft-only")
        self.assertEqual(manifest["pre_registration"]["manifest_sha_status"], "pending_commit")
        self.assertIsNone(manifest["pre_registration"]["manifest_commit_sha"])

    def test_manifest_targets_boundary_integrity_only(self):
        manifest = self.load_manifest()
        method = manifest["method"]

        self.assertEqual(method["parameters"]["c_star_component"], "B")
        self.assertEqual(method["parameters"]["component_name"], "Boundary Integrity")
        self.assertFalse(method["parameters"]["numeric_score_allowed"])
        self.assertTrue(method["parameters"]["requires_null_model"])
        self.assertTrue(method["parameters"]["requires_operator_review"])
        self.assertIn("Boundary Integrity", manifest["hypothesis"]["statement"])

    def test_manifest_requires_null_model_and_blocks_overclaiming(self):
        manifest = self.load_manifest()

        self.assertIn("static checklist", manifest["null_model"]["description"])
        self.assertIn("deterministic planbook", manifest["null_model"]["description"])
        self.assertEqual(manifest["claim_boundary"]["maximum_claim_grade_without_new_manifest"], "exploratory")
        self.assertTrue(manifest["claim_boundary"]["upgrade_requires_new_manifest"])

        joined_claims = "\n".join(manifest["claim_boundary"]["forbidden_claims"])
        self.assertIn("Do not claim consciousness.", joined_claims)
        self.assertIn("Do not assign numeric C_star scores without measurement data.", joined_claims)
        self.assertIn("Do not present observation-bank material as proof.", joined_claims)

    def test_manifest_preserves_dry_run_and_safety_boundaries(self):
        manifest = self.load_manifest()

        self.assertFalse(manifest["dry_run_boundary"]["can_execute_method_script"])
        self.assertTrue(manifest["dry_run_boundary"]["prints_would_run_command_only"])
        self.assertEqual(
            manifest["dry_run_boundary"]["execution_stage_required"],
            "Stage 3 approved execution",
        )
        self.assertTrue(manifest["dry_run_boundary"]["approval_required_before_execution"])

        for key, value in manifest["safety"].items():
            self.assertFalse(value, key)

    def test_manifest_viewer_accepts_manifest(self):
        result = subprocess.run(
            [
                sys.executable,
                str(VIEWER),
                "--repo",
                str(REPO_ROOT),
                "--manifest",
                str(MANIFEST_PATH),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("- status: PASS", result.stdout)
        self.assertIn(
            "manifest is superseded; it cannot authorize a future execution",
            result.stdout,
        )
        self.assertIn("cstar_boundary_integrity_008p_v1", result.stdout)
        self.assertIn("Boundary Integrity", result.stdout)
        self.assertIn("This viewer cannot execute the would-run command.", result.stdout)

    def test_decision_and_runbook_record_standing_sentence(self):
        decision = DECISION_PATH.read_text()
        runbook = RUNBOOK_PATH.read_text()

        self.assertIn("Measure the boundary before measuring the mind.", decision)
        self.assertIn("Measure the boundary before measuring the mind.", runbook)
        self.assertIn("No experiment execution.", decision)
        self.assertIn("It does not run the experiment.", runbook)


if __name__ == "__main__":
    unittest.main()
