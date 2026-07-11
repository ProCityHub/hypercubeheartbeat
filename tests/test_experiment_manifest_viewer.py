import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "experiment_manifest_viewer.py"
SCHEMA_PATH = REPO_ROOT / "ai_infrastructure" / "schemas" / "experiment_manifest_schema_v1.json"


def run(cmd):
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def sample_manifest():
    return {
        "manifest_id": "killer-control-001",
        "manifest_version": "1.0",
        "status": "draft",
        "stage": "Stage 2 draft-only",
        "pre_registration": {
            "manifest_sha_status": "pending_commit",
            "manifest_commit_sha": None,
            "result_must_cite_manifest_sha": True,
            "manifest_must_predate_run": True,
            "result_invalid_if_manifest_postdates_run": True
        },
        "hypothesis": {
            "statement": "Phi-angle circuits show a stronger target signal than arbitrary-angle controls.",
            "rationale": "This tests whether the claimed signal survives a direct control.",
            "scope": "Exploratory local manifest only."
        },
        "prediction": {
            "statement": "The phi-angle condition will exceed the arbitrary-angle control under the stated metric.",
            "measurable_signal": "Difference between target metric distributions.",
            "success_threshold": "Predefined threshold in a future manifest revision."
        },
        "counter_prediction": {
            "statement": "Phi-angle and arbitrary-angle conditions will not separate beyond the null model.",
            "falsifying_signal": "No meaningful separation from arbitrary-angle controls.",
            "expected_if_wrong": "Observed signal falls within noise expectation."
        },
        "null_model": {
            "description": "Randomized arbitrary-angle circuits produce apparent target ratios at a nonzero false-positive rate.",
            "expected_noise_behavior": "Some runs will appear patterned by chance.",
            "false_positive_risk": "Must be estimated before any supported claim.",
            "comparison_method": "Compare target metric against randomized controls.",
            "sample_size_or_trials": "Defined before execution in a locked manifest."
        },
        "data_needed": {
            "inputs": ["phi-angle circuit definition", "arbitrary-angle controls"],
            "exclusions": ["post-hoc selected ratios"],
            "provenance_requirement": "All inputs must be committed or explicitly referenced before run."
        },
        "method": {
            "method_summary": "Future approved run compares phi-angle circuit output against arbitrary-angle controls.",
            "would_run_command": ["python", "tools/future_experiment.py", "--manifest", "manifest.json"],
            "parameters": {"mode": "dry-run-design"},
            "outputs": ["future_result.json"]
        },
        "failure_conditions": {
            "would_weaken_hypothesis": ["Control performs similarly to phi condition."],
            "would_invalidate_run": ["Manifest SHA postdates run.", "Null model missing."],
            "known_failure_modes": ["Post-hoc threshold selection.", "Small sample overfit."]
        },
        "claim_boundary": {
            "allowed_result_grades": ["exploratory", "suggestive", "supported", "retracted"],
            "maximum_claim_grade_without_new_manifest": "exploratory",
            "forbidden_claims": ["proves consciousness", "proves AGI", "proves quantum advantage"],
            "upgrade_requires_new_manifest": True
        },
        "dry_run_boundary": {
            "can_execute_method_script": False,
            "prints_would_run_command_only": True,
            "execution_stage_required": "Stage 3 approved execution",
            "approval_required_before_execution": True
        },
        "approval": {
            "operator": "Adrien D Thomas",
            "approval_status": "not_requested",
            "approval_ledger_id": None
        },
        "ledger_chain": {
            "manifest_commit_sha": None,
            "approval_ledger_id": None,
            "run_id": None,
            "result_id": None,
            "claim_record_id": None
        },
        "safety": {
            "network_allowed": False,
            "llm_calls_allowed": False,
            "external_contact_allowed": False,
            "secret_access_allowed": False,
            "raw_sensor_payload_access_allowed": False
        }
    }


class ExperimentManifestViewerTests(unittest.TestCase):
    def write_manifest(self, tmp: Path, manifest):
        path = tmp / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2))
        return path

    def test_viewer_validates_and_prints_dry_run_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            manifest_path = self.write_manifest(tmp, sample_manifest())

            result = run([
                sys.executable,
                str(SCRIPT_PATH),
                "--repo",
                str(REPO_ROOT),
                "--schema",
                str(SCHEMA_PATH),
                "--manifest",
                str(manifest_path),
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("# GARVIS Experiment Manifest Viewer", result.stdout)
            self.assertIn("mode: validation-only", result.stdout)
            self.assertIn("execution: blocked", result.stdout)
            self.assertIn("- status: PASS", result.stdout)
            self.assertIn("Phi-angle circuits show", result.stdout)
            self.assertIn("## Null Model", result.stdout)
            self.assertIn("## Claim Boundary", result.stdout)
            self.assertIn("python tools/future_experiment.py --manifest manifest.json", result.stdout)
            self.assertIn("manifest commit SHA is not locked", result.stdout)
            self.assertIn("execution approval is missing", result.stdout)

    def test_missing_null_model_fails_safely(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            manifest = sample_manifest()
            del manifest["null_model"]
            manifest_path = self.write_manifest(tmp, manifest)

            result = run([
                sys.executable,
                str(SCRIPT_PATH),
                "--repo",
                str(REPO_ROOT),
                "--schema",
                str(SCHEMA_PATH),
                "--manifest",
                str(manifest_path),
            ])

            self.assertEqual(result.returncode, 2)
            self.assertIn("- status: FAIL", result.stdout)
            self.assertIn("missing required field: null_model", result.stdout)
            self.assertNotIn("Traceback", result.stderr)

    def test_dry_run_boundary_cannot_be_weakened(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            manifest = sample_manifest()
            manifest["dry_run_boundary"]["can_execute_method_script"] = True
            manifest_path = self.write_manifest(tmp, manifest)

            result = run([
                sys.executable,
                str(SCRIPT_PATH),
                "--repo",
                str(REPO_ROOT),
                "--schema",
                str(SCHEMA_PATH),
                "--manifest",
                str(manifest_path),
            ])

            self.assertEqual(result.returncode, 2)
            self.assertIn("dry_run_boundary.can_execute_method_script must be false", result.stdout)

    def test_safety_external_power_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            manifest = sample_manifest()
            manifest["safety"]["network_allowed"] = True
            manifest_path = self.write_manifest(tmp, manifest)

            result = run([
                sys.executable,
                str(SCRIPT_PATH),
                "--repo",
                str(REPO_ROOT),
                "--schema",
                str(SCHEMA_PATH),
                "--manifest",
                str(manifest_path),
            ])

            self.assertEqual(result.returncode, 2)
            self.assertIn("safety.network_allowed must be false", result.stdout)

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
            "http://",
            "https://",
        ]
        for item in forbidden:
            self.assertNotIn(item, source)


if __name__ == "__main__":
    unittest.main()
