import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "ai_infrastructure" / "schemas" / "experiment_manifest_schema_v1.json"
RUNBOOK_PATH = REPO_ROOT / "app_infrastructure" / "interfaces" / "EXPERIMENT_MANIFEST_SCHEMA_RUNBOOK.md"
DECISION_PATH = REPO_ROOT / "ai_infrastructure" / "decisions" / "EXPERIMENT_MANIFEST_SCHEMA_DECISION.md"


class ExperimentManifestSchemaTests(unittest.TestCase):
    def load_schema(self):
        return json.loads(SCHEMA_PATH.read_text())

    def test_schema_loads_as_json(self):
        schema = self.load_schema()
        self.assertEqual(schema["title"], "GARVIS Experiment Manifest Schema v1")
        self.assertEqual(schema["properties"]["stage"]["const"], "Stage 2 draft-only")

    def test_all_four_amendments_are_structurally_required(self):
        schema = self.load_schema()
        required = set(schema["required"])

        for field in [
            "pre_registration",
            "null_model",
            "claim_boundary",
            "dry_run_boundary",
            "ledger_chain",
        ]:
            self.assertIn(field, required)

        pre_registration = schema["properties"]["pre_registration"]
        self.assertIn("manifest_commit_sha", pre_registration["required"])
        self.assertEqual(
            pre_registration["properties"]["result_must_cite_manifest_sha"]["const"],
            True,
        )
        self.assertEqual(
            pre_registration["properties"]["manifest_must_predate_run"]["const"],
            True,
        )
        self.assertEqual(
            pre_registration["properties"]["result_invalid_if_manifest_postdates_run"]["const"],
            True,
        )

        null_model = schema["properties"]["null_model"]
        for field in [
            "description",
            "expected_noise_behavior",
            "false_positive_risk",
            "comparison_method",
            "sample_size_or_trials",
        ]:
            self.assertIn(field, null_model["required"])

        dry_run = schema["properties"]["dry_run_boundary"]
        self.assertEqual(dry_run["properties"]["can_execute_method_script"]["const"], False)
        self.assertEqual(dry_run["properties"]["prints_would_run_command_only"]["const"], True)
        self.assertEqual(
            dry_run["properties"]["execution_stage_required"]["const"],
            "Stage 3 approved execution",
        )

    def test_claim_vocabulary_is_closed(self):
        schema = self.load_schema()
        claim_items = (
            schema["properties"]["claim_boundary"]
            ["properties"]["allowed_result_grades"]
            ["items"]
            ["enum"]
        )
        self.assertEqual(
            sorted(claim_items),
            sorted(["exploratory", "suggestive", "supported", "retracted"]),
        )

        max_grade = (
            schema["properties"]["claim_boundary"]
            ["properties"]["maximum_claim_grade_without_new_manifest"]
            ["enum"]
        )
        self.assertEqual(sorted(max_grade), sorted(claim_items))

        self.assertEqual(
            schema["properties"]["claim_boundary"]
            ["properties"]["upgrade_requires_new_manifest"]
            ["const"],
            True,
        )

    def test_safety_defaults_block_external_power(self):
        schema = self.load_schema()
        safety = schema["properties"]["safety"]["properties"]

        for key in [
            "network_allowed",
            "llm_calls_allowed",
            "external_contact_allowed",
            "secret_access_allowed",
            "raw_sensor_payload_access_allowed",
        ]:
            self.assertEqual(safety[key]["const"], False)

    def test_runbook_records_amendments(self):
        text = RUNBOOK_PATH.read_text()

        for phrase in [
            "pre-registration hash",
            "mandatory null model",
            "fixed claim vocabulary",
            "dry-run means cannot-run",
            "manifest commit SHA",
            "exploratory",
            "suggestive",
            "supported",
            "retracted",
        ]:
            self.assertIn(phrase, text)

    def test_decision_record_contains_no_execution_authorization(self):
        text = DECISION_PATH.read_text()

        for forbidden in [
            "experiment runner\n",
            "local execution engine\n",
            "approval engine\n",
            "action queue\n",
            "autonomous action\n",
            "claim upgrade\n",
        ]:
            self.assertIn(forbidden, text)

        self.assertIn("It does not execute experiments.", text)


if __name__ == "__main__":
    unittest.main()
