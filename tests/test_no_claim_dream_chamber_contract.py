import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "ai_infrastructure" / "schemas" / "no_claim_dream_chamber_record_schema_v1.json"
CONTRACT_PATH = REPO_ROOT / "app_infrastructure" / "interfaces" / "NO_CLAIM_DREAM_CHAMBER_CONTRACT.md"
DECISION_PATH = REPO_ROOT / "ai_infrastructure" / "decisions" / "NO_CLAIM_DREAM_CHAMBER_CONTRACT_DECISION.md"


class NoClaimDreamChamberContractTests(unittest.TestCase):
    def load_schema(self):
        return json.loads(SCHEMA_PATH.read_text())

    def test_schema_loads_as_json(self):
        schema = self.load_schema()
        self.assertEqual(schema["title"], "GARVIS No-Claim Dream Chamber Record Schema v1")
        self.assertEqual(schema["properties"]["record_version"]["const"], "1.0")
        self.assertEqual(schema["properties"]["stage"]["const"], "Stage 2 dream chamber contract")

    def test_schema_requires_dream_and_science_boundaries(self):
        schema = self.load_schema()
        required = set(schema["required"])

        for field in [
            "problem_identity",
            "chamber_classification",
            "claim_boundary",
            "dream_material",
            "scientific_grounding",
            "falsifiability",
            "memory_links",
            "operator_review",
            "output_boundary",
            "safety",
        ]:
            self.assertIn(field, required)

    def test_claim_boundary_forbids_claims_but_allows_hypothesis_language(self):
        schema = self.load_schema()
        claim = schema["properties"]["claim_boundary"]["properties"]

        self.assertEqual(claim["claim_status"]["const"], "forbidden_to_claim")
        allowed = set(claim["allowed_language"]["items"]["enum"])
        forbidden = set(claim["forbidden_claims"]["items"]["enum"])

        for phrase in [
            "symbolic",
            "speculative",
            "metaphorical",
            "hypothesis",
            "open_question",
            "needs_test",
        ]:
            self.assertIn(phrase, allowed)

        for phrase in [
            "claim_solved",
            "claim_proof",
            "claim_consciousness",
            "claim_AGI",
            "claim_sentience",
            "claim_empirical_validation",
        ]:
            self.assertIn(phrase, forbidden)

    def test_scientific_bridge_requires_null_model_and_failure_path(self):
        schema = self.load_schema()
        upgrade_items = set(
            schema["properties"]["claim_boundary"]["properties"]["upgrade_requires"]["items"]["enum"]
        )

        for required in [
            "definition",
            "null_model",
            "testable_prediction",
            "failure_condition",
            "experiment_manifest",
            "operator_review",
            "lab_record_separation",
        ]:
            self.assertIn(required, upgrade_items)

    def test_output_boundary_blocks_public_claims_and_actions(self):
        schema = self.load_schema()
        boundary = schema["properties"]["output_boundary"]["properties"]

        self.assertEqual(boundary["can_be_used_as_public_claim"]["const"], False)
        self.assertEqual(boundary["can_be_used_as_scientific_result"]["const"], False)
        self.assertEqual(boundary["can_upgrade_claims"]["const"], False)
        self.assertEqual(boundary["can_trigger_action"]["const"], False)
        self.assertEqual(boundary["can_request_experiment_manifest"]["const"], True)
        self.assertEqual(boundary["output_is_symbolic"]["const"], True)

    def test_safety_boundary_blocks_runtime_power(self):
        schema = self.load_schema()
        safety = schema["properties"]["safety"]["properties"]

        for key in [
            "network_allowed",
            "llm_calls_allowed",
            "external_contact_allowed",
            "secret_access_allowed",
            "runtime_write_implemented",
            "autonomous_action_allowed",
        ]:
            self.assertEqual(safety[key]["const"], False)

    def test_contract_separates_dream_chamber_from_lab_record(self):
        text = CONTRACT_PATH.read_text()

        for phrase in [
            "Forbidden to claim",
            "Allowed to imagine, define, question, design tests, preserve uncertainty, and revisit",
            "The Dream Chamber protects imagination",
            "The Lab Record protects truth",
            "Hallucination is not treated as truth",
            "raw symbolic material",
            "It may not become proof without a lab path",
            "Dreams are allowed",
            "Claims are not",
        ]:
            self.assertIn(phrase, text)

    def test_decision_record_contains_no_claim_or_runtime_authorization(self):
        text = DECISION_PATH.read_text()

        for phrase in [
            "runtime writer",
            "database migration",
            "SQLite table",
            "append command",
            "autonomous runtime",
            "claim upgrade",
            "consciousness claim",
            "AGI claim",
            "sentience claim",
            "proof claim",
        ]:
            self.assertIn(phrase, text)

        self.assertIn(
            "The Dream Chamber lets imagination branch without letting imagination impersonate evidence.",
            text,
        )


if __name__ == "__main__":
    unittest.main()
