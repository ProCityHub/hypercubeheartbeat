import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "ai_infrastructure" / "schemas" / "dream_to_lab_bridge_record_schema_v1.json"
CONTRACT_PATH = REPO_ROOT / "app_infrastructure" / "interfaces" / "DREAM_TO_LAB_BRIDGE_CONTRACT.md"
DECISION_PATH = REPO_ROOT / "ai_infrastructure" / "decisions" / "DREAM_TO_LAB_BRIDGE_CONTRACT_DECISION.md"


class DreamToLabBridgeContractTests(unittest.TestCase):
    def load_schema(self):
        return json.loads(SCHEMA_PATH.read_text())

    def test_schema_loads_as_json(self):
        schema = self.load_schema()
        self.assertEqual(schema["title"], "GARVIS Dream-to-Lab Bridge Record Schema v1")
        self.assertEqual(schema["properties"]["record_version"]["const"], "1.0")
        self.assertEqual(schema["properties"]["stage"]["const"], "Stage 2 dream-to-lab bridge contract")

    def test_schema_requires_all_bridge_sections(self):
        schema = self.load_schema()
        required = set(schema["required"])

        for field in [
            "triadic_layer_map",
            "source_material",
            "translation_workbench",
            "definition_workbench",
            "variable_extraction",
            "hypothesis_candidates",
            "null_model_design",
            "measurement_plan",
            "falsifiability",
            "manifest_gate",
            "memory_links",
            "operator_review",
            "output_boundary",
            "safety",
        ]:
            self.assertIn(field, required)

    def test_triadic_layer_map_defines_three_rooms(self):
        schema = self.load_schema()
        triad = schema["properties"]["triadic_layer_map"]["properties"]

        self.assertEqual(triad["layer_1"]["const"], "Dream Chamber")
        self.assertEqual(triad["layer_2"]["const"], "Hypothesis Forge")
        self.assertEqual(triad["layer_3"]["const"], "Lab Record")
        self.assertIn("testable structure", triad["convergence_zone"]["const"])

    def test_bridge_source_material_cannot_claim(self):
        schema = self.load_schema()
        source = schema["properties"]["source_material"]["properties"]

        self.assertEqual(source["claim_status"]["const"], "forbidden_to_claim")
        self.assertIn("grok_report", source["source_type"]["enum"])
        self.assertIn("dream_chamber_record", source["source_type"]["enum"])

    def test_hypothesis_candidates_remain_exploratory(self):
        schema = self.load_schema()
        candidate = schema["properties"]["hypothesis_candidates"]["items"]["properties"]

        self.assertEqual(
            candidate["claim_boundary"]["const"],
            "exploratory_only_until_experiment_manifest_and_result",
        )
        self.assertIn("prediction", candidate)
        self.assertIn("counter_prediction", candidate)

    def test_null_model_and_manifest_gate_are_required(self):
        schema = self.load_schema()

        null_model = schema["properties"]["null_model_design"]["properties"]
        self.assertEqual(null_model["null_model_required"]["const"], True)

        gate = schema["properties"]["manifest_gate"]["properties"]
        self.assertEqual(gate["can_request_experiment_manifest"]["const"], True)
        self.assertEqual(gate["manifest_required_before_test"]["const"], True)
        self.assertEqual(gate["pre_registration_required"]["const"], True)
        self.assertEqual(gate["bridge_can_emit_claim"]["const"], False)

    def test_output_boundary_blocks_claims_results_and_actions(self):
        schema = self.load_schema()
        boundary = schema["properties"]["output_boundary"]["properties"]

        self.assertEqual(boundary["can_be_used_as_public_claim"]["const"], False)
        self.assertEqual(boundary["can_be_used_as_scientific_result"]["const"], False)
        self.assertEqual(boundary["can_upgrade_claims"]["const"], False)
        self.assertEqual(boundary["can_trigger_action"]["const"], False)
        self.assertEqual(boundary["can_write_lab_record"]["const"], False)
        self.assertEqual(boundary["output_is_translation"]["const"], True)

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

    def test_contract_describes_three_layer_architecture(self):
        text = CONTRACT_PATH.read_text()

        for phrase in [
            "The Dream Chamber protects imagination",
            "The Lab Record protects truth",
            "The Bridge translates imagination into testable structure",
            "The center is translation",
            "Grok symbolic QASM reports belong first in Dream Chamber",
            "Small QASM circuits may enter the Bridge as hypothesis seeds",
            "Only preregistered, tested experiments may enter the Lab Record",
            "Dreams are allowed",
            "Translation is required",
            "Evidence decides",
        ]:
            self.assertIn(phrase, text)

    def test_decision_record_contains_no_claim_or_runtime_authorization(self):
        text = DECISION_PATH.read_text()

        for phrase in [
            "runtime writer",
            "database migration",
            "SQLite table",
            "append command",
            "experiment execution",
            "empirical result",
            "claim upgrade",
            "consciousness claim",
            "AGI claim",
            "sentience claim",
            "proof claim",
        ]:
            self.assertIn(phrase, text)

        self.assertIn(
            "The Bridge is where imagination becomes testable without pretending it is already true.",
            text,
        )


if __name__ == "__main__":
    unittest.main()
