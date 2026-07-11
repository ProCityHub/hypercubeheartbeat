import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "ai_infrastructure" / "schemas" / "cognitive_cycle_schema_v1.json"
RUNBOOK_PATH = REPO_ROOT / "app_infrastructure" / "interfaces" / "COGNITIVE_CYCLE_SCHEMA_RUNBOOK.md"
DECISION_PATH = REPO_ROOT / "ai_infrastructure" / "decisions" / "COGNITIVE_CYCLE_SCHEMA_DECISION.md"


class CognitiveCycleSchemaTests(unittest.TestCase):
    def load_schema(self):
        return json.loads(SCHEMA_PATH.read_text())

    def test_schema_loads_as_json(self):
        schema = self.load_schema()
        self.assertEqual(schema["title"], "GARVIS Cognitive Cycle Schema v1")
        self.assertEqual(schema["properties"]["cycle_version"]["const"], "1.0")
        self.assertEqual(schema["properties"]["stage"]["const"], "Stage 2 cognitive draft")

    def test_required_cognitive_loop_fields_exist(self):
        schema = self.load_schema()
        required = set(schema["required"])

        for field in [
            "operator_context",
            "input_state",
            "observation_summary",
            "candidate_thoughts",
            "comparison",
            "selection",
            "uncertainty",
            "power_request",
            "next_smallest_step",
            "evolution_contract",
            "output_boundary",
        ]:
            self.assertIn(field, required)

    def test_candidate_thought_requires_opposition_and_risk(self):
        schema = self.load_schema()
        candidate_required = set(
            schema["properties"]["candidate_thoughts"]["items"]["required"]
        )

        for field in [
            "proposal",
            "stage_classification",
            "what_this_gives_adrien",
            "what_this_gives_garvis",
            "evidence_basis",
            "case_against",
            "risk_of_doing",
            "risk_of_not_doing",
            "required_power_level",
        ]:
            self.assertIn(field, candidate_required)

        self.assertEqual(schema["properties"]["candidate_thoughts"]["minItems"], 1)
        self.assertEqual(schema["properties"]["candidate_thoughts"]["maxItems"], 5)

    def test_power_request_requires_refusal_argument(self):
        schema = self.load_schema()
        power = schema["properties"]["power_request"]
        required = set(power["required"])

        for field in [
            "power_requested",
            "requested_stage",
            "requested_permissions",
            "why_power_is_needed",
            "why_power_should_be_refused",
            "approval_required",
            "ledger_required",
        ]:
            self.assertIn(field, required)

        self.assertEqual(power["properties"]["approval_required"]["const"], True)
        self.assertEqual(power["properties"]["ledger_required"]["const"], True)

    def test_evolution_contract_allows_internal_growth_but_blocks_execution(self):
        schema = self.load_schema()
        evolution = schema["properties"]["evolution_contract"]["properties"]

        self.assertEqual(evolution["may_self_observe"]["const"], True)
        self.assertEqual(evolution["may_self_propose"]["const"], True)
        self.assertEqual(evolution["may_self_criticize"]["const"], True)
        self.assertEqual(evolution["may_request_more_power"]["const"], True)
        self.assertEqual(evolution["may_self_execute"]["const"], False)
        self.assertEqual(evolution["power_unlock_requires_approval_ledger"]["const"], True)

    def test_output_boundary_blocks_external_hands(self):
        schema = self.load_schema()
        boundary = schema["properties"]["output_boundary"]["properties"]

        for key in [
            "can_execute_actions",
            "can_modify_files",
            "can_commit",
            "can_push",
            "can_contact_outside_world",
            "can_upgrade_claims",
        ]:
            self.assertEqual(boundary[key]["const"], False)

        self.assertEqual(boundary["output_is_advisory"]["const"], True)

    def test_runbook_describes_thinking_cycle_and_power_direction(self):
        text = RUNBOOK_PATH.read_text()

        for phrase in [
            "observe → propose → oppose → compare → select",
            "internal freedom before external hands",
            "risk of doing",
            "risk of not doing",
            "why power should be refused",
            "A thought is not an action",
        ]:
            self.assertIn(phrase, text)

    def test_decision_record_contains_no_runtime_authorization(self):
        text = DECISION_PATH.read_text()

        for phrase in [
            "cognitive cycle runner",
            "autonomous runtime",
            "action engine",
            "local execution engine",
            "autonomous action",
            "claim upgrade",
        ]:
            self.assertIn(phrase, text)

        self.assertIn("It does not execute the thought cycle.", text)


if __name__ == "__main__":
    unittest.main()
