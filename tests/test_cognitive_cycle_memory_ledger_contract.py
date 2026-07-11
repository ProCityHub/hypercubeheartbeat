import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "ai_infrastructure" / "schemas" / "cognitive_cycle_memory_ledger_record_schema_v1.json"
CONTRACT_PATH = REPO_ROOT / "app_infrastructure" / "interfaces" / "COGNITIVE_CYCLE_MEMORY_LEDGER_CONTRACT.md"
DECISION_PATH = REPO_ROOT / "ai_infrastructure" / "decisions" / "COGNITIVE_CYCLE_MEMORY_LEDGER_CONTRACT_DECISION.md"


class CognitiveCycleMemoryLedgerContractTests(unittest.TestCase):
    def load_schema(self):
        return json.loads(SCHEMA_PATH.read_text())

    def test_schema_loads_as_json(self):
        schema = self.load_schema()
        self.assertEqual(schema["title"], "GARVIS Cognitive Cycle Memory Ledger Record Schema v1")
        self.assertEqual(schema["properties"]["record_version"]["const"], "1.0")
        self.assertEqual(schema["properties"]["stage"]["const"], "Stage 2 memory contract")

    def test_schema_requires_memory_identity_chain_and_review(self):
        schema = self.load_schema()
        required = set(schema["required"])

        for field in [
            "storage_scope",
            "cycle_identity",
            "cycle_artifacts",
            "cycle_chain",
            "selection_summary",
            "power_request_summary",
            "operator_review",
            "audit_boundary",
            "safety",
        ]:
            self.assertIn(field, required)

    def test_storage_scope_is_contract_only_and_append_only(self):
        schema = self.load_schema()
        storage = schema["properties"]["storage_scope"]["properties"]

        self.assertEqual(storage["implementation_status"]["const"], "not_implemented")
        self.assertEqual(storage["local_only"]["const"], True)
        self.assertEqual(storage["append_only_required"]["const"], True)
        self.assertEqual(storage["raw_cycle_json_default_committed"]["const"], False)

    def test_cycle_chain_requires_previous_and_next_links(self):
        schema = self.load_schema()
        chain_required = set(schema["properties"]["cycle_chain"]["required"])

        for field in [
            "previous_cycle_id",
            "next_cycle_id",
            "parent_record_id",
            "supersedes_record_id",
            "chain_integrity_required",
        ]:
            self.assertIn(field, chain_required)

        self.assertEqual(
            schema["properties"]["cycle_chain"]["properties"]["chain_integrity_required"]["const"],
            True,
        )

    def test_power_request_does_not_grant_permission(self):
        schema = self.load_schema()
        power = schema["properties"]["power_request_summary"]["properties"]

        self.assertEqual(power["approval_required"]["const"], True)
        self.assertEqual(power["ledger_required"]["const"], True)
        self.assertEqual(power["permission_granted_by_this_record"]["const"], False)

    def test_audit_boundary_blocks_claim_and_action_confusion(self):
        schema = self.load_schema()
        audit = schema["properties"]["audit_boundary"]["properties"]

        for key in [
            "record_is_memory_not_action",
            "record_is_not_approval",
            "record_is_not_empirical_result",
            "record_is_not_claim_upgrade",
            "future_implementation_requires_new_directive",
        ]:
            self.assertEqual(audit[key]["const"], True)

    def test_safety_boundary_blocks_runtime_power(self):
        schema = self.load_schema()
        safety = schema["properties"]["safety"]["properties"]

        for key in [
            "network_allowed",
            "llm_calls_allowed",
            "external_contact_allowed",
            "secret_access_allowed",
            "raw_sensor_payload_access_allowed",
            "runtime_write_implemented",
        ]:
            self.assertEqual(safety[key]["const"], False)

    def test_contract_states_no_runtime_implementation(self):
        text = CONTRACT_PATH.read_text()

        for phrase in [
            "not a memory database implementation",
            "not a runtime writer",
            "Append-only requirement",
            "Local-only requirement",
            "Raw cognitive cycle JSON and Markdown artifacts are local drafts by default",
            "A future memory record may preserve a power request",
            "It must not grant that power",
            "This contract creates no database",
            "This contract creates no table",
            "Memory is not action",
        ]:
            self.assertIn(phrase, text)

    def test_decision_record_contains_no_runtime_authorization(self):
        text = DECISION_PATH.read_text()

        for forbidden_scope in [
            "memory database",
            "SQLite table",
            "runtime memory writer",
            "background service",
            "autonomous runtime",
            "action engine",
            "autonomous action",
            "claim upgrade",
        ]:
            self.assertIn(forbidden_scope, text)

        self.assertIn("It does not implement memory writes.", text)


if __name__ == "__main__":
    unittest.main()
