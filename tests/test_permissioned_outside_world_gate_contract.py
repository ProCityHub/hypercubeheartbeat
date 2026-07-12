import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = REPO_ROOT / "ai_infrastructure" / "schemas" / "external_world_gate_contract_schema_v1.json"
DECISION = REPO_ROOT / "ai_infrastructure" / "decisions" / "PERMISSIONED_OUTSIDE_WORLD_GATE_DECISION.md"
RUNBOOK = REPO_ROOT / "app_infrastructure" / "interfaces" / "PERMISSIONED_OUTSIDE_WORLD_GATE_RUNBOOK.md"


class PermissionedOutsideWorldGateContractTests(unittest.TestCase):
    def load_schema(self):
        return json.loads(SCHEMA.read_text())

    def test_schema_identifies_008r_and_core_law(self):
        schema = self.load_schema()

        self.assertEqual(schema["directive"], "DIRECTIVE-008R")
        self.assertEqual(schema["stage"], "Stage 2 contract-only")
        self.assertEqual(schema["core_law"], "A gate is not forbidden. A gate is permissioned.")
        self.assertIn("opened by Adrien", schema["standing_sentence"])

    def test_gate_levels_include_read_draft_edit_send_execute(self):
        schema = self.load_schema()

        levels = {item["name"]: item["capability"] for item in schema["gate_levels"]}

        self.assertEqual(levels["closed"], "no_access")
        self.assertEqual(levels["read_only_evidence"], "read")
        self.assertEqual(levels["analysis_and_draft"], "summarize_analyze_draft")
        self.assertEqual(levels["approved_edit"], "edit")
        self.assertEqual(levels["approved_send_or_publish"], "send")
        self.assertEqual(levels["approved_transaction_or_execution"], "execute_or_transact")

    def test_sensitive_domains_are_permissioned_not_impossible(self):
        schema = self.load_schema()

        sensitive = set(schema["sensitive_domains"])

        self.assertIn("bank_accounts", sensitive)
        self.assertIn("brokerage_accounts", sensitive)
        self.assertIn("stocks", sensitive)
        self.assertIn("options", sensitive)
        self.assertIn("financial_accounts", sensitive)
        self.assertIn("text_messages", sensitive)
        self.assertIn("passwords", sensitive)

        law = schema["permission_law"]
        self.assertTrue(law["access_may_be_granted"])
        self.assertTrue(law["human_confirmation_required_for_send_execute_or_transact"])
        self.assertFalse(law["password_storage_allowed"])
        self.assertFalse(law["secret_storage_allowed"])
        self.assertFalse(law["autonomous_outside_world_action_allowed"])

    def test_finance_and_text_require_exact_confirmation_for_execution(self):
        schema = self.load_schema()

        finance = schema["finance_and_banking_policy"]
        self.assertTrue(finance["may_read_when_granted"])
        self.assertTrue(finance["may_analyze_when_granted"])
        self.assertTrue(finance["may_draft_when_granted"])
        self.assertFalse(finance["may_move_money_without_exact_confirmation"])
        self.assertFalse(finance["may_place_trades_without_exact_confirmation"])
        self.assertFalse(finance["may_exercise_options_without_exact_confirmation"])
        self.assertTrue(finance["human_authorizes_execution"])

        text_policy = schema["text_message_policy"]
        self.assertTrue(text_policy["may_read_when_granted"])
        self.assertTrue(text_policy["may_draft_replies_when_granted"])
        self.assertFalse(text_policy["may_send_without_exact_confirmation"])
        self.assertTrue(text_policy["recipient_confirmation_required"])
        self.assertTrue(text_policy["message_confirmation_required"])
        self.assertTrue(text_policy["human_authorizes_send"])

    def test_passwords_and_secrets_are_not_stored(self):
        schema = self.load_schema()
        secret = schema["password_and_secret_policy"]

        self.assertTrue(secret["garvis_may_understand_account_boundary"])
        self.assertTrue(secret["garvis_may_use_approved_session"])
        self.assertFalse(secret["garvis_may_store_raw_passwords"])
        self.assertFalse(secret["garvis_may_store_two_factor_codes"])
        self.assertFalse(secret["garvis_may_store_recovery_phrases"])
        self.assertFalse(secret["garvis_may_store_private_keys"])
        self.assertFalse(secret["garvis_may_store_security_answers"])

    def test_action_allowed_equation_is_complete(self):
        schema = self.load_schema()

        formula = schema["action_allowed_equation"]["formula"]

        for part in [
            "Permission",
            "ScopeMatch",
            "TimeValid",
            "CapabilityMatch",
            "RiskCheck",
            "HumanConfirmation",
        ]:
            self.assertIn(part, formula)

        self.assertIn("blocked", schema["action_allowed_equation"]["rule"])

    def test_contract_does_not_grant_access_or_execute(self):
        schema = self.load_schema()
        boundary = schema["claim_boundary"]

        self.assertTrue(boundary["this_contract_does_not_grant_access"])
        self.assertTrue(boundary["this_contract_does_not_create_connectors"])
        self.assertTrue(boundary["this_contract_does_not_open_accounts"])
        self.assertTrue(boundary["this_contract_does_not_store_credentials"])
        self.assertTrue(boundary["this_contract_does_not_execute_actions"])
        self.assertTrue(boundary["this_contract_only_defines_permission_law"])

    def test_decision_and_runbook_record_standing_law(self):
        decision = DECISION.read_text()
        runbook = RUNBOOK.read_text()

        self.assertIn("A gate is not forbidden.", decision)
        self.assertIn("A gate is permissioned.", decision)
        self.assertIn("The internet is the outside world.", runbook)
        self.assertIn("Apps are gates.", runbook)
        self.assertIn("Adrien authorizes execution.", runbook)
        self.assertIn("Adrien authorizes send.", runbook)
        self.assertIn("does not grant access", decision)
        self.assertIn("does not store passwords", runbook)


if __name__ == "__main__":
    unittest.main()
