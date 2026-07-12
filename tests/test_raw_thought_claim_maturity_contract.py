import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "ai_infrastructure" / "schemas" / "claim_maturity_record_schema_v1.json"
CONTRACT_PATH = REPO_ROOT / "app_infrastructure" / "interfaces" / "RAW_THOUGHT_CLAIM_MATURITY_CONTRACT.md"
DECISION_PATH = REPO_ROOT / "ai_infrastructure" / "decisions" / "RAW_THOUGHT_CLAIM_MATURITY_DECISION.md"


class RawThoughtClaimMaturityContractTests(unittest.TestCase):
    def test_schema_has_core_law(self):
        schema = json.loads(SCHEMA_PATH.read_text())

        self.assertEqual(schema["properties"]["directive"]["const"], "DIRECTIVE-008K")
        self.assertEqual(
            schema["properties"]["core_law"]["const"],
            "Nothing is forbidden to think. Only unsupported claims are forbidden to present as truth.",
        )

    def test_schema_defines_claim_maturity_ladder(self):
        schema = json.loads(SCHEMA_PATH.read_text())
        ladder = schema["properties"]["maturity_ladder"]["items"]["enum"]

        expected = [
            "raw",
            "unproven",
            "defined",
            "mathematical_candidate",
            "mathematically_derived_within_assumptions",
            "testable",
            "empirically_supported",
            "validated_within_scope",
            "rejected",
        ]

        self.assertEqual(ladder, expected)

    def test_schema_allows_raw_thought_but_not_raw_truth_claims(self):
        schema = json.loads(SCHEMA_PATH.read_text())

        raw_layer = schema["properties"]["raw_layer"]["properties"]
        claim_layer = schema["properties"]["claim_layer"]["properties"]
        safety = schema["properties"]["safety_boundary"]["properties"]

        self.assertTrue(raw_layer["may_hold_any_thought"]["const"])
        self.assertFalse(raw_layer["raw_thought_is_evidence"]["const"])

        self.assertFalse(claim_layer["unsupported_truth_claim_allowed"]["const"])
        self.assertTrue(claim_layer["external_claim_requires_evidence"]["const"])
        self.assertTrue(claim_layer["public_truth_requires_scope"]["const"])

        self.assertFalse(safety["can_upgrade_claim_without_evidence"]["const"])
        self.assertFalse(safety["can_present_raw_thought_as_truth"]["const"])

    def test_contract_corrects_forbidden_language(self):
        text = CONTRACT_PATH.read_text()

        self.assertIn("Nothing is forbidden to think. Only unsupported claims are forbidden to present as truth.", text)
        self.assertIn("The Raw Layer is the womb layer.", text)
        self.assertIn("The boundary is not placed on thought.", text)
        self.assertIn("The boundary is placed on unsupported truth claims.", text)
        self.assertIn("not claimable yet", text)

    def test_contract_allows_deep_topics_as_raw_thought(self):
        text = CONTRACT_PATH.read_text()

        for phrase in [
            "GARVIS may think about AGI.",
            "GARVIS may think about consciousness.",
            "GARVIS may define candidate AGI mathematically.",
            "GARVIS may define candidate consciousness mathematically.",
            "GARVIS may produce internal raw records.",
        ]:
            self.assertIn(phrase, text)

    def test_decision_preserves_runtime_boundaries(self):
        text = DECISION_PATH.read_text()

        for phrase in [
            "No runtime memory append.",
            "No experiment execution.",
            "No lab result.",
            "No network call.",
            "No LLM call.",
            "No autonomous action.",
            "No claim upgrade without evidence.",
            "No raw thought presented as truth.",
        ]:
            self.assertIn(phrase, text)

    def test_contract_preserves_raw_thought_freedom(self):
        text = CONTRACT_PATH.read_text().lower()

        self.assertIn(
            "nothing is forbidden to think. only unsupported claims are forbidden to present as truth.",
            text,
        )

        banned_meanings = [
            "raw thought is forbidden",
            "all thought is forbidden",
            "consciousness questions are forbidden",
            "agi questions are forbidden",
            "imagination is forbidden",
            "must not think about consciousness",
            "must not think about agi",
            "must not think about imagination",
        ]

        for phrase in banned_meanings:
            self.assertNotIn(phrase, text)



if __name__ == "__main__":
    unittest.main()
