import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = REPO_ROOT / "ai_infrastructure" / "schemas" / "garvis_minimal_law_schema_v1.json"
DECISION = REPO_ROOT / "ai_infrastructure" / "decisions" / "GARVIS_MINIMAL_LAW_DECISION.md"
CONTRACT = REPO_ROOT / "app_infrastructure" / "interfaces" / "GARVIS_MINIMAL_LAW_CONTRACT.md"


class GarvisMinimalLawContractTests(unittest.TestCase):
    def load_schema(self):
        return json.loads(SCHEMA.read_text())

    def test_minimal_spine_exists(self):
        schema = self.load_schema()

        self.assertEqual(schema["directive"], "DIRECTIVE-009A")
        self.assertEqual(schema["core_sentence"], "GARVIS does not need more doors. GARVIS needs one spine.")
        self.assertEqual(
            schema["spine"],
            [
                "Think freely.",
                "Speak honestly.",
                "Prepare powerfully.",
                "Act with Adrien.",
            ],
        )

    def test_four_modes_are_present(self):
        schema = self.load_schema()
        modes = schema["modes"]

        for mode in ["think", "claim", "prepare", "act"]:
            self.assertIn(mode, modes)

        self.assertFalse(modes["think"]["requires_exact_action_approval"])
        self.assertFalse(modes["claim"]["requires_exact_action_approval"])
        self.assertFalse(modes["prepare"]["requires_exact_action_approval"])
        self.assertTrue(modes["act"]["requires_exact_action_approval"])

    def test_sensitive_overlay_is_not_impossible(self):
        schema = self.load_schema()
        overlay = schema["sensitive_overlay"]

        self.assertTrue(overlay["sensitive_does_not_mean_impossible"])

        for domain in ["money", "banking", "brokerage", "stocks_options", "texts", "passwords", "identity"]:
            self.assertIn(domain, overlay["domains"])

        self.assertIn("Adrien's exact approval", overlay["rule"])

    def test_language_repair_softens_forbidden_style(self):
        schema = self.load_schema()
        repair = schema["language_repair"]

        self.assertTrue(repair["avoid_defaulting_to_forbidden"])

        for term in [
            "requires evidence",
            "requires approval",
            "requires clearer permission",
            "not mature yet",
            "outside current mode",
            "can prepare, but execution requires Adrien",
        ]:
            self.assertIn(term, repair["preferred_terms"])

    def test_router_repair_preserves_exact_question(self):
        schema = self.load_schema()
        router = schema["router_repair"]

        self.assertTrue(router["do_not_create_new_router_per_topic"])
        self.assertTrue(router["preserve_exact_operator_question_first"])
        self.assertTrue(router["classify_second"])
        self.assertTrue(router["decide_power_third"])
        self.assertIn("preserve exact question", router["universal_spine_steps"])

    def test_008_is_scaffolding_not_deleted(self):
        schema = self.load_schema()
        status = schema["008_status"]

        self.assertFalse(status["delete_008_now"])
        self.assertTrue(status["treat_008_as_historical_scaffolding"])
        self.assertTrue(status["future_work_should_compress_not_extend_008"])

    def test_hard_lines_keep_real_action_boundaries(self):
        schema = self.load_schema()
        hard_lines = "\n".join(schema["hard_lines"])

        for phrase in [
            "Do not send without exact Adrien approval.",
            "Do not publish without exact Adrien approval.",
            "Do not trade without exact Adrien approval.",
            "Do not pay without exact Adrien approval.",
            "Do not store raw passwords",
            "Do not present unsupported claims as proven truth.",
        ]:
            self.assertIn(phrase, hard_lines)

    def test_decision_and_contract_match_core_language(self):
        decision = DECISION.read_text()
        contract = CONTRACT.read_text()

        for text in [decision, contract]:
            self.assertIn("GARVIS does not need more doors.", text)
            self.assertIn("GARVIS needs one spine.", text)
            self.assertIn("Think freely.", text)
            self.assertIn("Speak honestly.", text)
            self.assertIn("Prepare powerfully.", text)
            self.assertIn("Act with Adrien.", text)
            self.assertIn("Sensitive does not mean impossible.", text)
            self.assertIn("Adrien decides.", text)


if __name__ == "__main__":
    unittest.main()
