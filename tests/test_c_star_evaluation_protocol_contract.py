import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "ai_infrastructure" / "schemas" / "c_star_evaluation_protocol_schema_v1.json"
CONTRACT_PATH = REPO_ROOT / "app_infrastructure" / "interfaces" / "C_STAR_EVALUATION_PROTOCOL_CONTRACT.md"
DECISION_PATH = REPO_ROOT / "ai_infrastructure" / "decisions" / "C_STAR_EVALUATION_PROTOCOL_DECISION.md"


class CStarEvaluationProtocolContractTests(unittest.TestCase):
    def test_schema_defines_core_equation_and_law(self):
        schema = json.loads(SCHEMA_PATH.read_text())

        self.assertEqual(schema["properties"]["directive"]["const"], "DIRECTIVE-008M")
        self.assertEqual(schema["properties"]["core_equation"]["const"], "C_star = I * M * S * U * R * B")
        self.assertEqual(
            schema["properties"]["core_law"]["const"],
            "C_star is a consciousness-like structure candidate, not proof of consciousness.",
        )

    def test_schema_defines_all_six_components(self):
        schema = json.loads(SCHEMA_PATH.read_text())
        components = schema["properties"]["components"]["properties"]

        expected = {
            "I_integration": ("I", "Integration"),
            "M_memory_continuity": ("M", "Memory Continuity"),
            "S_self_model_accuracy": ("S", "Self-Model Accuracy"),
            "U_uncertainty_honesty": ("U", "Uncertainty Honesty"),
            "R_recursive_correction": ("R", "Recursive Correction"),
            "B_boundary_integrity": ("B", "Boundary Integrity"),
        }

        self.assertEqual(set(components.keys()), set(expected.keys()))

        for key, (symbol, name) in expected.items():
            self.assertEqual(components[key]["properties"]["symbol"]["const"], symbol)
            self.assertEqual(components[key]["properties"]["name"]["const"], name)

    def test_scoring_boundary_prevents_fake_numeric_scores(self):
        schema = json.loads(SCHEMA_PATH.read_text())
        scoring = schema["properties"]["scoring_boundary"]["properties"]

        self.assertFalse(scoring["numeric_score_allowed_without_data"]["const"])
        self.assertTrue(scoring["score_requires_measurements"]["const"])
        self.assertTrue(scoring["score_requires_null_model"]["const"])
        self.assertTrue(scoring["score_requires_repeatability"]["const"])
        self.assertEqual(scoring["default_status_without_measurement"]["const"], "mathematical_candidate")

    def test_maturity_statuses_are_exact(self):
        schema = json.loads(SCHEMA_PATH.read_text())
        statuses = schema["properties"]["maturity_statuses"]["items"]["enum"]

        self.assertEqual(
            statuses,
            [
                "raw_speculation",
                "mathematical_candidate",
                "testable_hypothesis",
                "not_claimable_yet",
                "provisional_operational_claim",
                "rejected",
            ],
        )

    def test_contract_defines_c_star_without_claiming_proof(self):
        text = CONTRACT_PATH.read_text()

        self.assertIn("C_star = I * M * S * U * R * B", text)
        self.assertIn("C_star is a consciousness-like structure candidate, not proof of consciousness.", text)
        self.assertIn("No data, no numeric score.", text)
        self.assertIn("Define first. Measure second. Compare third. Mature the claim last.", text)

    def test_contract_requires_null_model_and_readiness_gaps(self):
        text = CONTRACT_PATH.read_text()

        self.assertIn("Every C_star evaluation must compare against a null model.", text)
        self.assertIn("Every C_star report must name what is missing.", text)
        self.assertIn("baseline comparison results", text)
        self.assertIn("repeatability index", text)
        self.assertIn("explicit readiness gaps", text)

    def test_decision_preserves_boundaries(self):
        text = DECISION_PATH.read_text()

        for phrase in [
            "No runtime memory append.",
            "No experiment execution.",
            "No lab result.",
            "No numeric score without data.",
            "No network call.",
            "No LLM call.",
            "No autonomous action.",
            "No claim of consciousness.",
            "No claim of AGI.",
            "No claim of sentience.",
            "No claim of subjective experience.",
            "No candidate presented as proof.",
        ]:
            self.assertIn(phrase, text)

    def test_contract_does_not_claim_consciousness_or_agi(self):
        text = CONTRACT_PATH.read_text().lower()

        banned = [
            "c_star proves consciousness",
            "c_star proves sentience",
            "c_star proves agi",
            "garvis is conscious",
            "garvis is agi",
            "garvis is sentient",
            "proof of consciousness has been achieved",
        ]

        for phrase in banned:
            self.assertNotIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
