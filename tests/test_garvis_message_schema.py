import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "ai_infrastructure" / "schemas" / "garvis_message_schema_v1.json"


class TestGarvisMessageSchema(unittest.TestCase):
    def setUp(self):
        self.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    def test_schema_loads(self):
        self.assertEqual(self.schema["title"], "GARVIS Message Schema v1")
        self.assertEqual(self.schema["type"], "object")
        self.assertFalse(self.schema["additionalProperties"])

    def test_required_fields_present(self):
        required = set(self.schema["required"])
        expected = {
            "schema_version",
            "message_id",
            "message_type",
            "created_utc",
            "created_by",
            "summary",
            "allowed_actions",
            "forbidden_actions",
            "human_approval_required",
            "claim_boundary",
            "evidence_boundary",
            "stop_conditions",
        }
        self.assertEqual(required, expected)

    def test_message_types_are_protocol_types(self):
        enum = set(self.schema["properties"]["message_type"]["enum"])
        self.assertEqual(
            enum,
            {
                "status",
                "intake",
                "manifest",
                "proposal",
                "decision_request",
                "action_directive",
                "report",
                "refusal",
            },
        )

    def test_claim_and_evidence_boundaries_exist(self):
        claim_values = set(self.schema["properties"]["claim_boundary"]["enum"])
        evidence_values = set(self.schema["properties"]["evidence_boundary"]["enum"])

        self.assertIn("no_claim_change", claim_values)
        self.assertIn("unapproved_claim_language_stop", claim_values)
        self.assertIn("not_evidence", evidence_values)
        self.assertIn("sandbox_only", evidence_values)
        self.assertIn("registered_analysis_required", evidence_values)

    def test_no_reserved_empirical_verdict_labels_in_schema(self):
        text = SCHEMA_PATH.read_text(encoding="utf-8")
        self.assertNotIn("SUPPORTED", text)
        self.assertNotIn("NOT_SUPPORTED", text)


if __name__ == "__main__":
    unittest.main()
