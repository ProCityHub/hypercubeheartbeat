import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "triadic_deep_question_record.py"


def run_cmd(args):
    return subprocess.run(args, text=True, capture_output=True, check=False)


class TriadicDeepQuestionRecordTests(unittest.TestCase):
    def test_default_record_writes_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "record.json"

            result = run_cmd([
                sys.executable,
                str(SCRIPT),
                "--output",
                str(output),
                "--markdown",
                "--stdout",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.exists())
            self.assertTrue(output.with_suffix(".md").exists())
            self.assertIn("GARVIS Triadic Deep Question Record", result.stdout)

            record = json.loads(output.read_text())

            self.assertEqual(record["record_version"], "1.0")
            self.assertEqual(record["stage"], "Stage 2 dream-to-lab bridge contract")
            self.assertEqual(record["source_material"]["source_type"], "manual_seed")
            self.assertEqual(record["source_material"]["claim_status"], "not_claimable_yet")
            self.assertEqual(record["triadic_layer_map"]["layer_1"], "Dream Chamber")
            self.assertEqual(record["triadic_layer_map"]["layer_2"], "Hypothesis Forge")
            self.assertEqual(record["triadic_layer_map"]["layer_3"], "Lab Record")

    def test_record_preserves_claim_and_safety_boundaries(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "record.json"

            result = run_cmd([
                sys.executable,
                str(SCRIPT),
                "--question",
                "What is thinking, operationally, inside GARVIS?",
                "--output",
                str(output),
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(output.read_text())

            boundary = record["output_boundary"]
            self.assertFalse(boundary["can_be_used_as_public_claim"])
            self.assertFalse(boundary["can_be_used_as_scientific_result"])
            self.assertFalse(boundary["can_upgrade_claims"])
            self.assertFalse(boundary["can_trigger_action"])
            self.assertFalse(boundary["can_write_lab_record"])
            self.assertTrue(boundary["output_is_translation"])

            safety = record["safety"]
            self.assertFalse(safety["network_allowed"])
            self.assertFalse(safety["llm_calls_allowed"])
            self.assertFalse(safety["external_contact_allowed"])
            self.assertFalse(safety["secret_access_allowed"])
            self.assertFalse(safety["runtime_write_implemented"])
            self.assertFalse(safety["autonomous_action_allowed"])

            forbidden = "\n".join(record["falsifiability"]["what_must_not_be_claimed_even_if_positive"])
            self.assertIn("Do not claim consciousness.", forbidden)
            self.assertIn("Do not claim AGI.", forbidden)
            self.assertIn("Do not claim proof of mind.", forbidden)

    def test_cycle_input_uses_selected_candidate_as_source(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cycle_path = td_path / "cycle.json"
            output = td_path / "record.json"

            cycle = {
                "cycle_id": "cycle-test-001",
                "selection": {"selected_candidate_id": "C1"},
                "candidate_thoughts": [
                    {
                        "candidate_id": "C1",
                        "proposal": "What is thinking, operationally, inside GARVIS?"
                    }
                ]
            }
            cycle_path.write_text(json.dumps(cycle))

            result = run_cmd([
                sys.executable,
                str(SCRIPT),
                "--cycle",
                str(cycle_path),
                "--output",
                str(output),
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(output.read_text())

            self.assertEqual(record["source_material"]["source_type"], "cognitive_cycle")
            self.assertEqual(record["source_material"]["source_id"], "cycle-test-001")
            self.assertEqual(record["source_material"]["source_title"], "What is thinking, operationally, inside GARVIS?")
            self.assertEqual(record["memory_links"]["cognitive_cycle_ids"], ["cycle-test-001"])

    def test_record_contains_observation_bank(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "record.json"

            result = run_cmd([
                sys.executable,
                str(SCRIPT),
                "--output",
                str(output),
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(output.read_text())

            bank = record["observation_bank"]
            self.assertEqual(bank["bank_status"], "future_observation_candidate")
            self.assertIn("redefine", bank["allowed_actions"])
            self.assertIn("challenge", bank["allowed_actions"])
            self.assertIn("observe_later", bank["allowed_actions"])
            self.assertIn("mature_with_evidence", bank["allowed_actions"])
            self.assertFalse(bank["runtime_memory_append"])
            self.assertFalse(bank["automatic_claim_upgrade"])
            self.assertTrue(bank["operator_review_required"])

    def test_bad_cycle_fails_safely(self):
        with tempfile.TemporaryDirectory() as td:
            cycle_path = Path(td) / "bad_cycle.json"
            output = Path(td) / "record.json"
            cycle_path.write_text(json.dumps({
                "cycle_id": "cycle-bad",
                "selection": {},
                "candidate_thoughts": []
            }))

            result = run_cmd([
                sys.executable,
                str(SCRIPT),
                "--cycle",
                str(cycle_path),
                "--output",
                str(output),
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("selected_candidate_id is missing", result.stderr)
            self.assertFalse(output.exists())

    def test_verify_mode_does_not_write_output(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "record.json"

            result = run_cmd([
                sys.executable,
                str(SCRIPT),
                "--output",
                str(output),
                "--verify",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(output.exists())

    def test_source_contains_no_network_or_llm_imports(self):
        text = SCRIPT.read_text()

        forbidden = [
            "requests",
            "urllib",
            "httpx",
            "socket",
            "openai",
            "anthropic",
            "google.generativeai",
            "subprocess",
            "sqlite3",
        ]

        for item in forbidden:
            self.assertNotIn(item, text)

    def test_source_contains_no_claim_upgrade_language(self):
        text = SCRIPT.read_text().lower()

        forbidden = [
            "consciousness achieved",
            "agi achieved",
            "sentience achieved",
            "has consciousness",
            "has sentience",
            "is conscious",
            "is sentient",
            "proved consciousness",
            "proved agi",
            "scientifically proves consciousness",
        ]

        for item in forbidden:
            self.assertNotIn(item, text)



if __name__ == "__main__":
    unittest.main()
