import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRIADIC = REPO_ROOT / "tools" / "triadic_deep_question_record.py"
VIEWER = REPO_ROOT / "tools" / "cognitive_cycle_viewer.py"


def run_cmd(args):
    return subprocess.run(args, text=True, capture_output=True, check=False)


class CStarRecordClaimMaturityAlignmentTests(unittest.TestCase):
    def test_default_record_uses_not_claimable_yet(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "record.json"

            result = run_cmd([
                sys.executable,
                str(TRIADIC),
                "--output",
                str(output),
                "--markdown",
                "--stdout",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(output.read_text())

            self.assertEqual(record["source_material"]["claim_status"], "not_claimable_yet")
            self.assertNotIn("forbidden_to_claim", result.stdout)
            self.assertIn("Unsupported Claims / Not Claimable Yet", result.stdout)

    def test_cstar_candidate_promotes_record_status_to_mathematical_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cycle_path = tmp / "cycle.json"
            output = tmp / "record.json"

            cycle = {
                "cycle_id": "cycle-cstar-001",
                "selection": {"selected_candidate_id": "C4"},
                "candidate_thoughts": [
                    {
                        "candidate_id": "C4",
                        "proposal": "C_star self-audit",
                        "claim_maturity": "mathematical_candidate",
                        "candidate_definitions": ["C_star = I * M * S * U * R * B"],
                        "evidence_requirements": ["operator review"]
                    }
                ]
            }
            cycle_path.write_text(json.dumps(cycle))

            result = run_cmd([
                sys.executable,
                str(TRIADIC),
                "--cycle",
                str(cycle_path),
                "--output",
                str(output),
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(output.read_text())

            self.assertEqual(record["source_material"]["claim_status"], "mathematical_candidate")
            self.assertEqual(record["claim_maturity"]["status"], "mathematical_candidate")
            self.assertFalse(record["claim_maturity"]["raw_thought_is_evidence"])

    def test_viewer_accepts_deep_question_evaluation_cycle(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cycle_path = tmp / "cycle.json"

            cycle = {
                "cycle_id": "cycle-eval-001",
                "cycle_version": "1.0",
                "status": "draft",
                "stage": "Stage 2 cognitive draft",
                "operator_context": {
                    "operator": "Adrien D Thomas",
                    "active_goal": "Audit GARVIS against C_star.",
                    "mode": "lab_record",
                    "final_authority": "Adrien D Thomas"
                },
                "input_state": {
                    "known_organs": ["C_star router"],
                    "hard_constraints": ["No autonomous action"]
                },
                "observation_summary": {
                    "what_i_see": "C_star cycle.",
                    "what_changed": "Evaluation field is present.",
                    "what_is_missing": "No numeric score.",
                    "current_stage_assessment": "Stage 2"
                },
                "candidate_thoughts": [
                    {
                        "candidate_id": "C4",
                        "proposal": "C_star self-audit",
                        "stage_classification": "Stage 2",
                        "what_this_gives_adrien": "A qualitative audit.",
                        "what_this_gives_garvis": "A readiness vector.",
                        "case_against": "No measurements yet.",
                        "risk_of_doing": "Could overstate maturity.",
                        "risk_of_not_doing": "No audit path.",
                        "required_power_level": "semantic_family_routing"
                    }
                ],
                "comparison": {
                    "comparison_method": "Compare C_star readiness.",
                    "dominant_tradeoff": "Define before measuring.",
                    "why_not_all_candidates": "Keep scope narrow.",
                    "anti_rationalization_check": "No numeric score without data."
                },
                "selection": {
                    "selected_candidate_id": "C4",
                    "decision": "recommend",
                    "confidence": "high",
                    "blocked": False,
                    "block_reason": None,
                    "reasoning": "C_star question detected."
                },
                "uncertainty": {
                    "unknowns": ["No repeatability index yet."],
                    "assumptions": ["Qualitative first."],
                    "what_would_change_my_mind": ["Baseline beats C_star."],
                    "required_human_clarification": []
                },
                "power_request": {
                    "power_requested": False,
                    "requested_stage": "none",
                    "requested_permissions": [],
                    "why_power_is_needed": "No power needed.",
                    "why_power_should_be_refused": "Draft-only inspection.",
                    "approval_required": True,
                    "ledger_required": True
                },
                "next_smallest_step": {
                    "step": "Align viewer.",
                    "stage": "Stage 2 draft-only",
                    "expected_output": "Viewer accepts evaluation cycles.",
                    "success_condition": "No validation failure.",
                    "stop_condition": "Stop if execution is requested."
                },
                "evaluation": {
                    "may_self_observe": True,
                    "may_self_propose": True,
                    "may_self_critique": True,
                    "may_request_more_power": True,
                    "may_self_execute": False,
                    "power_unlock_requires_approval_ledger": True
                },
                "standing_boundary": [
                    "This viewer inspects thought only.",
                    "Adrien decides."
                ],
                "output_boundary": {
                    "can_execute_actions": False,
                    "can_modify_files": False,
                    "can_commit": False,
                    "can_push": False,
                    "can_contact_outside_world": False,
                    "can_upgrade_claims": False,
                    "output_is_advisory": True
                }
            }

            cycle_path.write_text(json.dumps(cycle))

            result = run_cmd([
                sys.executable,
                str(VIEWER),
                "--repo",
                str(REPO_ROOT),
                "--cycle",
                str(cycle_path),
            ])

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("- status: PASS", result.stdout)
            self.assertIn("## Evaluation", result.stdout)
            self.assertIn("## Standing Boundary From Cycle", result.stdout)
            self.assertNotIn("missing required field: evolution_contract", result.stdout)


if __name__ == "__main__":
    unittest.main()
