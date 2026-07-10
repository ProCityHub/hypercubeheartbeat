import json
import unittest

from lattice_diagnostic_report import (
    REQUIRED_FIELDS,
    build_diagnostic_report,
)


class TestLatticeDiagnosticReport(unittest.TestCase):
    def test_report_is_deterministic(self):
        kwargs = dict(
            observer=0.7,
            actor=0.6,
            bridge=0.2,
            cube_corner="110",
            heartbeat_phase="observe",
            coherence_score=0.4,
            action="bounce",
            habit_signatures={"abc": 5},
            history_length=6,
            timing_coherence=0.2,
            information_coupling=0.7,
            integration_pressure=0.7,
        )

        a = build_diagnostic_report(**kwargs).to_json()
        b = build_diagnostic_report(**kwargs).to_json()

        self.assertEqual(a, b)

    def test_report_includes_required_fields(self):
        report = build_diagnostic_report(
            observer=0.5,
            actor=0.5,
            bridge=0.5,
            cube_corner="111",
            heartbeat_phase="act",
            coherence_score=0.8,
            action="settle",
        ).to_dict()

        for field in REQUIRED_FIELDS:
            self.assertIn(field, report)

    def test_missing_history_produces_warning(self):
        report = build_diagnostic_report(
            observer=0.8,
            actor=0.8,
            bridge=0.1,
            cube_corner="110",
            heartbeat_phase="bridge_pause",
            coherence_score=0.3,
            action="bounce",
            history_length=0,
        ).to_dict()

        self.assertEqual(report["limit_origin"], "insufficient_evidence")
        self.assertTrue(any("insufficient evidence" in w for w in report["warnings"]))

    def test_sparse_firing_is_not_automatic_bridge_failure(self):
        report = build_diagnostic_report(
            observer=0.9,
            actor=0.8,
            bridge=0.2,
            cube_corner="110",
            heartbeat_phase="bridge_pause",
            coherence_score=0.4,
            action="bounce",
            history_length=10,
            timing_coherence=0.1,
            information_coupling=0.8,
            integration_pressure=0.75,
        ).to_dict()

        self.assertEqual(report["limiting_constraint"], "bridge")
        self.assertEqual(report["limit_origin"], "formula_made_possible")
        self.assertTrue(
            any("do not treat this as automatic Bridge failure" in w for w in report["warnings"])
        )

    def test_json_is_valid(self):
        report = build_diagnostic_report(
            observer=0.4,
            actor=0.8,
            bridge=0.9,
            cube_corner="011",
            heartbeat_phase="act",
            coherence_score=0.6,
            action="settle",
            history_length=4,
        )

        parsed = json.loads(report.to_json())
        self.assertEqual(parsed["limiting_constraint"], "observer")


if __name__ == "__main__":
    unittest.main()
