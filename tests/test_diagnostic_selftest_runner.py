import unittest

from diagnostic_selftest_runner import load_fixture, run_pilot, stable_hash


class TestFrozenPilotRunner(unittest.TestCase):
    def test_fixture_is_locked(self):
        fixture = load_fixture()
        self.assertTrue(fixture["locked"])

    def test_pilot_is_deterministic(self):
        fixture = load_fixture()
        a = run_pilot(fixture)
        b = run_pilot(fixture)
        self.assertEqual(stable_hash(a), stable_hash(b))

    def test_pilot_passed(self):
        fixture = load_fixture()
        result = run_pilot(fixture)
        self.assertEqual(result["summary"], "PASS")
        self.assertTrue(result["criteria"]["passed"])
        self.assertEqual(result["criteria"]["failures"], [])

    def test_expected_limit_origins(self):
        fixture = load_fixture()
        result = run_pilot(fixture)
        by_id = {row["input_id"]: row["report"] for row in result["reports"]}

        self.assertEqual(
            by_id["formula_ceiling_001"]["limit_origin"],
            "formula_made_possible",
        )
        self.assertEqual(
            by_id["substrate_bridge_failure_001"]["limit_origin"],
            "substrate_made_possible",
        )
        self.assertEqual(
            by_id["insufficient_history_001"]["limit_origin"],
            "insufficient_evidence",
        )
        self.assertEqual(
            by_id["observer_limited_001"]["limit_origin"],
            "not_bridge_limited",
        )


if __name__ == "__main__":
    unittest.main()
