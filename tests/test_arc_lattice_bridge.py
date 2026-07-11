import json
import unittest
from pathlib import Path

from arc_lattice_bridge import run_arc_bridge


class TestArcLatticeBridge(unittest.TestCase):
    def fixture(self):
        return json.loads(Path("arc_bridge/fixtures/task_4_arc_inputs.json").read_text())

    def test_fixture_locked(self):
        fixture = self.fixture()
        self.assertTrue(fixture["locked"])

    def test_arc_bridge_is_deterministic(self):
        fixture = self.fixture()
        a = run_arc_bridge(fixture)
        b = run_arc_bridge(fixture)
        self.assertEqual(a, b)

    def test_arc_bridge_passed(self):
        result = run_arc_bridge(self.fixture())
        self.assertEqual(result["summary"], "PASS")

    def test_expected_transforms(self):
        result = run_arc_bridge(self.fixture())
        by_id = {row["task_id"]: row for row in result["results"]}

        self.assertEqual(by_id["arc_identity_001"]["best_transform"]["name"], "identity")
        self.assertEqual(
            by_id["arc_flip_horizontal_001"]["best_transform"]["name"],
            "flip_horizontal",
        )
        self.assertEqual(
            by_id["arc_rotate180_001"]["best_transform"]["name"],
            "rotate180",
        )

    def test_guardrail_present(self):
        result = run_arc_bridge(self.fixture())
        for row in result["results"]:
            self.assertIn("does not prove", row["guardrail"])


if __name__ == "__main__":
    unittest.main()
