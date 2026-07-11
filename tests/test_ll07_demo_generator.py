import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

GENERATOR_PATH = ROOT / "tools" / "ll07_generate_demo_ratings.py"
ANALYZER_PATH = ROOT / "tools" / "ll07_analyze_ratings.py"
STIMULI_PATH = ROOT / "data" / "ll07_word_stimuli.csv"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


generator = load_module("ll07_generate_demo_ratings", GENERATOR_PATH)
analyzer = load_module("ll07_analyze_ratings_for_demo_test", ANALYZER_PATH)


class TestLL07DemoGenerator(unittest.TestCase):
    def test_generate_demo_rows_count(self):
        stimuli = generator.read_stimuli(STIMULI_PATH)
        rows = generator.generate_rows(stimuli, n_raters=12)

        self.assertEqual(len(stimuli), 100)
        self.assertEqual(len(rows), 12 * 103)

        attention_rows = [row for row in rows if str(row["item_id"]).startswith("ATTN_")]
        self.assertEqual(len(attention_rows), 36)

    def test_generated_file_analyzes_as_pipeline_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "demo_ratings.csv"

            stimuli = generator.read_stimuli(STIMULI_PATH)
            rows = generator.generate_rows(stimuli, n_raters=12)
            generator.write_rows(output_path, rows)

            with output_path.open(newline="", encoding="utf-8") as handle:
                file_rows = list(csv.DictReader(handle))

            self.assertEqual(len(file_rows), 12 * 103)

            result = analyzer.analyze(
                STIMULI_PATH,
                output_path,
                permutations=9,
                seed=7707,
            )

            self.assertEqual(result["status"], "ll07_single_run_output")
            self.assertEqual(result["n_valid_raters"], 12)
            self.assertEqual(result["n_words"], 100)
            self.assertIn("criterion_met", result)

    def test_score_bounds(self):
        for rater_index in range(12):
            for item_index in range(100):
                for dimension in ("O", "A", "B"):
                    value = generator.score_for(item_index, rater_index, dimension)
                    self.assertGreaterEqual(value, 0)
                    self.assertLessEqual(value, 100)


if __name__ == "__main__":
    unittest.main()
