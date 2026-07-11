import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "ll07_analyze_ratings.py"
spec = importlib.util.spec_from_file_location("ll07_analyze_ratings", SCRIPT_PATH)
ll07 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ll07)


class TestLL07WordCoordinateAnalysis(unittest.TestCase):
    def test_kendalls_w_identical_rankings(self):
        matrix = [
            [10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0],
            [30.0, 30.0, 30.0],
            [40.0, 40.0, 40.0],
        ]
        self.assertAlmostEqual(ll07.kendalls_w(matrix), 1.0)

    def test_rank_values_ties(self):
        self.assertEqual(ll07.rank_values([5.0, 5.0, 9.0]), [1.5, 1.5, 3.0])

    def test_attention_checks_filter_bad_rater(self):
        rows = [
            {"rater_id": "good", "item_id": "ATTN_O", "O": "100", "A": "0", "B": "0"},
            {"rater_id": "good", "item_id": "ATTN_A", "O": "0", "A": "100", "B": "0"},
            {"rater_id": "good", "item_id": "ATTN_B", "O": "0", "A": "0", "B": "100"},
            {"rater_id": "bad", "item_id": "ATTN_O", "O": "10", "A": "0", "B": "0"},
            {"rater_id": "bad", "item_id": "ATTN_A", "O": "0", "A": "100", "B": "0"},
            {"rater_id": "bad", "item_id": "ATTN_B", "O": "0", "A": "0", "B": "100"},
        ]
        self.assertEqual(ll07.valid_raters(rows), ["good"])

    def test_analyze_synthetic_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stimuli_path = tmp_path / "stimuli.csv"
            ratings_path = tmp_path / "ratings.csv"

            with stimuli_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["item_id", "stratum", "word"])
                writer.writeheader()
                for i in range(100):
                    writer.writerow({
                        "item_id": f"W{i:03d}",
                        "stratum": "test",
                        "word": f"word{i:03d}",
                    })

            with ratings_path.open("w", newline="", encoding="utf-8") as handle:
                fieldnames = ["rater_id", "item_id", "word", "O", "A", "B"]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()

                for rater in range(12):
                    rater_id = f"R{rater:02d}"
                    writer.writerow({"rater_id": rater_id, "item_id": "ATTN_O", "word": "check", "O": 100, "A": 0, "B": 0})
                    writer.writerow({"rater_id": rater_id, "item_id": "ATTN_A", "word": "check", "O": 0, "A": 100, "B": 0})
                    writer.writerow({"rater_id": rater_id, "item_id": "ATTN_B", "word": "check", "O": 0, "A": 0, "B": 100})
                    for i in range(100):
                        writer.writerow({
                            "rater_id": rater_id,
                            "item_id": f"W{i:03d}",
                            "word": f"word{i:03d}",
                            "O": i,
                            "A": 100 - i,
                            "B": i,
                        })

            result = ll07.analyze(stimuli_path, ratings_path, permutations=19, seed=1)

            self.assertEqual(result["n_valid_raters"], 12)
            self.assertEqual(result["n_words"], 100)
            self.assertGreaterEqual(result["dimensions"]["O"]["kendalls_w"], 0.99)
            self.assertGreaterEqual(result["dimensions"]["A"]["kendalls_w"], 0.99)
            self.assertGreaterEqual(result["dimensions"]["B"]["kendalls_w"], 0.99)


if __name__ == "__main__":
    unittest.main()
