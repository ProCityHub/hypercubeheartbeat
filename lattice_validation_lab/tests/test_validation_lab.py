import json, subprocess, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class ValidationLabTests(unittest.TestCase):
    def test_source_archive_present(self):
        self.assertTrue((ROOT / "datasets/osf_ultimatum/osfstorage-archive.zip").exists())

    def test_hypothesis_has_failure_conditions(self):
        data = json.loads((ROOT / "hypotheses/LL-01-OSF-001.json").read_text())
        self.assertGreaterEqual(len(data["failure_conditions"]), 2)

    def test_experiment_reproducible(self):
        script = ROOT / "experiments/osf_ultimatum/run_experiment.py"
        subprocess.run([sys.executable, str(script)], check=True, capture_output=True, text=True)
        first = (ROOT / "results/osf_ultimatum/result.json").read_text()
        subprocess.run([sys.executable, str(script)], check=True, capture_output=True, text=True)
        second = (ROOT / "results/osf_ultimatum/result.json").read_text()
        self.assertEqual(first, second)

    def test_phi_identifiability_is_not_overclaimed(self):
        path = ROOT / "results/osf_ultimatum/result.json"
        if not path.exists():
            subprocess.run([sys.executable, str(ROOT / "experiments/osf_ultimatum/run_experiment.py")], check=True)
        data = json.loads(path.read_text())
        self.assertEqual(data["phi_identifiability"]["verdict"], "INVALID_TEST")

if __name__ == "__main__":
    unittest.main()
