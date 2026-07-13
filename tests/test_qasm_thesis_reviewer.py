import json
import tempfile
import unittest
from pathlib import Path
import qasm_thesis_reviewer as reviewer

SAMPLE = '''// Consciousness-inspired double slit memory test
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q;
ry(0.618034) q[0];
cx q[0], q[1];
cx q[1], q[2];
measure q -> c;
'''

class ReviewerTests(unittest.TestCase):
    def test_parse_and_analyze(self):
        circuit = reviewer.parse_qasm(SAMPLE, "sample.qasm")
        analysis = reviewer.analyze(circuit)
        self.assertEqual(analysis["gate_counts"]["h"], 3)
        self.assertEqual(analysis["measurement_count"], 3)
        self.assertEqual(analysis["topology"], "chain")
        statuses = {item["concept"]: item["status"] for item in analysis["semantic_assessment"]}
        self.assertEqual(statuses["consciousness"], "comment_only_not_observable")
        self.assertEqual(analysis["angles"][0]["nearest_reference"], "inverse_phi")

    def test_review_is_deterministic_and_serializable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); qasm = root / "x.qasm"; theory = root / "t.md"
            qasm.write_text(SAMPLE); theory.write_text("C = O * A * B * phi\nC = O^1 * A^(1/phi) * B^(1/phi^2)\n")
            one = reviewer.build_review([qasm], [theory]); two = reviewer.build_review([qasm], [theory])
            self.assertEqual(one, two); json.dumps(one); self.assertTrue(one["formula_conflicts"])

    def test_cli_writes_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); qasm = root / "x.qasm"; out = root / "out"; qasm.write_text(SAMPLE)
            self.assertEqual(reviewer.main([str(qasm), "--output-dir", str(out)]), 0)
            self.assertTrue((out / "review.json").exists()); self.assertTrue((out / "review.md").exists())

if __name__ == "__main__": unittest.main()
