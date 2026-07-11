import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "self_design_proposal_runner.py"


class SelfDesignProposalRunnerTests(unittest.TestCase):
    def test_runner_writes_local_report_with_required_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "self_design_report.md"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--repo",
                    str(REPO_ROOT),
                    "--output",
                    str(output),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("SELF_DESIGN_REPORT_READY", result.stdout)
            self.assertTrue(output.exists())

            report = output.read_text()
            self.assertIn("# Hypercube Self-Design Proposal Report", report)
            self.assertIn("Local Stage 2 draft-only report.", report)
            self.assertIn("No network call.", report)
            self.assertIn("No LLM call.", report)
            self.assertIn("No repository action.", report)
            self.assertIn("No autonomous action.", report)
            self.assertIn("## Proposal Cap", report)
            self.assertIn("## Proposal 1:", report)
            self.assertIn("## Proposal 2:", report)
            self.assertIn("## Proposal 3:", report)

            self.assertEqual(report.count("### Case Against This Proposal"), 3)
            self.assertEqual(report.count("### Advisor Inputs"), 3)
            proposal_headings = [
                line for line in report.splitlines()
                if line.startswith("## Proposal ") and ":" in line
            ]
            self.assertEqual(len(proposal_headings), 3)
            self.assertNotIn("## Proposal 4:", report)

    def test_report_cites_repository_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "self_design_report.md"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--repo",
                    str(REPO_ROOT),
                    "--output",
                    str(output),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            report = output.read_text()

            self.assertIn("Ledger Or Repository Evidence Relied On", report)
            self.assertIn("app_infrastructure", report)
            self.assertIn("ai_infrastructure", report)

    def test_stdout_mode_prints_report_and_writes_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "self_design_report.md"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--repo",
                    str(REPO_ROOT),
                    "--output",
                    str(output),
                    "--stdout",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.exists())
            self.assertIn("# Hypercube Self-Design Proposal Report", result.stdout)

    def test_non_git_repo_fails_safely_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.md"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--repo",
                    tmp,
                    "--output",
                    str(output),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("ERROR:", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse(output.exists())

    def test_source_contains_no_network_imports(self):
        source = SCRIPT_PATH.read_text()
        forbidden = [
            "import requests",
            "from requests",
            "import urllib",
            "from urllib",
            "import socket",
            "from socket",
            "http://",
            "https://",
        ]
        for item in forbidden:
            self.assertNotIn(item, source)


if __name__ == "__main__":
    unittest.main()
