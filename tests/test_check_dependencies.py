"""Regression tests for repository dependency-path parsing."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CHECKER = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "check_dependencies.py"
)


def _initialize_repo(repo: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "add", "--all"],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _run_checker(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), "--repo", str(repo)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_jsonl_reference_is_not_truncated(tmp_path: Path) -> None:
    _initialize_repo(
        tmp_path,
        {
            "OUTCOMES.md": "Evidence: `data/ll12_run_v1.jsonl`\n",
            "data/ll12_run_v1.jsonl": '{"beat": 1}\n',
        },
    )

    result = _run_checker(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "DEPENDENCY CHECK PASSED" in result.stdout


def test_jsonl_files_are_scanned_for_dependencies(tmp_path: Path) -> None:
    record_path = "data/" + "record.jsonl"
    missing_path = "data/" + "missing.json"

    _initialize_repo(
        tmp_path,
        {
            record_path: f'{{"source": "{missing_path}"}}\n',
        },
    )

    result = _run_checker(tmp_path)

    assert result.returncode == 1
    expected = f"{record_path}: missing referenced path: {missing_path}"
    assert expected in result.stdout
