#!/usr/bin/env python3
"""
GARVIS Self-Design Proposal Runner.

Stage 2 draft-only tool.

This tool inspects committed repository files as local read-only context and
generates up to three future infrastructure/tool proposals for Adrien.

It does not:
- call a network
- call an LLM
- read secrets
- read raw runtime sensor data
- commit files
- push branches
- open pull requests
- execute proposed actions
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


SAFE_PREFIXES = (
    ".github/workflows/",
    ".github/scripts/",
    "docs/",
    "ai_infrastructure/",
    "app_infrastructure/",
    "tools/",
    "tests/",
)

FORBIDDEN_PREFIXES = (
    "data/stage1_senses/",
    "AGI/",
)

FORBIDDEN_EXACT = {
    "brain.py",
}

TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".json",
    ".yml",
    ".yaml",
    ".txt",
}

MAX_FILE_BYTES = 180_000
MAX_REPORT_PROPOSALS = 3


class RunnerError(RuntimeError):
    """Safe user-facing failure."""


def run_git(repo: Path, args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RunnerError(result.stderr.strip() or "git command failed")
    return result.stdout


def is_git_repo(repo: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--is-inside-work-tree"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def list_committed_files(repo: Path) -> list[str]:
    if not is_git_repo(repo):
        raise RunnerError(f"not a git repository: {repo}")

    raw = run_git(repo, ["ls-files"])
    files = [line.strip() for line in raw.splitlines() if line.strip()]
    return sorted(files)


def is_safe_committed_context(path: str) -> bool:
    if path in FORBIDDEN_EXACT:
        return False
    if any(path.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
        return False
    if not any(path.startswith(prefix) for prefix in SAFE_PREFIXES):
        return False
    suffix = Path(path).suffix.lower()
    return suffix in TEXT_SUFFIXES


def read_text_safely(repo: Path, rel_path: str) -> str:
    path = repo / rel_path
    try:
        data = path.read_bytes()
    except OSError:
        return ""

    if len(data) > MAX_FILE_BYTES:
        data = data[:MAX_FILE_BYTES]

    return data.decode("utf-8", errors="replace")


def keyword_hits(text: str) -> list[str]:
    lowered = text.lower()
    keywords = [
        "constitution",
        "living constitution",
        "stage 1",
        "stage 2",
        "stage 3",
        "stage 4",
        "ledger",
        "viewer",
        "proposal",
        "case against",
        "approval",
        "guard",
        "test",
        "app infrastructure",
        "cockpit",
        "self-design",
        "read-only",
        "no-network",
        "claim",
        "evidence",
        "preregistration",
    ]
    return [word for word in keywords if word in lowered]


def summarize_files(repo: Path, files: Sequence[str]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for rel in files:
        if not is_safe_committed_context(rel):
            continue
        text = read_text_safely(repo, rel)
        summaries.append(
            {
                "path": rel,
                "bytes_read": len(text.encode("utf-8")),
                "line_count": text.count("\n") + 1 if text else 0,
                "keywords": keyword_hits(text),
            }
        )
    return summaries


def has_path(summaries: Sequence[dict[str, object]], needle: str) -> bool:
    return any(needle in str(item["path"]) for item in summaries)


def matching_paths(
    summaries: Sequence[dict[str, object]],
    needles: Sequence[str],
    limit: int = 8,
) -> list[str]:
    matches: list[str] = []
    for item in summaries:
        path = str(item["path"])
        text = path.lower()
        if any(needle.lower() in text for needle in needles):
            matches.append(path)
    return matches[:limit]


def detected_organs(summaries: Sequence[dict[str, object]]) -> list[str]:
    organs: list[str] = []

    checks = [
        (
            "GARVIS constitution / constitutional law",
            ["GARVIS_CONSTITUTION", "constitution"],
        ),
        (
            "Living constitution / growth doctrine",
            ["LIVING_CONSTITUTION", "living"],
        ),
        (
            "Staged action ladder",
            ["STAGED_ACTION_LADDER", "action_ladder"],
        ),
        (
            "Stage 1 local senses ledger",
            ["stage1_senses", "STAGE1", "senses"],
        ),
        (
            "App infrastructure shell",
            ["app_infrastructure", "APP_INFRASTRUCTURE"],
        ),
        (
            "App ledger viewer CLI",
            ["app_ledger_viewer", "APP_LEDGER_VIEWER"],
        ),
        (
            "Hypercube self-design contract",
            ["SELF_DESIGN", "self_design"],
        ),
        (
            "GARVIS message schema / communication protocol",
            ["garvis_message", "message_schema", "communication"],
        ),
        (
            "Guard checks and tests",
            ["guard_check", "tests/"],
        ),
    ]

    for label, needles in checks:
        if any(has_path(summaries, needle) for needle in needles):
            organs.append(label)

    return organs


def bullet_paths(paths: Sequence[str]) -> str:
    if not paths:
        return "- Evidence unavailable in scanned committed files."
    return "\n".join(f"- `{path}`" for path in paths)


def build_proposals(summaries: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    constitution_evidence = matching_paths(
        summaries,
        ["constitution", "STAGED_ACTION_LADDER", "SELF_DESIGN", "APP_LEDGER_VIEWER"],
    )
    viewer_evidence = matching_paths(
        summaries,
        ["app_ledger_viewer", "APP_LEDGER_VIEWER", "stage1_senses"],
    )
    app_evidence = matching_paths(
        summaries,
        ["app_infrastructure", "cockpit", "SELF_DESIGN", "message_schema"],
    )
    test_evidence = matching_paths(
        summaries,
        ["tests/", "guard_check", "ll07", "diagnostic", "stage1"],
    )

    proposals: list[dict[str, object]] = [
        {
            "title": "Scientific Cockpit Snapshot CLI",
            "stage": "Stage 0 view-only now; Stage 3 only if future versions run approved commands.",
            "purpose": (
                "Create a single local command that summarizes branch state, guard status, "
                "recent committed directives, viewer availability, and recent safe ledger metadata."
            ),
            "gives_adrien": (
                "A fast cockpit readout: what body exists, what memory exists, what branch is active, "
                "what checks are clean, and what is unsafe to commit."
            ),
            "gives_garvis": (
                "No new hands. It only gives GARVIS a clearer status surface to report back to Adrien."
            ),
            "files_touched": (
                "`tools/scientific_cockpit_snapshot.py`, tests, runbook, decision record."
            ),
            "network": "No.",
            "secrets": "No.",
            "sensor_data": "No raw sensor data. Safe ledger metadata only.",
            "outside_world": "No.",
            "approval": "Human-approved PR required.",
            "ledger": "No execution ledger needed for view-only mode; future command-running version would require approval records.",
            "tests": (
                "Tests should prove no network calls, no raw payload display, safe failure, and clear untracked-file warnings."
            ),
            "failure_modes": (
                "Could become a fake dashboard if it hides uncertainty; could be mistaken for approval; "
                "could accidentally expand into a command runner if not bounded."
            ),
            "case_against": (
                "This may be premature because the current CLI viewer already gives visibility into memory. "
                "A cockpit snapshot could duplicate existing commands and create maintenance overhead. "
                "It gives GARVIS a broader self-reporting surface, which could steer Adrien if the report becomes too polished. "
                "It should not be built if the next priority is proposal comparison or experiment reproducibility."
            ),
            "alternatives": (
                "Use raw git commands plus app_ledger_viewer.py; build a full graphical app first; or build the proposal runner deeper."
            ),
            "rejected": (
                "Raw commands are fragmented. A full app is too early. A deeper proposal runner before a cockpit snapshot may lack operational visibility."
            ),
            "basis": (
                "Consistent with Stage 0 view-only, App Infrastructure shell, App Ledger Viewer Contract, and Cockpit Vision."
            ),
            "evidence": sorted(set(viewer_evidence + constitution_evidence + app_evidence))[:10],
            "advisor_inputs": "None. Generated locally from committed repository context only.",
            "next_step": "Build a read-only snapshot CLI that displays status and never executes tests by default.",
        },
        {
            "title": "Proposal Comparison Matrix",
            "stage": "Stage 2 draft-only preparation.",
            "purpose": (
                "Turn self-design proposals into a comparison table showing value, risk, stage, files touched, "
                "case against, evidence basis, and next smallest safe step."
            ),
            "gives_adrien": (
                "A better decision instrument. Adrien can compare proposals instead of being persuaded by the first strong narrative."
            ),
            "gives_garvis": (
                "A stricter proposal grammar. It makes GARVIS argue against its own proposals and expose tradeoffs."
            ),
            "files_touched": (
                "`tools/proposal_comparison_matrix.py`, tests, runbook, decision record."
            ),
            "network": "No.",
            "secrets": "No.",
            "sensor_data": "No.",
            "outside_world": "No.",
            "approval": "Human-approved PR required.",
            "ledger": "No execution ledger needed because it is draft-only.",
            "tests": (
                "Tests should prove max proposal count, required Case Against section, advisory labeling, and no network imports."
            ),
            "failure_modes": (
                "Could become bureaucracy if too heavy; could make weak proposals look formal; could hide qualitative judgment behind a table."
            ),
            "case_against": (
                "This may slow real building by turning every idea into a form. It could make the system feel safer than it is. "
                "A matrix is not truth; it is only a decision aid. It gives GARVIS more framing power unless Adrien remains the final judge."
            ),
            "alternatives": (
                "Continue manual review; build the cockpit snapshot first; build an experiment dashboard first."
            ),
            "rejected": (
                "Manual review does not scale. A cockpit snapshot shows state but not proposal tradeoffs. "
                "An experiment dashboard is powerful but needs clearer proposal selection first."
            ),
            "basis": (
                "Directly implements the Case Against requirement, proposal volume boundary, traceability requirement, and advisory ring law."
            ),
            "evidence": sorted(set(constitution_evidence + app_evidence))[:10],
            "advisor_inputs": "None. Generated locally from committed repository context only.",
            "next_step": "Add a local draft-only matrix generator for proposal reports.",
        },
        {
            "title": "Experiment Reproducibility Harness",
            "stage": "Stage 3 local approved execution for running commands; Stage 0 for viewing manifests.",
            "purpose": (
                "Create manifest-based experiment runs so tests, diagnostics, LL-07 demos, and future studies can be executed with "
                "clear inputs, outputs, boundaries, and audit records."
            ),
            "gives_adrien": (
                "More scientific power: reproducible runs, visible parameters, repeatable reports, and clearer separation between demo, sandbox, and evidence."
            ),
            "gives_garvis": (
                "No autonomous science. It gives GARVIS a stricter way to prepare and document approved experiments."
            ),
            "files_touched": (
                "`tools/experiment_repro_harness.py`, manifest schema, tests, runbook, decision record."
            ),
            "network": "No by default.",
            "secrets": "No.",
            "sensor_data": "No raw sensor data by default.",
            "outside_world": "No.",
            "approval": "Explicit approval required for each run command or run manifest.",
            "ledger": "Future approved runs should reference an approval ID if they execute beyond view-only inspection.",
            "tests": (
                "Tests should prove dry-run mode, no unapproved execution, output separation, and no claim upgrade."
            ),
            "failure_modes": (
                "This is powerful and can drift into action automation. It could create impressive reports that look like evidence when they are only demos. "
                "It must preserve the boundary between sandbox outputs and empirical results."
            ),
            "case_against": (
                "This may be too powerful as the immediate next step because it introduces command execution structure. "
                "If built too early, it could blur Stage 2 proposals with Stage 3 approved execution. "
                "It should wait until the cockpit snapshot and proposal comparison tools are mature."
            ),
            "alternatives": (
                "Keep running tests manually; build only dry-run manifests; build dashboard viewing before execution."
            ),
            "rejected": (
                "Manual runs are hard to audit. Dry-run-only is safer but less useful. Dashboard-only does not improve reproducibility."
            ),
            "basis": (
                "Aligned with Staged Action Ladder, approval ledger law, LL-07 demo boundaries, and scientific-power direction."
            ),
            "evidence": sorted(set(test_evidence + constitution_evidence))[:10],
            "advisor_inputs": "None. Generated locally from committed repository context only.",
            "next_step": "Do not build full execution yet; first draft a dry-run manifest viewer.",
        },
    ]

    return proposals[:MAX_REPORT_PROPOSALS]


def render_report(repo: Path, summaries: Sequence[dict[str, object]]) -> str:
    organs = detected_organs(summaries)
    proposals = build_proposals(summaries)

    lines: list[str] = []
    lines.append("# Hypercube Self-Design Proposal Report")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append("Local Stage 2 draft-only report.")
    lines.append("")
    lines.append("No network call.")
    lines.append("")
    lines.append("No LLM call.")
    lines.append("")
    lines.append("No repository action.")
    lines.append("")
    lines.append("No autonomous action.")
    lines.append("")
    lines.append("No raw runtime sensor data read.")
    lines.append("")
    lines.append("## Repository Context")
    lines.append("")
    lines.append(f"- Repository: `{repo}`")
    lines.append(f"- Committed safe files inspected: {len(summaries)}")
    lines.append("- Inspection mode: committed files only via `git ls-files`")
    lines.append("")
    lines.append("## Detected Infrastructure Organs")
    lines.append("")
    if organs:
        for organ in organs:
            lines.append(f"- {organ}")
    else:
        lines.append("- No known infrastructure organs detected.")
    lines.append("")
    lines.append("## Proposal Cap")
    lines.append("")
    lines.append(f"This report is capped at {MAX_REPORT_PROPOSALS} proposals.")
    lines.append("")
    lines.append("## Proposals")
    lines.append("")

    for index, proposal in enumerate(proposals, start=1):
        lines.append(f"## Proposal {index}: {proposal['title']}")
        lines.append("")
        lines.append("### Stage Classification")
        lines.append("")
        lines.append(str(proposal["stage"]))
        lines.append("")
        lines.append("### Purpose")
        lines.append("")
        lines.append(str(proposal["purpose"]))
        lines.append("")
        lines.append("### What This Gives Adrien")
        lines.append("")
        lines.append(str(proposal["gives_adrien"]))
        lines.append("")
        lines.append("### What This Gives GARVIS")
        lines.append("")
        lines.append(str(proposal["gives_garvis"]))
        lines.append("")
        lines.append("### Required Files Or Systems Touched")
        lines.append("")
        lines.append(str(proposal["files_touched"]))
        lines.append("")
        lines.append("### Network Access")
        lines.append("")
        lines.append(str(proposal["network"]))
        lines.append("")
        lines.append("### Secrets")
        lines.append("")
        lines.append(str(proposal["secrets"]))
        lines.append("")
        lines.append("### Sensor Data")
        lines.append("")
        lines.append(str(proposal["sensor_data"]))
        lines.append("")
        lines.append("### Outside-World Effects")
        lines.append("")
        lines.append(str(proposal["outside_world"]))
        lines.append("")
        lines.append("### Approval Requirement")
        lines.append("")
        lines.append(str(proposal["approval"]))
        lines.append("")
        lines.append("### Ledger Requirement")
        lines.append("")
        lines.append(str(proposal["ledger"]))
        lines.append("")
        lines.append("### Test Requirement")
        lines.append("")
        lines.append(str(proposal["tests"]))
        lines.append("")
        lines.append("### Failure Modes")
        lines.append("")
        lines.append(str(proposal["failure_modes"]))
        lines.append("")
        lines.append("### Case Against This Proposal")
        lines.append("")
        lines.append(str(proposal["case_against"]))
        lines.append("")
        lines.append("### Alternatives Considered")
        lines.append("")
        lines.append(str(proposal["alternatives"]))
        lines.append("")
        lines.append("### Why Alternatives Were Rejected")
        lines.append("")
        lines.append(str(proposal["rejected"]))
        lines.append("")
        lines.append("### Constitutional Basis")
        lines.append("")
        lines.append(str(proposal["basis"]))
        lines.append("")
        lines.append("### Ledger Or Repository Evidence Relied On")
        lines.append("")
        lines.append(bullet_paths(proposal["evidence"]))  # type: ignore[arg-type]
        lines.append("")
        lines.append("### Advisor Inputs")
        lines.append("")
        lines.append(str(proposal["advisor_inputs"]))
        lines.append("")
        lines.append("### Next Smallest Safe Step")
        lines.append("")
        lines.append(str(proposal["next_step"]))
        lines.append("")

    lines.append("## Final Boundary")
    lines.append("")
    lines.append("This report is advice, not authority.")
    lines.append("")
    lines.append("Adrien decides.")
    lines.append("")
    lines.append("GitHub records only after approved repository procedure.")
    lines.append("")

    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a local Stage 2 self-design proposal report from committed repository context."
    )
    parser.add_argument(
        "--repo",
        default=".",
        help="Repository root. Default: current directory.",
    )
    parser.add_argument(
        "--output",
        default="tmp/self_design_proposals/self_design_proposals.md",
        help="Local draft report output path.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Also print the report to stdout.",
    )
    return parser.parse_args(argv)


def run(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    repo = Path(args.repo).expanduser().resolve()
    output = Path(args.output).expanduser()

    try:
        files = list_committed_files(repo)
        summaries = summarize_files(repo, files)
        report = render_report(repo, summaries)

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")

        if args.stdout:
            print(report)
        else:
            print(f"SELF_DESIGN_REPORT_READY path={output}")

        return 0
    except RunnerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"ERROR: file operation failed: {exc}", file=sys.stderr)
        return 2


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
