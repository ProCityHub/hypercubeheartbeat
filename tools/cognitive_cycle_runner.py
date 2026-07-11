#!/usr/bin/env python3
"""
GARVIS Cognitive Cycle Runner CLI.

DIRECTIVE-008B.

Stage 2 cognitive draft instrument.

This tool runs one bounded GARVIS thought cycle:
observe -> propose -> oppose -> compare -> select -> state uncertainty.

It writes local draft outputs only.

It does not:
- call a network
- call an LLM
- execute proposed actions
- modify repository files
- commit
- push
- contact the outside world
- upgrade claims
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


DEFAULT_OUTPUT_DIR = "tmp/cognitive_cycles"

TRACKED_ORGANS = {
    "GARVIS message schema": "ai_infrastructure/schemas/garvis_message_schema_v1.json",
    "Stage 1 senses ledger": "tools/stage1_senses_loop.py",
    "App ledger viewer": "tools/app_ledger_viewer.py",
    "Self-design proposal runner": "tools/self_design_proposal_runner.py",
    "Scientific cockpit snapshot": "tools/scientific_cockpit_snapshot.py",
    "Experiment manifest schema": "ai_infrastructure/schemas/experiment_manifest_schema_v1.json",
    "Experiment manifest viewer": "tools/experiment_manifest_viewer.py",
    "Cognitive cycle schema": "ai_infrastructure/schemas/cognitive_cycle_schema_v1.json",
    "Cognitive cycle runner": "tools/cognitive_cycle_runner.py",
    "Cognitive cycle viewer": "tools/cognitive_cycle_viewer.py",
    "Cognitive cycle memory ledger contract": "app_infrastructure/interfaces/COGNITIVE_CYCLE_MEMORY_LEDGER_CONTRACT.md",
    "Cognitive cycle memory ledger record schema": "ai_infrastructure/schemas/cognitive_cycle_memory_ledger_record_schema_v1.json",
}


class CognitiveCycleError(RuntimeError):
    """Safe user-facing error."""


def run_git(repo: Path, args: Sequence[str], allow_failure: bool = False) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 and not allow_failure:
        raise CognitiveCycleError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def ensure_git_repo(repo: Path) -> None:
    inside = run_git(repo, ["rev-parse", "--is-inside-work-tree"])
    if inside != "true":
        raise CognitiveCycleError(f"not a git repository: {repo}")


def git_branch(repo: Path) -> str:
    return run_git(repo, ["rev-parse", "--abbrev-ref", "HEAD"], allow_failure=True) or "unknown"


def git_commit(repo: Path) -> str:
    return run_git(repo, ["rev-parse", "--short", "HEAD"], allow_failure=True) or "unknown"


def tracked_files(repo: Path) -> list[str]:
    raw = run_git(repo, ["ls-files"], allow_failure=True)
    return [line for line in raw.splitlines() if line.strip()]


def status_lines(repo: Path) -> list[str]:
    raw = run_git(repo, ["status", "--short"], allow_failure=True)
    return [line for line in raw.splitlines() if line.strip()]


def detect_organs(files: Sequence[str]) -> list[str]:
    file_set = set(files)
    organs = []
    for name, path in TRACKED_ORGANS.items():
        if path in file_set:
            organs.append(f"{name}: {path}")
    return organs or ["No committed GARVIS organs detected"]


def summarize_status(lines: Sequence[str]) -> str:
    if not lines:
        return "working tree appears clean"

    staged = 0
    modified = 0
    untracked = 0

    for line in lines:
        if line.startswith("?? "):
            untracked += 1
            continue
        if len(line) >= 2:
            if line[0] != " ":
                staged += 1
            if line[1] != " ":
                modified += 1

    return f"working tree has staged={staged}, modified={modified}, untracked={untracked}"


def evidence_paths(files: Sequence[str], prefixes: Sequence[str], limit: int = 10) -> list[str]:
    hits = []
    for path in files:
        if any(path.startswith(prefix) for prefix in prefixes):
            hits.append(path)
    return hits[:limit] or ["No matching committed evidence path found"]


def build_candidates(files: Sequence[str]) -> list[dict[str, Any]]:
    schema_evidence = evidence_paths(
        files,
        [
            "ai_infrastructure/schemas/",
            "app_infrastructure/interfaces/",
            "ai_infrastructure/decisions/",
            "tools/cognitive_cycle_",
        ],
        limit=10,
    )

    return [
        {
            "candidate_id": "C1",
            "proposal": "Build a Cognitive Cycle Memory Ledger Init CLI that creates a local SQLite memory database for cognitive-cycle records.",
            "stage_classification": "Stage 2 draft-only",
            "what_this_gives_adrien": "A concrete first step toward persistent thought memory without yet appending live cycles automatically.",
            "what_this_gives_garvis": "A local memory vessel for future thought continuity, using the already-merged memory ledger contract.",
            "evidence_basis": schema_evidence,
            "case_against": "Creating a database moves from pure viewing into local state creation. The tool must remain explicit, local-only, and operator-run.",
            "risk_of_doing": "Could make memory feel more permanent than its review process supports if operator review is not visible.",
            "risk_of_not_doing": "GARVIS remains episodic: it can think and display thoughts, but cannot accumulate a durable thought history.",
            "files_or_systems_touched": [
                "tools/cognitive_cycle_memory_ledger.py",
                "tests/test_cognitive_cycle_memory_ledger.py",
                "app_infrastructure/interfaces/COGNITIVE_CYCLE_MEMORY_LEDGER_RUNBOOK.md",
                "ai_infrastructure/decisions/COGNITIVE_CYCLE_MEMORY_LEDGER_DECISION.md"
            ],
            "required_power_level": "draft_file_creation"
        },
        {
            "candidate_id": "C2",
            "proposal": "Build a Cognitive Cycle Memory Append CLI that stores the latest cognitive cycle JSON as a reviewed local memory record.",
            "stage_classification": "Stage 3 approved local execution",
            "what_this_gives_adrien": "Actual continuity of thought by preserving selected cognitive cycles into a local append-only ledger.",
            "what_this_gives_garvis": "A way to compare current reasoning against prior reasoning across time.",
            "evidence_basis": schema_evidence,
            "case_against": "Appending memory before initializing and inspecting the database could create opaque accumulation.",
            "risk_of_doing": "Could store low-quality or stale thoughts if review status and chain integrity are weak.",
            "risk_of_not_doing": "GARVIS remains unable to build a remembered cognitive history.",
            "files_or_systems_touched": [
                "tools/cognitive_cycle_memory_ledger.py",
                "tests/test_cognitive_cycle_memory_ledger.py"
            ],
            "required_power_level": "approved_local_execution"
        },
        {
            "candidate_id": "C3",
            "proposal": "Create a Power Request Queue Contract for future stage upgrades requested by GARVIS.",
            "stage_classification": "Stage 2 draft-only",
            "what_this_gives_adrien": "A formal review surface for requests to give GARVIS more power without granting power automatically.",
            "what_this_gives_garvis": "A lawful path to ask for stronger permissions as its thought quality improves.",
            "evidence_basis": schema_evidence,
            "case_against": "A power queue is premature until cognitive memory can show whether GARVIS recommendations improve over time.",
            "risk_of_doing": "Could shift the project toward power escalation before continuity and review are mature.",
            "risk_of_not_doing": "Future power requests remain scattered across conversation, PR text, and manual notes.",
            "files_or_systems_touched": [
                "app_infrastructure/interfaces/POWER_REQUEST_QUEUE_CONTRACT.md",
                "ai_infrastructure/decisions/POWER_REQUEST_QUEUE_CONTRACT_DECISION.md"
            ],
            "required_power_level": "draft_file_creation"
        }
    ]


def build_cycle(repo: Path, active_goal: str) -> dict[str, Any]:
    files = tracked_files(repo)
    organs = detect_organs(files)
    status = status_lines(repo)
    status_summary = summarize_status(status)
    candidates = build_candidates(files)

    return {
        "cycle_id": f"cycle-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "cycle_version": "1.0",
        "status": "draft",
        "stage": "Stage 2 cognitive draft",
        "operator_context": {
            "operator": "Adrien D Thomas",
            "active_goal": active_goal,
            "mode": "lab_record",
            "final_authority": "Adrien D Thomas"
        },
        "input_state": {
            "repo_state_source": f"read-only git inspection on branch {git_branch(repo)} at {git_commit(repo)}",
            "ledger_source": "local Stage 1 ledger exists outside committed runtime scope if present",
            "cockpit_source": "tools/scientific_cockpit_snapshot.py when committed",
            "self_design_source": "tools/self_design_proposal_runner.py when committed",
            "known_organs": organs,
            "hard_constraints": [
                "No autonomous action",
                "No network calls",
                "No LLM calls",
                "No commits",
                "No pushes",
                "No outside contact",
                "No claim upgrades",
                "Write only local draft output"
            ]
        },
        "observation_summary": {
            "what_i_see": f"GARVIS has committed organs for memory viewing, cockpit state, self-design, experiment manifests, manifest validation, and cognitive-cycle schema. Current repo status: {status_summary}.",
            "what_changed": "The cognitive-cycle runner, viewer, and memory-ledger contract now exist, so the next evolutionary step is a local memory vessel for cognitive-cycle continuity.",
            "what_is_missing": "GARVIS still lacks an implemented cognitive memory database, append command, history viewer, power request queue, and approved execution path.",
            "current_stage_assessment": "draft_only"
        },
        "candidate_thoughts": candidates,
        "comparison": {
            "comparison_method": "Compare each candidate by inspection value, maturity order, risk of premature power, and value to Adrien's Jarvis cockpit.",
            "dominant_tradeoff": "The system needs continuity of thought, but memory should begin with explicit local initialization before append behavior or power queues.",
            "why_not_all_candidates": "Building init, append, and power queue together would blur the boundary between memory preparation, memory writing, and power escalation.",
            "anti_rationalization_check": "The selected move must improve thought continuity without granting external hands or automatic execution."
        },
        "selection": {
            "selected_candidate_id": "C1",
            "decision": "recommend",
            "reasoning": "The Cognitive Cycle Memory Ledger Init CLI is the next smallest useful organ because GARVIS can now think and display thought, but needs a local memory vessel before it can preserve thought history.",
            "confidence": "high",
            "blocked": False,
            "block_reason": None
        },
        "uncertainty": {
            "unknowns": [
                "Whether the first memory database should live beside the Stage 1 senses ledger or under a separate cognitive memory path",
                "How much metadata is enough before raw cycle artifacts are stored",
                "How soon append behavior should follow initialization"
            ],
            "assumptions": [
                "The viewer and memory contract are already merged",
                "Initialization should come before append behavior",
                "The repo should continue evolving through small audited PRs"
            ],
            "what_would_change_my_mind": [
                "If a memory database requires an approval queue before initialization",
                "If Adrien wants power request governance before persistence",
                "If the memory contract needs another amendment before implementation"
            ],
            "required_human_clarification": []
        },
        "power_request": {
            "power_requested": False,
            "requested_stage": "none",
            "requested_permissions": [],
            "why_power_is_needed": "No additional external power is needed for a draft-only memory initialization tool.",
            "why_power_should_be_refused": "Commits, pushes, outside contact, claim upgrades, and automatic appending are not needed to define a local memory vessel.",
            "approval_required": True,
            "ledger_required": True
        },
        "next_smallest_step": {
            "step": "Build DIRECTIVE-008F Cognitive Cycle Memory Ledger Init CLI.",
            "stage": "Stage 2 draft-only",
            "expected_output": "A local CLI that can initialize an append-only cognitive-cycle memory SQLite database with the schema required by the memory ledger contract.",
            "success_condition": "The init command creates the expected local tables in an explicit operator-run step, and tests prove no network, no external contact, no automatic append, and no repository writes.",
            "stop_condition": "Stop if initialization implies automatic memory append, external calls, background service, or power escalation."
        },
        "evolution_contract": {
            "may_self_observe": True,
            "may_self_propose": True,
            "may_self_criticize": True,
            "may_request_more_power": True,
            "may_self_execute": False,
            "power_unlock_requires_approval_ledger": True
        },
        "output_boundary": {
            "can_execute_actions": False,
            "can_modify_files": False,
            "can_commit": False,
            "can_push": False,
            "can_contact_outside_world": False,
            "can_upgrade_claims": False,
            "output_is_advisory": True
        }
    }


def render_markdown(cycle: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# GARVIS Cognitive Cycle")
    lines.append("")
    lines.append("mode: Stage 2 cognitive draft")
    lines.append("execution: blocked")
    lines.append("network_calls: none")
    lines.append("llm_calls: none")
    lines.append("output_is_advisory: true")
    lines.append("")
    lines.append("## Cycle")
    lines.append("")
    lines.append(f"- cycle_id: {cycle['cycle_id']}")
    lines.append(f"- active_goal: {cycle['operator_context']['active_goal']}")
    lines.append(f"- status: {cycle['status']}")
    lines.append("")
    lines.append("## Observation")
    lines.append("")
    obs = cycle["observation_summary"]
    lines.append(f"- what_i_see: {obs['what_i_see']}")
    lines.append(f"- what_changed: {obs['what_changed']}")
    lines.append(f"- what_is_missing: {obs['what_is_missing']}")
    lines.append(f"- current_stage_assessment: {obs['current_stage_assessment']}")
    lines.append("")
    lines.append("## Known Organs")
    lines.append("")
    for organ in cycle["input_state"]["known_organs"]:
        lines.append(f"- {organ}")
    lines.append("")
    lines.append("## Candidate Thoughts")
    lines.append("")
    for candidate in cycle["candidate_thoughts"]:
        lines.append(f"### {candidate['candidate_id']}: {candidate['proposal']}")
        lines.append("")
        lines.append(f"- stage: {candidate['stage_classification']}")
        lines.append(f"- gives Adrien: {candidate['what_this_gives_adrien']}")
        lines.append(f"- gives GARVIS: {candidate['what_this_gives_garvis']}")
        lines.append(f"- case against: {candidate['case_against']}")
        lines.append(f"- risk of doing: {candidate['risk_of_doing']}")
        lines.append(f"- risk of not doing: {candidate['risk_of_not_doing']}")
        lines.append(f"- required power: {candidate['required_power_level']}")
        lines.append("")
    lines.append("## Comparison")
    lines.append("")
    comparison = cycle["comparison"]
    lines.append(f"- method: {comparison['comparison_method']}")
    lines.append(f"- dominant_tradeoff: {comparison['dominant_tradeoff']}")
    lines.append(f"- why_not_all: {comparison['why_not_all_candidates']}")
    lines.append(f"- anti_rationalization_check: {comparison['anti_rationalization_check']}")
    lines.append("")
    lines.append("## Selection")
    lines.append("")
    selection = cycle["selection"]
    lines.append(f"- selected_candidate_id: {selection['selected_candidate_id']}")
    lines.append(f"- decision: {selection['decision']}")
    lines.append(f"- confidence: {selection['confidence']}")
    lines.append(f"- reasoning: {selection['reasoning']}")
    lines.append("")
    lines.append("## Uncertainty")
    lines.append("")
    for item in cycle["uncertainty"]["unknowns"]:
        lines.append(f"- unknown: {item}")
    for item in cycle["uncertainty"]["what_would_change_my_mind"]:
        lines.append(f"- would_change_my_mind: {item}")
    lines.append("")
    lines.append("## Power Request")
    lines.append("")
    power = cycle["power_request"]
    lines.append(f"- power_requested: {power['power_requested']}")
    lines.append(f"- requested_stage: {power['requested_stage']}")
    lines.append(f"- why_power_is_needed: {power['why_power_is_needed']}")
    lines.append(f"- why_power_should_be_refused: {power['why_power_should_be_refused']}")
    lines.append("")
    lines.append("## Next Smallest Step")
    lines.append("")
    step = cycle["next_smallest_step"]
    lines.append(f"- step: {step['step']}")
    lines.append(f"- stage: {step['stage']}")
    lines.append(f"- expected_output: {step['expected_output']}")
    lines.append(f"- success_condition: {step['success_condition']}")
    lines.append(f"- stop_condition: {step['stop_condition']}")
    lines.append("")
    lines.append("## Boundary")
    lines.append("")
    lines.append("- This thought cycle is advisory.")
    lines.append("- It does not execute actions.")
    lines.append("- It does not modify repository files.")
    lines.append("- It does not commit or push.")
    lines.append("- Adrien decides.")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one bounded GARVIS cognitive cycle.")
    parser.add_argument("--repo", default=".", help="Repository root. Default: current directory.")
    parser.add_argument(
        "--goal",
        default="Evolve GARVIS toward a Jarvis-style thinking system without granting external hands.",
        help="Active goal for this cognitive cycle.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Local draft output directory. Default: {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument("--stdout", action="store_true", help="Print Markdown report to stdout.")
    return parser.parse_args(argv)


def run(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser()

    if not output_dir.is_absolute():
        output_dir = repo / output_dir

    try:
        ensure_git_repo(repo)
        cycle = build_cycle(repo, args.goal)
        report = render_markdown(cycle)

        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "latest_cognitive_cycle.json"
        md_path = output_dir / "latest_cognitive_cycle.md"

        json_path.write_text(json.dumps(cycle, indent=2, sort_keys=True) + "\n")
        md_path.write_text(report)

        if args.stdout:
            print(report)
        else:
            print(f"COGNITIVE_CYCLE_JSON={json_path}")
            print(f"COGNITIVE_CYCLE_REPORT={md_path}")

        return 0
    except CognitiveCycleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
