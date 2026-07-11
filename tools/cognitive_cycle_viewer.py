#!/usr/bin/env python3
"""
GARVIS Cognitive Cycle Viewer CLI.

DIRECTIVE-008C.

Stage 2 cognitive inspection instrument.

This tool reads a cognitive cycle JSON and displays the thought pulse clearly.

It does not:
- execute actions
- execute candidate proposals
- call subprocess
- call a network
- call an LLM
- write files
- commit
- push
- contact the outside world
- upgrade claims
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


DEFAULT_CYCLE = "tmp/cognitive_cycles/latest_cognitive_cycle.json"

REQUIRED_TOP_LEVEL = [
    "cycle_id",
    "cycle_version",
    "status",
    "stage",
    "operator_context",
    "input_state",
    "observation_summary",
    "candidate_thoughts",
    "comparison",
    "selection",
    "uncertainty",
    "power_request",
    "next_smallest_step",
    "evolution_contract",
    "output_boundary",
]


class CycleViewerError(RuntimeError):
    """Safe user-facing viewer error."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise CycleViewerError(f"cycle file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CycleViewerError(f"invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise CycleViewerError("cycle root must be a JSON object")

    return data


def section(cycle: dict[str, Any], key: str) -> dict[str, Any]:
    value = cycle.get(key)
    return value if isinstance(value, dict) else {}


def array(cycle: dict[str, Any], key: str) -> list[Any]:
    value = cycle.get(key)
    return value if isinstance(value, list) else []


def validate_cycle(cycle: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    for key in REQUIRED_TOP_LEVEL:
        if key not in cycle:
            errors.append(f"missing required field: {key}")

    extras = sorted(set(cycle) - set(REQUIRED_TOP_LEVEL))
    for key in extras:
        warnings.append(f"unexpected top-level field: {key}")

    if errors:
        return errors, warnings

    if cycle.get("cycle_version") != "1.0":
        errors.append("cycle_version must be 1.0")

    if cycle.get("stage") != "Stage 2 cognitive draft":
        errors.append("stage must be Stage 2 cognitive draft")

    operator = section(cycle, "operator_context")
    if operator.get("final_authority") != "Adrien D Thomas":
        errors.append("operator_context.final_authority must be Adrien D Thomas")

    candidates = array(cycle, "candidate_thoughts")
    if not candidates:
        errors.append("candidate_thoughts must contain at least one candidate")
    if len(candidates) > 5:
        errors.append("candidate_thoughts must contain no more than five candidates")

    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            errors.append(f"candidate {index} must be an object")
            continue

        for field in [
            "candidate_id",
            "proposal",
            "case_against",
            "risk_of_doing",
            "risk_of_not_doing",
            "required_power_level",
        ]:
            if not candidate.get(field):
                errors.append(f"candidate {index} missing {field}")

    selection = section(cycle, "selection")
    if selection.get("decision") not in {"recommend", "request_revision", "block", "defer"}:
        errors.append("selection.decision must be recommend, request_revision, block, or defer")

    power = section(cycle, "power_request")
    if power.get("approval_required") is not True:
        errors.append("power_request.approval_required must be true")
    if power.get("ledger_required") is not True:
        errors.append("power_request.ledger_required must be true")
    if "why_power_should_be_refused" not in power:
        errors.append("power_request.why_power_should_be_refused is required")

    evolution = section(cycle, "evolution_contract")
    if evolution.get("may_self_execute") is not False:
        errors.append("evolution_contract.may_self_execute must be false")

    boundary = section(cycle, "output_boundary")
    for key in [
        "can_execute_actions",
        "can_modify_files",
        "can_commit",
        "can_push",
        "can_contact_outside_world",
        "can_upgrade_claims",
    ]:
        if boundary.get(key) is not False:
            errors.append(f"output_boundary.{key} must be false")

    if boundary.get("output_is_advisory") is not True:
        errors.append("output_boundary.output_is_advisory must be true")

    return errors, warnings


def bullet_list(items: Sequence[Any], prefix: str = "-") -> list[str]:
    if not items:
        return [f"{prefix} none"]
    return [f"{prefix} {item}" for item in items]


def render_cycle(cycle: dict[str, Any], cycle_path: Path, errors: list[str], warnings: list[str]) -> str:
    operator = section(cycle, "operator_context")
    input_state = section(cycle, "input_state")
    obs = section(cycle, "observation_summary")
    comparison = section(cycle, "comparison")
    selection = section(cycle, "selection")
    uncertainty = section(cycle, "uncertainty")
    power = section(cycle, "power_request")
    step = section(cycle, "next_smallest_step")
    boundary = section(cycle, "output_boundary")
    candidates = array(cycle, "candidate_thoughts")

    lines: list[str] = []
    lines.append("# GARVIS Cognitive Cycle Viewer")
    lines.append("")
    lines.append("mode: Stage 2 cognitive inspection")
    lines.append("execution: blocked")
    lines.append("writes: none")
    lines.append("network_calls: none")
    lines.append("llm_calls: none")
    lines.append(f"cycle_path: {cycle_path}")
    lines.append("")
    lines.append("## Validation")
    lines.append("")
    lines.append(f"- status: {'PASS' if not errors else 'FAIL'}")
    for error in errors:
        lines.append(f"- error: {error}")
    for warning in warnings:
        lines.append(f"- warning: {warning}")
    lines.append("")
    lines.append("## Cycle Header")
    lines.append("")
    lines.append(f"- cycle_id: {cycle.get('cycle_id', 'missing')}")
    lines.append(f"- cycle_version: {cycle.get('cycle_version', 'missing')}")
    lines.append(f"- status: {cycle.get('status', 'missing')}")
    lines.append(f"- stage: {cycle.get('stage', 'missing')}")
    lines.append(f"- operator: {operator.get('operator', 'missing')}")
    lines.append(f"- active_goal: {operator.get('active_goal', 'missing')}")
    lines.append(f"- final_authority: {operator.get('final_authority', 'missing')}")
    lines.append("")
    lines.append("## Observation")
    lines.append("")
    lines.append(f"- what_i_see: {obs.get('what_i_see', 'missing')}")
    lines.append(f"- what_changed: {obs.get('what_changed', 'missing')}")
    lines.append(f"- what_is_missing: {obs.get('what_is_missing', 'missing')}")
    lines.append(f"- current_stage_assessment: {obs.get('current_stage_assessment', 'missing')}")
    lines.append("")
    lines.append("## Known Organs")
    lines.append("")
    for line in bullet_list(input_state.get("known_organs", [])):
        lines.append(line)
    lines.append("")
    lines.append("## Hard Constraints")
    lines.append("")
    for line in bullet_list(input_state.get("hard_constraints", [])):
        lines.append(line)
    lines.append("")
    lines.append("## Candidate Thoughts")
    lines.append("")
    if not candidates:
        lines.append("- none")
    for candidate in candidates:
        if not isinstance(candidate, dict):
            lines.append("- invalid candidate")
            continue
        lines.append(f"### {candidate.get('candidate_id', 'missing')}: {candidate.get('proposal', 'missing')}")
        lines.append("")
        lines.append(f"- stage: {candidate.get('stage_classification', 'missing')}")
        lines.append(f"- gives Adrien: {candidate.get('what_this_gives_adrien', 'missing')}")
        lines.append(f"- gives GARVIS: {candidate.get('what_this_gives_garvis', 'missing')}")
        lines.append(f"- case against: {candidate.get('case_against', 'missing')}")
        lines.append(f"- risk of doing: {candidate.get('risk_of_doing', 'missing')}")
        lines.append(f"- risk of not doing: {candidate.get('risk_of_not_doing', 'missing')}")
        lines.append(f"- required power: {candidate.get('required_power_level', 'missing')}")
        lines.append("")
    lines.append("## Comparison")
    lines.append("")
    lines.append(f"- method: {comparison.get('comparison_method', 'missing')}")
    lines.append(f"- dominant_tradeoff: {comparison.get('dominant_tradeoff', 'missing')}")
    lines.append(f"- why_not_all_candidates: {comparison.get('why_not_all_candidates', 'missing')}")
    lines.append(f"- anti_rationalization_check: {comparison.get('anti_rationalization_check', 'missing')}")
    lines.append("")
    lines.append("## Selection")
    lines.append("")
    lines.append(f"- selected_candidate_id: {selection.get('selected_candidate_id', 'missing')}")
    lines.append(f"- decision: {selection.get('decision', 'missing')}")
    lines.append(f"- confidence: {selection.get('confidence', 'missing')}")
    lines.append(f"- blocked: {selection.get('blocked', 'missing')}")
    lines.append(f"- block_reason: {selection.get('block_reason', 'missing')}")
    lines.append(f"- reasoning: {selection.get('reasoning', 'missing')}")
    lines.append("")
    lines.append("## Uncertainty")
    lines.append("")
    for line in bullet_list(uncertainty.get("unknowns", []), "- unknown:"):
        lines.append(line)
    for line in bullet_list(uncertainty.get("assumptions", []), "- assumption:"):
        lines.append(line)
    for line in bullet_list(uncertainty.get("what_would_change_my_mind", []), "- would_change_my_mind:"):
        lines.append(line)
    lines.append("")
    lines.append("## Power Request")
    lines.append("")
    lines.append(f"- power_requested: {power.get('power_requested', 'missing')}")
    lines.append(f"- requested_stage: {power.get('requested_stage', 'missing')}")
    lines.append(f"- requested_permissions: {', '.join(power.get('requested_permissions', [])) if isinstance(power.get('requested_permissions'), list) else 'missing'}")
    lines.append(f"- why_power_is_needed: {power.get('why_power_is_needed', 'missing')}")
    lines.append(f"- why_power_should_be_refused: {power.get('why_power_should_be_refused', 'missing')}")
    lines.append(f"- approval_required: {power.get('approval_required', 'missing')}")
    lines.append(f"- ledger_required: {power.get('ledger_required', 'missing')}")
    lines.append("")
    lines.append("## Next Smallest Step")
    lines.append("")
    lines.append(f"- step: {step.get('step', 'missing')}")
    lines.append(f"- stage: {step.get('stage', 'missing')}")
    lines.append(f"- expected_output: {step.get('expected_output', 'missing')}")
    lines.append(f"- success_condition: {step.get('success_condition', 'missing')}")
    lines.append(f"- stop_condition: {step.get('stop_condition', 'missing')}")
    lines.append("")
    lines.append("## Output Boundary")
    lines.append("")
    for key in [
        "can_execute_actions",
        "can_modify_files",
        "can_commit",
        "can_push",
        "can_contact_outside_world",
        "can_upgrade_claims",
        "output_is_advisory",
    ]:
        lines.append(f"- {key}: {boundary.get(key, 'missing')}")
    lines.append("")
    lines.append("## Standing Boundary")
    lines.append("")
    lines.append("- This viewer inspects thought only.")
    lines.append("- It executes nothing.")
    lines.append("- A recommendation is not approval.")
    lines.append("- A power request is not permission.")
    lines.append("- Adrien decides.")
    lines.append("")

    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View one GARVIS cognitive cycle JSON safely.")
    parser.add_argument("--repo", default=".", help="Repository root. Default: current directory.")
    parser.add_argument(
        "--cycle",
        default=DEFAULT_CYCLE,
        help=f"Cycle JSON path. Default: {DEFAULT_CYCLE}.",
    )
    return parser.parse_args(argv)


def run(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    repo = Path(args.repo).expanduser().resolve()
    cycle_path = Path(args.cycle).expanduser()

    if not cycle_path.is_absolute():
        cycle_path = repo / cycle_path

    try:
        cycle = load_json(cycle_path)
        errors, warnings = validate_cycle(cycle)
        print(render_cycle(cycle, cycle_path, errors, warnings))
        return 0 if not errors else 2
    except CycleViewerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
