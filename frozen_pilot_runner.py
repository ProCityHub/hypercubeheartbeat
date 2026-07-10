#!/usr/bin/env python3
"""
Frozen Pilot Runner

Autonomy Directive Task 2.

Runs a locked diagnostic fixture through the Autonomy Task 1 diagnostic report.
Writes preserved results without changing scoring after execution.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from lattice_diagnostic_report import REQUIRED_FIELDS, build_diagnostic_report


FIXTURE_PATH = Path("frozen_pilot/fixtures/task_2_inputs.json")
RESULTS_DIR = Path("results/frozen_pilot")
RESULTS_JSON = RESULTS_DIR / "task_2_results.json"
SUMMARY_MD = RESULTS_DIR / "TASK_2_SUMMARY.md"


def load_fixture(path: Path = FIXTURE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_pilot(fixture: dict[str, Any]) -> dict[str, Any]:
    inputs = fixture["inputs"]
    reports = []

    for item in inputs:
        report = build_diagnostic_report(
            observer=item["observer"],
            actor=item["actor"],
            bridge=item["bridge"],
            cube_corner=item["cube_corner"],
            heartbeat_phase=item["heartbeat_phase"],
            coherence_score=item["coherence_score"],
            action=item["action"],
            habit_signatures=item.get("habit_signatures", {}),
            history_length=item.get("history_length", 0),
            timing_coherence=item.get("timing_coherence"),
            information_coupling=item.get("information_coupling"),
            integration_pressure=item.get("integration_pressure"),
        ).to_dict()

        reports.append(
            {
                "input_id": item["id"],
                "report": report,
                "report_hash": stable_hash(report),
            }
        )

    criteria = evaluate_criteria(reports)

    return {
        "pilot_name": fixture["pilot_name"],
        "version": fixture["version"],
        "locked": fixture["locked"],
        "fixture_hash": stable_hash(fixture),
        "reports": reports,
        "criteria": criteria,
        "summary": "SUPPORTED" if criteria["supported"] else "NOT_SUPPORTED",
    }


def evaluate_criteria(reports: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["input_id"]: row["report"] for row in reports}
    failures = []

    for row in reports:
        report = row["report"]

        for field in REQUIRED_FIELDS:
            if field not in report:
                failures.append(f"{row['input_id']}: missing required field {field}")

        guardrail = report.get("evidence", {}).get("claim_guardrail", "")
        lowered = guardrail.lower()
        if "does not prove consciousness" not in lowered:
            failures.append(f"{row['input_id']}: missing consciousness guardrail")
        if "completed agi" not in lowered:
            failures.append(f"{row['input_id']}: missing AGI guardrail")

    expected = {
        "formula_ceiling_001": "formula_made_possible",
        "substrate_bridge_failure_001": "substrate_made_possible",
        "insufficient_history_001": "insufficient_evidence",
        "observer_limited_001": "not_bridge_limited",
    }

    for input_id, expected_origin in expected.items():
        actual = by_id[input_id]["limit_origin"]
        if actual != expected_origin:
            failures.append(
                f"{input_id}: expected {expected_origin}, got {actual}"
            )

    formula_warning = "do not treat this as automatic Bridge failure"
    warnings = by_id["formula_ceiling_001"]["warnings"]
    if not any(formula_warning in warning for warning in warnings):
        failures.append(
            "formula_ceiling_001: sparse firing warning was not preserved"
        )

    return {
        "supported": len(failures) == 0,
        "failures": failures,
        "checked_reports": len(reports),
    }


def write_results(result: dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    RESULTS_JSON.write_text(json.dumps(result, indent=2, sort_keys=True))

    lines = [
        "# Autonomy Directive Task 2 — Frozen Pilot Summary",
        "",
        f"Pilot: {result['pilot_name']}",
        f"Version: {result['version']}",
        f"Locked fixture: {result['locked']}",
        f"Fixture hash: `{result['fixture_hash']}`",
        f"Summary: **{result['summary']}**",
        "",
        "## Criteria",
        "",
        f"Supported: `{result['criteria']['supported']}`",
        f"Checked reports: `{result['criteria']['checked_reports']}`",
        "",
        "## Failures",
        "",
    ]

    if result["criteria"]["failures"]:
        for failure in result["criteria"]["failures"]:
            lines.append(f"- {failure}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Report Hashes",
            "",
        ]
    )

    for row in result["reports"]:
        lines.append(f"- `{row['input_id']}` → `{row['report_hash']}`")

    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "This pilot does not prove consciousness or completed AGI.",
            "It only tests whether the diagnostic report behaves deterministically under frozen inputs.",
        ]
    )

    SUMMARY_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    fixture = load_fixture()
    result = run_pilot(fixture)
    write_results(result)

    print(json.dumps(
        {
            "summary": result["summary"],
            "supported": result["criteria"]["supported"],
            "failures": result["criteria"]["failures"],
            "results_json": str(RESULTS_JSON),
            "summary_md": str(SUMMARY_MD),
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
