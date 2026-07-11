#!/usr/bin/env python3
"""
ARC Lattice Bridge

Autonomy Directive Task 4.

Purpose:
Connect simple ARC-style grid transformations to the Lattice Diagnostic Report.

This does not claim ARC completion, consciousness, or AGI.
It creates a deterministic bridge:

ARC grid pair -> transformation candidate -> coherence score -> diagnostic report
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from lattice_diagnostic_report import build_diagnostic_report


Grid = list[list[int]]


def shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0


def identity(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def rotate90(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid[::-1])]


def rotate180(grid: Grid) -> Grid:
    return [row[::-1] for row in grid[::-1]]


def rotate270(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)][::-1]


def flip_horizontal(grid: Grid) -> Grid:
    return [row[::-1] for row in grid]


def flip_vertical(grid: Grid) -> Grid:
    return grid[::-1]


TRANSFORMS: dict[str, Callable[[Grid], Grid]] = {
    "identity": identity,
    "rotate90": rotate90,
    "rotate180": rotate180,
    "rotate270": rotate270,
    "flip_horizontal": flip_horizontal,
    "flip_vertical": flip_vertical,
}


def grid_similarity(a: Grid, b: Grid) -> float:
    ha, wa = shape(a)
    hb, wb = shape(b)

    if ha == 0 or wa == 0 or hb == 0 or wb == 0:
        return 0.0

    shape_delta = abs(ha - hb) + abs(wa - wb)
    shape_score = 1.0 - min(1.0, shape_delta / max(ha + wa, hb + wb, 1))

    h = min(ha, hb)
    w = min(wa, wb)

    matches = 0
    for r in range(h):
        for c in range(w):
            if a[r][c] == b[r][c]:
                matches += 1

    cell_score = matches / (h * w) if h * w else 0.0

    return round((0.40 * shape_score) + (0.60 * cell_score), 6)


def score_transform(task: dict[str, Any], transform_name: str) -> float:
    fn = TRANSFORMS[transform_name]
    scores = []

    for pair in task["train"]:
        predicted = fn(pair["input"])
        scores.append(grid_similarity(predicted, pair["output"]))

    return round(sum(scores) / len(scores), 6) if scores else 0.0


def best_transform(task: dict[str, Any]) -> dict[str, Any]:
    scored = [
        {
            "name": name,
            "score": score_transform(task, name),
        }
        for name in sorted(TRANSFORMS)
    ]

    scored.sort(key=lambda item: (item["score"], item["name"]), reverse=True)
    return scored[0]


def palette_pressure(task: dict[str, Any]) -> float:
    values = []

    for pair in task["train"]:
        for grid_name in ("input", "output"):
            for row in pair[grid_name]:
                values.extend(row)

    if not values:
        return 0.0

    palette_size = len(Counter(values))
    return min(1.0, palette_size / 10.0)


def arc_task_to_diagnostic(task: dict[str, Any]) -> dict[str, Any]:
    best = best_transform(task)
    score = best["score"]

    observer = min(1.0, len(task.get("train", [])) / 5.0)
    actor = score
    bridge = score
    coherence_score = score
    action = "settle" if score >= 0.75 else "bounce"

    if observer >= 0.5 and actor >= 0.5 and bridge >= 0.5:
        cube_corner = "111"
    elif observer >= 0.5 and actor >= 0.5:
        cube_corner = "110"
    elif observer >= 0.5:
        cube_corner = "100"
    else:
        cube_corner = "000"

    diagnostic = build_diagnostic_report(
        observer=observer,
        actor=actor,
        bridge=bridge,
        cube_corner=cube_corner,
        heartbeat_phase="arc_bridge",
        coherence_score=coherence_score,
        action=action,
        habit_signatures={best["name"]: 1} if score >= 0.75 else {},
        history_length=len(task.get("train", [])),
        timing_coherence=score,
        information_coupling=score,
        integration_pressure=palette_pressure(task),
    ).to_dict()

    return {
        "task_id": task["id"],
        "best_transform": best,
        "diagnostic": diagnostic,
        "guardrail": (
            "ARC bridge only; does not prove ARC completion, consciousness, or completed AGI"
        ),
    }


def run_arc_bridge(fixture: dict[str, Any]) -> dict[str, Any]:
    results = [arc_task_to_diagnostic(task) for task in fixture["tasks"]]

    supported = all(
        "diagnostic" in row
        and "best_transform" in row
        and "does not prove" in row["guardrail"]
        for row in results
    )

    return {
        "pilot_name": fixture["pilot_name"],
        "locked": fixture["locked"],
        "results": results,
        "summary": "SUPPORTED" if supported else "NOT_SUPPORTED",
    }


def main() -> None:
    fixture_path = Path("arc_bridge/fixtures/task_4_arc_inputs.json")
    output_dir = Path("results/arc_bridge")
    output_dir.mkdir(parents=True, exist_ok=True)

    fixture = json.loads(fixture_path.read_text())
    result = run_arc_bridge(fixture)

    (output_dir / "task_4_arc_bridge_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )

    lines = [
        "# Autonomy Directive Task 4 — ARC Bridge Summary",
        "",
        f"Pilot: {result['pilot_name']}",
        f"Locked: {result['locked']}",
        f"Summary: **{result['summary']}**",
        "",
        "## Results",
        "",
    ]

    for row in result["results"]:
        lines.append(
            f"- `{row['task_id']}` → `{row['best_transform']['name']}` "
            f"score `{row['best_transform']['score']}` "
            f"action `{row['diagnostic']['action']}`"
        )

    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "This ARC bridge does not prove ARC completion, consciousness, or completed AGI.",
            "It only proves the diagnostic system can accept ARC-style grid tasks as structured input.",
        ]
    )

    (output_dir / "TASK_4_ARC_BRIDGE_SUMMARY.md").write_text("\n".join(lines) + "\n")

    print(json.dumps(
        {
            "summary": result["summary"],
            "results_json": str(output_dir / "task_4_arc_bridge_results.json"),
            "summary_md": str(output_dir / "TASK_4_ARC_BRIDGE_SUMMARY.md"),
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
