#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
from pathlib import Path


def read_optional(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "_missing_"
    return p.read_text()


def main() -> None:
    today = dt.date.today().isoformat()
    out_dir = Path("reports/coherence")
    out_dir.mkdir(parents=True, exist_ok=True)

    diagnostic_summary = read_optional("results/diagnostic_selftest/TASK_2_SUMMARY.md")
    arc_summary = read_optional("results/arc_bridge/TASK_4_ARC_BRIDGE_SUMMARY.md")

    path = out_dir / f"coherence-report-{today}.md"
    path.write_text(
        "\n".join(
            [
                f"# Weekly Coherence Report — {today}",
                "",
                "Automated weekly coherence report.",
                "",
                "## Diagnostic Self-Test",
                "",
                diagnostic_summary,
                "",
                "## ARC Bridge",
                "",
                arc_summary,
                "",
                "## Guardrail",
                "",
                "This report summarizes repository coherence only.",
                "It does not prove consciousness, completed AGI, or the Lattice Law as physics.",
                "",
            ]
        )
    )

    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
