#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path


def main() -> None:
    today = dt.date.today().isoformat()
    commit = os.environ.get("GITHUB_SHA", "local")
    out_dir = Path("reports/heartbeat")
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"heartbeat-{today}.md"
    path.write_text(
        "\n".join(
            [
                f"# Heartbeat Report — {today}",
                "",
                "Automated daily repository heartbeat.",
                "",
                f"- Commit: `{commit}`",
                "- Smoke tests: run by workflow before this report is written",
                "- Claim guardrail: no consciousness or completed AGI claim is made",
                "- Empirical status: pre-registered AUC pilot remains separate from software self-tests",
                "",
            ]
        )
    )

    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
