#!/usr/bin/env python3
"""
Repository guard checks.

Blocks:
1. Accidental edits to frozen files listed in FROZEN_FILES.txt.
2. Reintroduction of the retracted scalar-phi formula pattern.
3. Misuse of empirical support language outside the pre-registration record.

Self-test outputs must use PASS / FAIL.
SUPPORTED / NOT_SUPPORTED is reserved for real pre-registered data outcomes.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


BASE_REF = os.environ.get("GITHUB_BASE_REF", "main")


def run(cmd: list[str], *, check: bool = True) -> str:
    result = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)
    return result.stdout


def diff_base_ref() -> str:
    base = f"origin/{BASE_REF}"
    run(["git", "fetch", "origin", BASE_REF, "--depth=100"], check=False)

    probe = subprocess.run(
        ["git", "merge-base", base, "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if probe.returncode == 0 and probe.stdout.strip():
        return probe.stdout.strip()

    print(f"WARN: no merge base for {base}; using {base} directly")
    return base


def changed_files() -> list[str]:
    output = run(["git", "diff", "--name-only", diff_base_ref(), "HEAD"])
    return [line.strip() for line in output.splitlines() if line.strip()]


def parse_frozen_files(text: str) -> set[str]:
    frozen = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        frozen.add(stripped)
    return frozen


def read_current_frozen_files() -> set[str]:
    path = Path("FROZEN_FILES.txt")
    if not path.exists():
        return set()
    return parse_frozen_files(path.read_text())


def read_base_frozen_files() -> set[str]:
    result = subprocess.run(
        ["git", "show", f"origin/{BASE_REF}:FROZEN_FILES.txt"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return set()
    return parse_frozen_files(result.stdout)


def added_lines_for(path: str) -> list[str]:
    diff = run(["git", "diff", "-U0", diff_base_ref(), "HEAD", "--", path])
    lines = []
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            lines.append(line[1:])
    return lines


SCALAR_PHI_PATTERNS = [
    re.compile(
        r"\b(?:score|s)\s*=\s*(?:phi|PHI|φ)\s*[*×·]\s*"
        r"(?:o|observer)\s*[*×·]\s*(?:a|actor)\s*[*×·]\s*"
        r"(?:b|bridge|environment)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:phi|PHI|φ)\s*[*×·]\s*(?:o|observer)\s*[*×·]\s*"
        r"(?:a|actor)\s*[*×·]\s*(?:b|bridge|environment)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:o|observer)\s*[*×·]\s*(?:a|actor)\s*[*×·]\s*"
        r"(?:b|bridge|environment)\s*[*×·]\s*(?:phi|PHI|φ)\b",
        re.IGNORECASE,
    ),
]

EMPIRICAL_SUPPORT_WORDS = re.compile(r"\b(SUPPORTED|NOT_SUPPORTED)\b")


def is_claim_language_guard_path(path: str) -> bool:
    """Return True for claim-controlled documents.

    Reserved empirical verdict words are guarded in claims-ledger files
    and claim/protocol documents. Charters, maps, and exploratory notes
    may discuss vocabulary without being treated as empirical outcomes.
    """
    normalized = path.replace("\\", "/")
    name = Path(normalized).name.lower()

    if normalized.startswith("claims/"):
        return True

    if normalized.startswith("docs/") and (
        "claim" in name
        or "positive_result_protocol" in name
        or "outcome" in name
        or "success" in name
    ):
        return True

    return False


def is_text_path(path: str) -> bool:
    suffix = Path(path).suffix.lower()
    return suffix in {
        ".py",
        ".md",
        ".txt",
        ".yml",
        ".yaml",
        ".json",
        ".csv",
        ".toml",
    }


def main() -> None:
    changed = changed_files()
    # Only files already frozen on the base branch are blocked.
    # This permits a PR to create a new file and add it to FROZEN_FILES.txt
    # in the same PR, while still blocking later edits after merge.
    frozen = read_base_frozen_files()
    failures: list[str] = []

    frozen_changes = sorted(path for path in changed if path in frozen)
    if frozen_changes:
        failures.append(
            "Frozen files changed without explicit audit exception:\n"
            + "\n".join(f"  - {path}" for path in frozen_changes)
        )

    for path in changed:
        if not is_text_path(path):
            continue
        if not Path(path).exists():
            continue

        added = added_lines_for(path)

        for line in added:
            for pattern in SCALAR_PHI_PATTERNS:
                if pattern.search(line):
                    failures.append(
                        f"{path}: retracted scalar-phi formula pattern added: {line}"
                    )

        if is_claim_language_guard_path(path):
            for line in added:
                if EMPIRICAL_SUPPORT_WORDS.search(line):
                    failures.append(
                        f"{path}: reserved empirical verdict language added "
                        f"inside a claim-controlled document: {line}"
                    )

    if failures:
        print("GUARD FAILED")
        print()
        for failure in failures:
            print(failure)
            print()
        raise SystemExit(1)

    print("GUARD PASS")
    print(f"Checked {len(changed)} changed file(s).")


if __name__ == "__main__":
    main()
