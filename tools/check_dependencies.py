#!/usr/bin/env python3
"""Repository dependency-integrity checker.

Author/concept authority: Adrien D. Thomas / ProCityHub.

The checker is intentionally conservative. It validates explicit repository-relative
references and stale 008M/008N references without interpreting prose as executable truth.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

TEXT_SUFFIXES = {
    ".py", ".md", ".jsonl", ".json", ".yml", ".yaml", ".toml", ".txt", ".qasm", ".csv", ".sh"
}
PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_./-])"
    r"((?:ai_infrastructure|app_infrastructure|tools|tests|docs|research|claims|reports|data|"
    r"\.github)/[A-Za-z0-9_.?*+\-/]+\.(?:py|md|jsonl|json|ya?ml|toml|txt|qasm|csv|sh))(?![A-Za-z0-9_])"
)
STALE_DIRECTIVES = ("008M", "008N")


def tracked_files(repo: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [repo / p.decode() for p in result.stdout.split(b"\0") if p]


def read_allowlist(repo: Path) -> set[str]:
    path = repo / ".dependency-allowlist"
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def is_literal_reference(value: str) -> bool:
    return not any(token in value for token in ("*", "?", "{", "}", "<", ">", "$"))


def scan(repo: Path) -> list[str]:
    issues: list[str] = []
    allow = read_allowlist(repo)
    tracked = tracked_files(repo)
    tracked_rel = {str(path.relative_to(repo)) for path in tracked}

    for path in tracked:
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        if path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        rel = str(path.relative_to(repo))
        for match in PATH_PATTERN.finditer(text):
            reference = match.group(1).rstrip(".,:;)'\"`")
            key = f"{rel}:{reference}"
            if key in allow or reference in allow or not is_literal_reference(reference):
                continue
            # Generated Python caches are not repository dependencies.
            if "__pycache__" in Path(reference).parts or reference.endswith(".pyc"):
                continue

            # References may be repository-relative or relative to a local
            # manifest/package directory.
            repo_candidate = repo / reference
            source_candidate = path.parent / reference
            if (
                reference not in tracked_rel
                and not repo_candidate.exists()
                and not source_candidate.exists()
            ):
                issues.append(f"{rel}: missing referenced path: {reference}")

        # 008M/008N may remain in historical records, but a filename-like reference must resolve.
        for directive in STALE_DIRECTIVES:
            for candidate in re.findall(
                rf"[A-Za-z0-9_./-]*{directive}[A-Za-z0-9_./-]*\.(?:md|json|py|yml|yaml)",
                text,
            ):
                candidate = candidate.lstrip("./")
                key = f"{rel}:{candidate}"
                if key in allow or candidate in allow:
                    continue
                if (
                    candidate not in tracked_rel
                    and not (repo / candidate).exists()
                    and not (path.parent / candidate).exists()
                ):
                    issues.append(f"{rel}: stale {directive} file reference: {candidate}")

    return sorted(set(issues))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    repo = args.repo.resolve()

    try:
        issues = scan(repo)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"DEPENDENCY CHECK ERROR: {exc}", file=sys.stderr)
        return 2

    if issues:
        print("DEPENDENCY CHECK FAILED")
        for issue in issues:
            print(f"- {issue}")
        print("\nAdd intentional historical references to .dependency-allowlist.")
        return 1

    print("DEPENDENCY CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
