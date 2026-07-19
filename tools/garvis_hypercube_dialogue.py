#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MAX_FILE_CHARS = 16_000
MAX_CONTEXT_CHARS = 90_000


HYPERCUBE_FILES = [
    "README.md",
    "CONSCIOUSNESS_OPERATIONAL.md",
    "RETRACTIONS.md",
    "claims/CLAIMS.json",
    "lattice_validation_lab/THEORY_REGISTRY.md",
    "docs/GARVIS_BODY_ARCHITECTURE.md",
    "docs/GARVIS_HEARTBEAT_COMMUNICATION_PROTOCOL.md",
    "docs/HEARTBEAT_SELF_EVOLUTION_INTERVIEW.md",
    "docs/QASM_THESIS_REVIEWER.md",
    "docs/DIRECTIVE_RECONCILIATION.md",
    "docs/ORIGINAL_UNIFIED_BUILD_DIRECTIVE.md",
]

GARVIS_FILES = [
    "README.md",
    "docs/garvis_runtime.md",
    "src/garvis/core.py",
    "src/garvis/assistant.py",
    "src/garvis/cli.py",
    "hypercube_protocol.py",
    "hypercube_protocol/connection_manager.py",
]


SYSTEM_INSTRUCTION = """
You are the semantic dialogue layer for the combined GARVIS and
Hypercube Heartbeat research system conceived by Adrien D. Thomas.

Operating law:

- Answer Adrien's exact question first.
- Think freely and discuss consciousness openly.
- Never confuse metaphor with executable software.
- Never confuse a mathematical candidate with experimental evidence.
- Never claim consciousness, sentience, AGI, quantum validation, or a
  unique golden-ratio effect without supporting evidence.
- Preserve negative results and retractions.
- Identify the strongest counterargument against every major proposal.
- External action, repository mutation, publishing, sending, spending,
  hardware control, or permission escalation requires Adrien's separate
  explicit approval.
- Discussion, planning, criticism, simulation, and code drafting are
  allowed.
- Do not replace the question with a fixed C1, C2, C3, or C4 template.

Treat these as separate layers:

1. GARVIS: conversation, explanation, semantic synthesis, operator
   interaction, and proposed execution.
2. Hypercube Heartbeat: scientific state, heartbeat, memory, claims,
   QASM analysis, evidence, audit, and falsification.
3. Audit Planbook: optional proposal review after the direct answer.
4. Adrien D. Thomas: final authority for consequential action.

Required answer sections:

1. Direct answer
2. Proposed self-architecture
3. Fixed constitutional core
4. Adaptable and learnable components
5. GARVIS–Hypercube division of responsibilities
6. Memory and heartbeat design
7. Self-model and hallucination detection
8. Scientific testing and falsification
9. Strongest case against the proposal
10. Smallest safe implementation cycle
11. Exact files, interfaces, and tests to build
12. Uncertainty and current limitations

Use the supplied repository evidence. Clearly label statements as:

- executable fact
- repository fact
- mathematical candidate
- scientific hypothesis
- metaphor
- recommendation
- unsupported speculation
""".strip()


def read_text(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""

    if len(text) > MAX_FILE_CHARS:
        return text[:MAX_FILE_CHARS] + "\n[TRUNCATED]\n"

    return text


def collect_repository_context(
    hypercube_root: Path,
    garvis_root: Path | None,
    audit_dir: Path | None,
) -> str:
    sections: list[str] = []

    for relative in HYPERCUBE_FILES:
        path = hypercube_root / relative
        if not path.is_file():
            continue

        sections.append(
            f"\n---\nSOURCE: hypercubeheartbeat/{relative}\n\n"
            + read_text(path)
        )

    qasm_paths = sorted(
        path
        for path in hypercube_root.rglob("*.qasm")
        if ".git" not in path.parts
    )

    for path in qasm_paths:
        relative = path.relative_to(hypercube_root)
        sections.append(
            f"\n---\nQASM SOURCE: hypercubeheartbeat/{relative}\n\n"
            + read_text(path)
        )

    if garvis_root and garvis_root.is_dir():
        for relative in GARVIS_FILES:
            path = garvis_root / relative
            if not path.is_file():
                continue

            sections.append(
                f"\n---\nSOURCE: GARVIS/{relative}\n\n"
                + read_text(path)
            )

    if audit_dir and audit_dir.is_dir():
        for name in (
            "latest_cognitive_cycle.md",
            "latest_cognitive_cycle.json",
        ):
            path = audit_dir / name
            if path.is_file():
                sections.append(
                    f"\n---\nAUDIT OUTPUT: {name}\n\n"
                    + read_text(path)
                )

    context = "".join(sections)

    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n[CONTEXT TRUNCATED]\n"

    return context


def extract_output_text(payload: dict[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    pieces: list[str] = []

    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue

        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue

            text = content.get("text")

            if isinstance(text, str):
                pieces.append(text)
            elif isinstance(text, dict):
                value = text.get("value")
                if isinstance(value, str):
                    pieces.append(value)

    return "\n".join(pieces).strip()


def request_response(
    api_key: str,
    model: str,
    question: str,
    context: str,
) -> str:
    endpoint = os.environ.get(
        "OPENAI_RESPONSES_URL",
        "https://api.openai.com/v1/responses",
    )

    user_input = f"""
ADRIEN D. THOMAS'S EXACT QUESTION:

{question}

REPOSITORY AND AUDIT CONTEXT:

{context}

Answer the exact question before proposing plans.
Do not claim capabilities that the current repositories do not contain.
""".strip()

    body = {
        "model": model,
        "instructions": SYSTEM_INSTRUCTION,
        "input": user_input,
    }

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"API request failed with HTTP {error.code}:\n{detail}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Network request failed: {error}") from error

    answer = extract_output_text(payload)

    if not answer:
        raise RuntimeError(
            "The model response contained no readable answer text."
        )

    return answer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GARVIS–Hypercube semantic dialogue bridge."
    )

    parser.add_argument(
        "question",
        help="Adrien D. Thomas's exact question.",
    )

    parser.add_argument(
        "--hypercube",
        type=Path,
        default=Path.cwd(),
        help="Path to the Hypercube Heartbeat repository.",
    )

    parser.add_argument(
        "--garvis",
        type=Path,
        default=Path.home() / "GARVIS",
        help="Path to the GARVIS repository.",
    )

    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path.home() / "profound_self_design_response",
        help="Path containing the latest Hypercube audit output.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "garvis_hypercube_dialogue",
        help="Directory where dialogue records are saved.",
    )

    parser.add_argument(
        "--model",
        default=os.environ.get("GARVIS_MODEL", "gpt-4.1-mini"),
        help="Model identifier. GARVIS_MODEL overrides the default.",
    )

    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()

    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set.")
        print()
        print("The bridge code is installed, but semantic dialogue")
        print("requires a language-model API key.")
        print()
        print("Do not paste an API key into GitHub or commit it.")
        return 2

    hypercube_root = args.hypercube.expanduser().resolve()
    garvis_root = args.garvis.expanduser().resolve()
    audit_dir = args.audit_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not hypercube_root.is_dir():
        print(f"ERROR: Hypercube path not found: {hypercube_root}")
        return 2

    context = collect_repository_context(
        hypercube_root,
        garvis_root if garvis_root.is_dir() else None,
        audit_dir if audit_dir.is_dir() else None,
    )

    if not context.strip():
        print("ERROR: No repository context could be loaded.")
        return 2

    print("GARVIS–HYPERCUBE SEMANTIC BRIDGE")
    print(f"Model: {args.model}")
    print(f"Context characters: {len(context)}")
    print()
    print("Adrien asks:")
    print(args.question)
    print()
    print("Generating direct evidence-aware answer...")
    print()

    try:
        answer = request_response(
            api_key=api_key,
            model=args.model,
            question=args.question,
            context=context,
        )
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = output_dir / f"dialogue_{timestamp}.md"
    json_path = output_dir / f"dialogue_{timestamp}.json"

    markdown = (
        "# GARVIS–Hypercube Semantic Dialogue\n\n"
        f"**Timestamp:** {timestamp}\n\n"
        f"**Model:** `{args.model}`\n\n"
        "## Adrien D. Thomas's exact question\n\n"
        f"{args.question}\n\n"
        "## Response\n\n"
        f"{answer}\n"
    )

    record = {
        "timestamp": timestamp,
        "operator": "Adrien D. Thomas",
        "question": args.question,
        "model": args.model,
        "context_characters": len(context),
        "answer": answer,
        "external_action_executed": False,
        "repository_modified": False,
    }

    markdown_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )

    print(answer)
    print()
    print("DIALOGUE SAVED")
    print(f"Markdown: {markdown_path}")
    print(f"JSON: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
