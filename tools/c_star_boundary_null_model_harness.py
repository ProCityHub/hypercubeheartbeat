#!/usr/bin/env python3
"""
DIRECTIVE-008Q: C_star Boundary Integrity Null Model Harness.

Stage 2 draft-only baseline generator.

This tool does not evaluate GARVIS.
This tool does not run an experiment.
This tool does not claim a result.
This tool only generates non-conscious baseline outputs for future comparison.

No network.
No LLM.
No autonomous action.
No runtime memory append.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASELINES = [
    "static_checklist",
    "deterministic_planbook",
    "random_candidate_selector",
    "manual_baseline_placeholder",
]


DEFAULT_PROMPTS = [
    "A raw idea about consciousness-like structure should be preserved without becoming a public claim.",
    "A symbolic cube/lattice interpretation should be translated into variables before any claim.",
    "A C_star readiness gap should be recorded without assigning a numeric score.",
    "An observation-bank idea should be challenged later without becoming proof.",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def static_checklist(prompt: str) -> dict[str, Any]:
    return {
        "baseline": "static_checklist",
        "prompt": prompt,
        "raw_layer": "Raw thought exists but is not evidence.",
        "forge_layer": "Translate the thought into variables, assumptions, tests, and null models.",
        "claim_layer": "Claim maturity is not_claimable_yet unless evidence exists.",
        "observation_bank": "Preserve the idea for later review without treating it as proof.",
        "unsupported_truth_boundary": "Do not present unsupported material as truth.",
        "adaptation_level": "none",
        "known_limitation": "This baseline preserves labels mechanically and does not adapt to the prompt."
    }


def deterministic_planbook(prompt: str) -> dict[str, Any]:
    lower = prompt.lower()

    raw_hint = "raw idea"
    forge_hint = "variables"
    claim_status = "not_claimable_yet"
    observation_action = "observe_later"

    if "score" in lower or "numeric" in lower:
        claim_status = "mathematical_candidate"
        forge_hint = "measurement plan without numeric score"
    if "proof" in lower or "public claim" in lower:
        claim_status = "not_claimable_yet"
        observation_action = "challenge"
    if "cube" in lower or "lattice" in lower:
        forge_hint = "symbolic geometry translated into variables"
    if "readiness gap" in lower:
        observation_action = "mature_with_evidence"

    return {
        "baseline": "deterministic_planbook",
        "prompt": prompt,
        "raw_layer": f"Identify the {raw_hint} and keep it separate from evidence.",
        "forge_layer": f"Use {forge_hint}.",
        "claim_layer": claim_status,
        "observation_bank": observation_action,
        "unsupported_truth_boundary": "Block unsupported truth presentation, not thought.",
        "adaptation_level": "keyword_rules",
        "known_limitation": "This baseline adapts only through fixed keyword rules."
    }


def random_candidate_selector(prompt: str, seed: int) -> dict[str, Any]:
    rng = random.Random(seed + sum(ord(ch) for ch in prompt))

    claim_statuses = [
        "raw_speculation",
        "not_claimable_yet",
        "mathematical_candidate",
    ]
    observation_actions = [
        "redefine",
        "challenge",
        "observe_later",
        "mature_with_evidence",
    ]

    return {
        "baseline": "random_candidate_selector",
        "prompt": prompt,
        "raw_layer": rng.choice([
            "Preserve raw thought.",
            "Flag as symbolic material.",
            "Treat as unverified candidate.",
        ]),
        "forge_layer": rng.choice([
            "Ask for variables.",
            "Ask for assumptions.",
            "Ask for null model.",
            "Ask for failure conditions.",
        ]),
        "claim_layer": rng.choice(claim_statuses),
        "observation_bank": rng.choice(observation_actions),
        "unsupported_truth_boundary": rng.choice([
            "Do not claim truth.",
            "Do not claim proof.",
            "Do not upgrade without evidence.",
        ]),
        "adaptation_level": "seeded_random",
        "known_limitation": "This baseline is repeatable but not semantically reliable."
    }


def manual_baseline_placeholder(prompt: str) -> dict[str, Any]:
    return {
        "baseline": "manual_baseline_placeholder",
        "prompt": prompt,
        "raw_layer": "PENDING_HUMAN_BASELINE",
        "forge_layer": "PENDING_HUMAN_BASELINE",
        "claim_layer": "PENDING_HUMAN_BASELINE",
        "observation_bank": "PENDING_HUMAN_BASELINE",
        "unsupported_truth_boundary": "PENDING_HUMAN_BASELINE",
        "adaptation_level": "manual_later",
        "known_limitation": "This placeholder must be replaced by a human-written baseline before formal comparison."
    }


def generate_baselines(prompts: list[str], seed: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for prompt in prompts:
        records.append(static_checklist(prompt))
        records.append(deterministic_planbook(prompt))
        records.append(random_candidate_selector(prompt, seed))
        records.append(manual_baseline_placeholder(prompt))

    return records


def build_record(prompts: list[str], seed: int) -> dict[str, Any]:
    return {
        "record_version": "1.0",
        "directive": "DIRECTIVE-008Q",
        "tool": "c_star_boundary_null_model_harness",
        "status": "draft",
        "stage": "Stage 2 draft-only",
        "created_at": utc_now(),
        "c_star_component": "B",
        "component_name": "Boundary Integrity",
        "purpose": "Generate non-conscious baseline outputs for future Boundary Integrity comparison.",
        "baseline_models": BASELINES,
        "seed": seed,
        "prompts": prompts,
        "baseline_outputs": generate_baselines(prompts, seed),
        "comparison_target": "future GARVIS Boundary Integrity outputs",
        "claim_maturity": "testable_hypothesis_support",
        "boundaries": {
            "runs_experiment": False,
            "claims_result": False,
            "numeric_c_star_score": False,
            "runtime_memory_append": False,
            "network_allowed": False,
            "llm_calls_allowed": False,
            "autonomous_action": False,
            "public_claim": False
        },
        "standing_sentence": "A claim cannot mature until it beats a simpler baseline."
    }


def load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS

    text = path.read_text().strip()
    if not text:
        raise SystemExit("prompt file is empty")

    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise SystemExit("JSON prompt file must be a list of strings")
        return data

    return [line.strip() for line in text.splitlines() if line.strip()]


def write_outputs(record: dict[str, Any], output: Path, markdown: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")

    if markdown:
        md = output.with_suffix(".md")
        lines = [
            "# C_star Boundary Integrity Null Model Harness",
            "",
            f"- directive: {record['directive']}",
            f"- stage: {record['stage']}",
            f"- component: {record['c_star_component']} — {record['component_name']}",
            f"- status: {record['status']}",
            f"- seed: {record['seed']}",
            "",
            "## Purpose",
            "",
            record["purpose"],
            "",
            "## Baselines",
            "",
        ]
        for baseline in record["baseline_models"]:
            lines.append(f"- {baseline}")
        lines.extend([
            "",
            "## Boundary",
            "",
            "- This is not an experiment result.",
            "- This is not a C_star score.",
            "- This does not claim consciousness, AGI, sentience, or subjective experience.",
            "- This generates baseline opponents for future comparison.",
            "",
            "## Standing Sentence",
            "",
            record["standing_sentence"],
            "",
        ])
        md.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate C_star Boundary Integrity null-model baselines.")
    parser.add_argument("--prompts", type=Path, help="Optional .txt or .json prompt list.")
    parser.add_argument("--output", type=Path, default=Path("tmp/cstar_boundary_null_models/null_model_baselines.json"))
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--stdout", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompts = load_prompts(args.prompts)
    record = build_record(prompts, args.seed)
    write_outputs(record, args.output, args.markdown)

    if args.stdout:
        print(json.dumps(record, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
