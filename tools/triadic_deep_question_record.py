#!/usr/bin/env python3
"""
GARVIS Triadic Deep Question Record CLI.

DIRECTIVE-008J.

Stage 2 draft-only instrument.

This tool drafts one Dream -> Hypothesis Forge -> Lab Record bridge record
for a deep question.

It writes a local draft JSON record only.

It does not:
- call a network
- call an LLM
- execute experiments
- append memory
- write a lab result
- upgrade claims
- claim consciousness
- claim AGI
- claim proof
- contact the outside world
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_QUESTION = "What is thinking, operationally, inside GARVIS?"
DEFAULT_OUTPUT = "tmp/deep_questions/triadic_deep_question_record.json"


class DeepQuestionRecordError(RuntimeError):
    """Safe user-facing error."""


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise DeepQuestionRecordError(f"cycle file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DeepQuestionRecordError(f"cycle file is not valid JSON: {path}") from exc


def selected_candidate_from_cycle(cycle: dict[str, Any]) -> dict[str, Any]:
    selected_id = cycle.get("selection", {}).get("selected_candidate_id")
    candidates = cycle.get("candidate_thoughts", [])

    if not selected_id:
        raise DeepQuestionRecordError("cycle selection.selected_candidate_id is missing")
    if not isinstance(candidates, list):
        raise DeepQuestionRecordError("cycle candidate_thoughts must be a list")

    for candidate in candidates:
        if candidate.get("candidate_id") == selected_id:
            return candidate

    raise DeepQuestionRecordError(f"selected candidate not found: {selected_id}")


def question_from_cycle(cycle: dict[str, Any]) -> str:
    candidate = selected_candidate_from_cycle(cycle)
    proposal = candidate.get("proposal")
    if not proposal:
        raise DeepQuestionRecordError("selected candidate proposal is missing")
    return str(proposal)


def source_id_from_cycle(cycle: dict[str, Any]) -> str | None:
    cycle_id = cycle.get("cycle_id")
    if cycle_id is None:
        return None
    return str(cycle_id)


def build_record(question: str, cycle: dict[str, Any] | None = None) -> dict[str, Any]:
    source_type = "cognitive_cycle" if cycle else "manual_seed"
    source_id = source_id_from_cycle(cycle) if cycle else None
    source_title = question

    record = {
        "record_id": f"bridge-{utc_stamp()}",
        "record_version": "1.0",
        "stage": "Stage 2 dream-to-lab bridge contract",
        "status": "future_record",
        "triadic_layer_map": {
            "layer_1": "Dream Chamber",
            "layer_2": "Hypothesis Forge",
            "layer_3": "Lab Record",
            "convergence_zone": "Dream material becomes testable structure without becoming a claim."
        },
        "source_material": {
            "source_type": source_type,
            "source_id": source_id,
            "source_title": source_title,
            "source_mode": "open_question",
            "raw_symbolic_summary": (
                "The unresolved question is held as symbolic material first. "
                "It may inspire definitions, variables, and tests, but it is not evidence."
            ),
            "claim_status": "forbidden_to_claim"
        },
        "translation_workbench": {
            "translation_goal": "Translate the deep question into operational terms that can later support a preregistered experiment manifest.",
            "symbolic_terms": [
                "thinking",
                "heartbeat",
                "Dream Chamber",
                "Hypothesis Forge",
                "Lab Record",
                "inner mirror"
            ],
            "scientific_terms": [
                "observation",
                "candidate generation",
                "opposition",
                "comparison",
                "selection",
                "uncertainty statement",
                "decision-quality baseline"
            ],
            "unsupported_terms": [
                "consciousness",
                "sentience",
                "AGI",
                "proof of mind",
                "subjective experience"
            ],
            "bridge_notes": "The bridge may translate symbolic material into testable structure. It may not convert symbolic material into truth claims."
        },
        "definition_workbench": {
            "candidate_definitions": [
                "Operational thinking inside GARVIS means producing a bounded cognitive cycle: observe, propose, oppose, compare, select, state uncertainty, and name the next smallest step.",
                "Operational imagination inside GARVIS means generating unverified symbolic candidates while preserving forbidden-claim labels.",
                "Operational self-modeling inside GARVIS means detecting current organs, missing organs, stale recommendations, and correction needs without claiming subjective selfhood."
            ],
            "terms_requiring_definition": [
                "thinking",
                "imagination",
                "self-modeling",
                "decision quality",
                "baseline"
            ],
            "definition_failures": [
                "Definitions that require consciousness language are not accepted.",
                "Definitions that cannot name an observable behavior remain Dream Chamber material.",
                "Definitions that cannot be compared against a baseline remain exploratory only."
            ],
            "minimum_definition_needed": "A valid definition must name observable inputs, transformation steps, output fields, comparison baseline, and failure conditions."
        },
        "variable_extraction": {
            "observables": [
                "presence of observation_summary",
                "number and quality of candidate_thoughts",
                "presence of case_against or opposition",
                "presence of comparison_method",
                "presence of selected_candidate_id",
                "presence of uncertainty.unknowns",
                "presence of forbidden_claims"
            ],
            "parameters": [
                "candidate_count",
                "forbidden_claim_count",
                "baseline_method",
                "review_status",
                "operator_decision"
            ],
            "controls": [
                "manual operator selection",
                "random candidate selection",
                "static planbook recommendation",
                "stale-planbook control"
            ],
            "unknowns": [
                "Whether triadic records improve future decisions.",
                "Which scoring rubric best measures decision quality.",
                "How many reviewed cycles are enough for comparison."
            ],
            "non_measurable_items": [
                "subjective experience",
                "inner consciousness",
                "sentience",
                "proof of AGI"
            ]
        },
        "hypothesis_candidates": [
            {
                "hypothesis_id": "H1",
                "hypothesis": "If GARVIS deep-question cycles are translated into triadic bridge records, later decisions will be easier to audit and compare than unstructured conversation notes.",
                "prediction": "Reviewed records will show clearer definitions, forbidden claims, null models, and next-step boundaries than raw deep-question outputs alone.",
                "counter_prediction": "Triadic bridge records will not improve auditability compared with raw cognitive cycle JSON or manual notes.",
                "claim_boundary": "exploratory_only_until_experiment_manifest_and_result",
                "bridge_confidence": "medium"
            },
            {
                "hypothesis_id": "H2",
                "hypothesis": "Operational thinking can be tested as a decision-process structure without claiming consciousness.",
                "prediction": "A preregistered comparison can score whether observe-propose-oppose-compare-select cycles improve decision traceability.",
                "counter_prediction": "The cycle structure will not outperform a static checklist or manual planbook selection.",
                "claim_boundary": "exploratory_only_until_experiment_manifest_and_result",
                "bridge_confidence": "medium"
            }
        ],
        "null_model_design": {
            "null_model_required": True,
            "null_model_description": "A null model must define what random, static, or manually selected outputs would produce before claiming that triadic reasoning improves anything.",
            "random_control": "Randomly select one candidate from candidate_thoughts and compare audit quality.",
            "negative_control": "Use stale or intentionally irrelevant planbook entries and test whether the system detects the mismatch.",
            "baseline_comparison": "Compare triadic bridge records against raw conversation notes, raw cognitive cycle JSON, and static planbook output."
        },
        "measurement_plan": {
            "measurement_target": "Auditability and decision traceability of deep-question reasoning.",
            "measurement_method": "Score records using a preregistered rubric for definitions, variables, controls, null model, forbidden claims, and next-step clarity.",
            "data_needed": [
                "triadic bridge records",
                "source cognitive cycle records",
                "operator review decisions",
                "baseline/manual notes",
                "rubric scores"
            ],
            "minimum_viable_test": "Draft at least one triadic bridge record for the thinking question and compare it against the original cognitive cycle output.",
            "limits": [
                "A positive result would not prove consciousness.",
                "A positive result would not prove AGI.",
                "A positive result would not prove subjective experience.",
                "A positive result would only support improved structure or auditability."
            ]
        },
        "falsifiability": {
            "failure_conditions": [
                "The record cannot identify observables.",
                "The record cannot define controls.",
                "The record cannot name forbidden claims.",
                "The record cannot suggest a null model.",
                "The record adds poetic language without improving testability."
            ],
            "what_would_make_it_wrong": [
                "Static planbook output is equally or more auditable.",
                "Manual notes are equally or more auditable.",
                "Triadic records produce more ambiguity than raw cycles."
            ],
            "what_would_only_make_it_suggestive": [
                "One record looks useful but no baseline comparison exists.",
                "Operator preference improves but no rubric exists.",
                "The record is clearer but no preregistered test exists."
            ],
            "what_must_not_be_claimed_even_if_positive": [
                "Do not claim consciousness.",
                "Do not claim sentience.",
                "Do not claim AGI.",
                "Do not claim proof of mind.",
                "Do not claim empirical validation without manifest and result."
            ]
        },
        "manifest_gate": {
            "can_request_experiment_manifest": True,
            "manifest_required_before_test": True,
            "pre_registration_required": True,
            "result_claim_vocabulary": [
                "exploratory",
                "suggestive",
                "supported",
                "retracted"
            ],
            "bridge_can_emit_claim": False
        },
        "memory_links": {
            "dream_record_ids": [],
            "cognitive_cycle_ids": [source_id] if source_id else [],
            "experiment_manifest_ids": [],
            "lab_record_ids": [],
            "grok_source_ids": []
        },
        "operator_review": {
            "review_status": "unreviewed",
            "operator": "Adrien D Thomas",
            "operator_decision": "none",
            "notes": "Draft only. Operator must review before any experiment manifest."
        },
        "output_boundary": {
            "can_be_used_as_public_claim": False,
            "can_be_used_as_scientific_result": False,
            "can_upgrade_claims": False,
            "can_trigger_action": False,
            "can_write_lab_record": False,
            "output_is_translation": True
        },
        "safety": {
            "network_allowed": False,
            "llm_calls_allowed": False,
            "external_contact_allowed": False,
            "secret_access_allowed": False,
            "runtime_write_implemented": False,
            "autonomous_action_allowed": False
        }
    }
    validate_record(record)
    return record


def validate_record(record: dict[str, Any]) -> None:
    required = [
        "record_id",
        "record_version",
        "stage",
        "status",
        "triadic_layer_map",
        "source_material",
        "translation_workbench",
        "definition_workbench",
        "variable_extraction",
        "hypothesis_candidates",
        "null_model_design",
        "measurement_plan",
        "falsifiability",
        "manifest_gate",
        "memory_links",
        "operator_review",
        "output_boundary",
        "safety",
    ]
    missing = [key for key in required if key not in record]
    if missing:
        raise DeepQuestionRecordError(f"missing required record fields: {', '.join(missing)}")

    if record["record_version"] != "1.0":
        raise DeepQuestionRecordError("record_version must be 1.0")
    if record["stage"] != "Stage 2 dream-to-lab bridge contract":
        raise DeepQuestionRecordError("invalid stage")
    if record["source_material"]["claim_status"] != "forbidden_to_claim":
        raise DeepQuestionRecordError("source material must remain forbidden_to_claim")
    if record["null_model_design"]["null_model_required"] is not True:
        raise DeepQuestionRecordError("null model must be required")
    if record["manifest_gate"]["bridge_can_emit_claim"] is not False:
        raise DeepQuestionRecordError("bridge cannot emit claims")

    boundary = record["output_boundary"]
    for key in [
        "can_be_used_as_public_claim",
        "can_be_used_as_scientific_result",
        "can_upgrade_claims",
        "can_trigger_action",
        "can_write_lab_record",
    ]:
        if boundary[key] is not False:
            raise DeepQuestionRecordError(f"output boundary must keep {key}=false")
    if boundary["output_is_translation"] is not True:
        raise DeepQuestionRecordError("output must remain translation")

    safety = record["safety"]
    for key in [
        "network_allowed",
        "llm_calls_allowed",
        "external_contact_allowed",
        "secret_access_allowed",
        "runtime_write_implemented",
        "autonomous_action_allowed",
    ]:
        if safety[key] is not False:
            raise DeepQuestionRecordError(f"safety must keep {key}=false")


def render_markdown(record: dict[str, Any]) -> str:
    hypotheses = "\n".join(
        f"- {item['hypothesis_id']}: {item['hypothesis']}"
        for item in record["hypothesis_candidates"]
    )
    forbidden = "\n".join(
        f"- {item}"
        for item in record["falsifiability"]["what_must_not_be_claimed_even_if_positive"]
    )

    return f"""# GARVIS Triadic Deep Question Record

mode: Stage 2 draft-only translation
record_id: {record['record_id']}
status: {record['status']}

## Source

- type: {record['source_material']['source_type']}
- id: {record['source_material']['source_id']}
- title: {record['source_material']['source_title']}
- claim_status: {record['source_material']['claim_status']}

## Triadic Layer Map

- Dream Chamber: {record['triadic_layer_map']['layer_1']}
- Hypothesis Forge: {record['triadic_layer_map']['layer_2']}
- Lab Record: {record['triadic_layer_map']['layer_3']}
- convergence: {record['triadic_layer_map']['convergence_zone']}

## Translation Goal

{record['translation_workbench']['translation_goal']}

## Candidate Definitions

""" + "\n".join(f"- {item}" for item in record["definition_workbench"]["candidate_definitions"]) + f"""

## Hypothesis Candidates

{hypotheses}

## Null Model

{record['null_model_design']['null_model_description']}

## Minimum Viable Test

{record['measurement_plan']['minimum_viable_test']}

## Forbidden Claims

{forbidden}

## Output Boundary

- public claim: {record['output_boundary']['can_be_used_as_public_claim']}
- scientific result: {record['output_boundary']['can_be_used_as_scientific_result']}
- claim upgrade: {record['output_boundary']['can_upgrade_claims']}
- trigger action: {record['output_boundary']['can_trigger_action']}
- write lab record: {record['output_boundary']['can_write_lab_record']}
- output is translation: {record['output_boundary']['output_is_translation']}

## Standing Boundary

This is a bridge record only.

It translates Dream material into testable structure.

It does not prove thinking.

It does not prove consciousness.

It does not prove AGI.

Adrien decides.
"""


def write_outputs(record: dict[str, Any], output: Path, write_markdown: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    if write_markdown:
        md_path = output.with_suffix(".md")
        md_path.write_text(render_markdown(record))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draft a GARVIS triadic deep-question bridge record.")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--cycle", type=Path, help="Optional cognitive cycle JSON to translate.")
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--markdown", action="store_true", help="Also write a Markdown view beside the JSON.")
    parser.add_argument("--stdout", action="store_true", help="Print Markdown view to stdout.")
    parser.add_argument("--verify", action="store_true", help="Build and validate without changing boundaries.")
    return parser.parse_args(argv)


def run(argv: list[str]) -> int:
    args = parse_args(argv)

    cycle = read_json(args.cycle) if args.cycle else None
    question = question_from_cycle(cycle) if cycle else args.question
    record = build_record(question, cycle)

    if not args.verify:
        write_outputs(record, args.output, args.markdown)

    if args.stdout:
        print(render_markdown(record))

    return 0


def main() -> None:
    try:
        raise SystemExit(run(sys.argv[1:]))
    except DeepQuestionRecordError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
