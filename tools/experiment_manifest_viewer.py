#!/usr/bin/env python3
"""
GARVIS Experiment Manifest Viewer / Validator CLI.

DIRECTIVE-007B.

Stage 2 dry-run validation instrument.

This tool reads an experiment manifest JSON, validates the required v1
scientific-method boundaries, and prints what a future approved run would do.

It does not:
- execute method commands
- call subprocess
- call a network
- call an LLM
- write files
- approve actions
- create results
- upgrade claims
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Sequence


DEFAULT_SCHEMA = "ai_infrastructure/schemas/experiment_manifest_schema_v1.json"

CLAIM_GRADES = {"exploratory", "suggestive", "supported", "retracted"}

TOP_LEVEL_REQUIRED = [
    "manifest_id",
    "manifest_version",
    "status",
    "stage",
    "pre_registration",
    "hypothesis",
    "prediction",
    "counter_prediction",
    "null_model",
    "data_needed",
    "method",
    "failure_conditions",
    "claim_boundary",
    "dry_run_boundary",
    "approval",
    "ledger_chain",
    "safety",
]


class ManifestError(RuntimeError):
    """Safe user-facing manifest failure."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ManifestError(f"file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ManifestError("manifest root must be a JSON object")
    return data


def require_object(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ManifestError(f"{key} must be an object")
    return value


def require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{key} must be a non-empty string")
    return value


def require_array(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ManifestError(f"{key} must be an array")
    return value


def validate_top_level(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in TOP_LEVEL_REQUIRED:
        if key not in manifest:
            errors.append(f"missing required field: {key}")

    extras = sorted(set(manifest) - set(TOP_LEVEL_REQUIRED))
    for key in extras:
        errors.append(f"unexpected top-level field: {key}")

    return errors


def validate_manifest(manifest: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors = validate_top_level(manifest)
    warnings: list[str] = []

    if errors:
        return errors, warnings

    if manifest.get("manifest_version") != "1.0":
        errors.append("manifest_version must be 1.0")

    if manifest.get("stage") != "Stage 2 draft-only":
        errors.append("stage must be Stage 2 draft-only")

    if manifest.get("status") not in {
        "draft",
        "pre_registered",
        "retired",
        "superseded",
    }:
        errors.append(
            "status must be draft, pre_registered, retired, or superseded"
        )

    if manifest.get("status") == "superseded":
        warnings.append(
            "manifest is superseded; it cannot authorize a future execution"
        )

    pre = require_object(manifest, "pre_registration")
    if pre.get("result_must_cite_manifest_sha") is not True:
        errors.append("pre_registration.result_must_cite_manifest_sha must be true")
    if pre.get("manifest_must_predate_run") is not True:
        errors.append("pre_registration.manifest_must_predate_run must be true")
    if pre.get("result_invalid_if_manifest_postdates_run") is not True:
        errors.append("pre_registration.result_invalid_if_manifest_postdates_run must be true")
    if pre.get("manifest_sha_status") not in {"pending_commit", "locked"}:
        errors.append("pre_registration.manifest_sha_status must be pending_commit or locked")

    manifest_sha = pre.get("manifest_commit_sha")
    if pre.get("manifest_sha_status") != "locked" or manifest_sha in (None, ""):
        warnings.append("manifest commit SHA is not locked; this cannot support a valid execution result yet")

    for section in ["hypothesis", "prediction", "counter_prediction", "null_model"]:
        require_object(manifest, section)

    null_model = require_object(manifest, "null_model")
    for key in [
        "description",
        "expected_noise_behavior",
        "false_positive_risk",
        "comparison_method",
        "sample_size_or_trials",
    ]:
        require_string(null_model, key)

    method = require_object(manifest, "method")
    command = require_array(method, "would_run_command")
    if not command or not all(isinstance(item, str) and item for item in command):
        errors.append("method.would_run_command must be a non-empty array of command tokens")

    claim = require_object(manifest, "claim_boundary")
    allowed = set(require_array(claim, "allowed_result_grades"))
    if allowed != CLAIM_GRADES:
        errors.append("claim_boundary.allowed_result_grades must be exactly exploratory, suggestive, supported, retracted")
    if claim.get("maximum_claim_grade_without_new_manifest") not in CLAIM_GRADES:
        errors.append("claim_boundary.maximum_claim_grade_without_new_manifest must be a fixed claim grade")
    if claim.get("upgrade_requires_new_manifest") is not True:
        errors.append("claim_boundary.upgrade_requires_new_manifest must be true")

    dry = require_object(manifest, "dry_run_boundary")
    if dry.get("can_execute_method_script") is not False:
        errors.append("dry_run_boundary.can_execute_method_script must be false")
    if dry.get("prints_would_run_command_only") is not True:
        errors.append("dry_run_boundary.prints_would_run_command_only must be true")
    if dry.get("execution_stage_required") != "Stage 3 approved execution":
        errors.append("dry_run_boundary.execution_stage_required must be Stage 3 approved execution")
    if dry.get("approval_required_before_execution") is not True:
        errors.append("dry_run_boundary.approval_required_before_execution must be true")

    approval = require_object(manifest, "approval")
    if approval.get("approval_status") != "approved" or approval.get("approval_ledger_id") is None:
        warnings.append("execution approval is missing; this manifest is validation-only")

    ledger = require_object(manifest, "ledger_chain")
    if not ledger.get("manifest_commit_sha"):
        warnings.append("ledger chain has no manifest_commit_sha")
    if ledger.get("approval_ledger_id") is None:
        warnings.append("ledger chain has no approval_ledger_id")
    if not ledger.get("run_id"):
        warnings.append("ledger chain has no run_id")
    if not ledger.get("result_id"):
        warnings.append("ledger chain has no result_id")
    if not ledger.get("claim_record_id"):
        warnings.append("ledger chain has no claim_record_id")

    safety = require_object(manifest, "safety")
    for key in [
        "network_allowed",
        "llm_calls_allowed",
        "external_contact_allowed",
        "secret_access_allowed",
        "raw_sensor_payload_access_allowed",
    ]:
        if safety.get(key) is not False:
            errors.append(f"safety.{key} must be false")

    return errors, warnings


def section(manifest: dict[str, Any], key: str) -> dict[str, Any]:
    value = manifest.get(key)
    return value if isinstance(value, dict) else {}


def command_text(manifest: dict[str, Any]) -> str:
    method = section(manifest, "method")
    command = method.get("would_run_command")
    if not isinstance(command, list) or not command:
        return "(missing or invalid would-run command)"
    if not all(isinstance(part, str) and part for part in command):
        return "(missing or invalid would-run command)"
    return shlex.join(command)


def render_manifest(manifest: dict[str, Any], schema_path: Path, errors: list[str], warnings: list[str]) -> str:
    hypothesis = section(manifest, "hypothesis")
    prediction = section(manifest, "prediction")
    counter_prediction = section(manifest, "counter_prediction")
    null_model = section(manifest, "null_model")
    method = section(manifest, "method")
    claim = section(manifest, "claim_boundary")
    pre = section(manifest, "pre_registration")
    approval = section(manifest, "approval")

    lines: list[str] = []
    lines.append("# GARVIS Experiment Manifest Viewer")
    lines.append("")
    lines.append("mode: validation-only")
    lines.append("stage: Stage 2 dry-run validation")
    lines.append("execution: blocked")
    lines.append("network_calls: none")
    lines.append("llm_calls: none")
    lines.append("writes: none")
    lines.append(f"schema: {schema_path.as_posix()}")
    lines.append("")
    lines.append("## Validation")
    lines.append("")
    lines.append(f"- status: {'PASS' if not errors else 'FAIL'}")
    for error in errors:
        lines.append(f"- error: {error}")
    for warning in warnings:
        lines.append(f"- warning: {warning}")
    lines.append("")
    lines.append("## Manifest")
    lines.append("")
    lines.append(f"- manifest_id: {manifest.get('manifest_id')}")
    lines.append(f"- manifest_version: {manifest.get('manifest_version')}")
    lines.append(f"- status: {manifest.get('status')}")
    lines.append(f"- manifest_sha_status: {pre.get('manifest_sha_status')}")
    lines.append(f"- manifest_commit_sha: {pre.get('manifest_commit_sha') or 'pending'}")
    lines.append(f"- approval_status: {approval.get('approval_status')}")
    lines.append(f"- approval_ledger_id: {approval.get('approval_ledger_id')}")
    lines.append("")
    lines.append("## Hypothesis")
    lines.append("")
    lines.append(hypothesis.get("statement", ""))
    lines.append("")
    lines.append("## Prediction")
    lines.append("")
    lines.append(str(prediction.get("statement", "")))
    lines.append(f"measurable_signal: {prediction.get('measurable_signal', '')}")
    lines.append(f"success_threshold: {prediction.get('success_threshold', '')}")
    lines.append("")
    lines.append("## Counter-Prediction")
    lines.append("")
    lines.append(str(counter_prediction.get("statement", "")))
    lines.append(f"falsifying_signal: {counter_prediction.get('falsifying_signal', '')}")
    lines.append("")
    lines.append("## Null Model")
    lines.append("")
    lines.append(str(null_model.get("description", "")))
    lines.append(f"expected_noise_behavior: {null_model.get('expected_noise_behavior', '')}")
    lines.append(f"false_positive_risk: {null_model.get('false_positive_risk', '')}")
    lines.append(f"comparison_method: {null_model.get('comparison_method', '')}")
    lines.append(f"sample_size_or_trials: {null_model.get('sample_size_or_trials', '')}")
    lines.append("")
    lines.append("## Claim Boundary")
    lines.append("")
    lines.append(f"allowed_result_grades: {', '.join(claim.get('allowed_result_grades', []))}")
    lines.append(f"maximum_claim_grade_without_new_manifest: {claim.get('maximum_claim_grade_without_new_manifest')}")
    lines.append(f"upgrade_requires_new_manifest: {claim.get('upgrade_requires_new_manifest')}")
    lines.append("forbidden_claims:")
    for item in claim.get("forbidden_claims", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Would-Run Command")
    lines.append("")
    lines.append(command_text(manifest))
    lines.append("")
    lines.append("## Method Summary")
    lines.append("")
    lines.append(str(method.get("method_summary", "")))
    lines.append("")
    lines.append("## Boundary")
    lines.append("")
    lines.append("- This viewer cannot execute the would-run command.")
    lines.append("- Execution requires a separate Stage 3 approved event.")
    lines.append("- A result without manifest SHA, approval ledger ID, run ID, result ID, and claim record ID is not a valid scientific result.")
    lines.append("- A dry-run is not evidence.")
    lines.append("")

    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and display a GARVIS experiment manifest without executing it."
    )
    parser.add_argument("--repo", default=".", help="Repository root. Default: current directory.")
    parser.add_argument("--manifest", required=True, help="Path to experiment manifest JSON.")
    parser.add_argument(
        "--schema",
        default=DEFAULT_SCHEMA,
        help=f"Schema path for display/reference. Default: {DEFAULT_SCHEMA}.",
    )
    return parser.parse_args(argv)


def run(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    repo = Path(args.repo).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser()
    schema_path = Path(args.schema).expanduser()

    if not manifest_path.is_absolute():
        manifest_path = repo / manifest_path
    if not schema_path.is_absolute():
        schema_path = repo / schema_path

    try:
        if not schema_path.exists():
            raise ManifestError(f"schema not found: {schema_path}")

        manifest = load_json(manifest_path)
        errors, warnings = validate_manifest(manifest)
        print(render_manifest(manifest, schema_path, errors, warnings))
        return 0 if not errors else 2
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
