#!/usr/bin/env python3
"""
Lattice Diagnostic Report

Autonomy Directive Task 1.

Purpose:
Generate an honest diagnostic report for the Lattice Brain / Hypercube Heartbeat
without claiming consciousness or completed AGI.

Core rule:
Sparse firing is not automatically Bridge failure.
The report must distinguish substrate-made limits from formula-made ceilings.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import json


REQUIRED_FIELDS = [
    "observer",
    "actor",
    "bridge",
    "cube_corner",
    "heartbeat_phase",
    "coherence_score",
    "action",
    "habit_signatures",
    "limiting_constraint",
    "limit_origin",
    "warnings",
]


@dataclass(frozen=True)
class DiagnosticReport:
    observer: float
    actor: float
    bridge: float
    cube_corner: str
    heartbeat_phase: str
    coherence_score: float
    action: str
    habit_signatures: dict[str, int]
    limiting_constraint: str
    limit_origin: str
    warnings: list[str]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _limiting_constraint(observer: float, actor: float, bridge: float) -> str:
    values = {
        "observer": observer,
        "actor": actor,
        "bridge": bridge,
    }
    return min(values, key=values.get)


def _history_warning(history_length: int) -> list[str]:
    if history_length <= 0:
        return ["insufficient evidence: no state history recorded yet"]
    if history_length < 3:
        return ["insufficient evidence: state history too short for strong diagnosis"]
    return []


def classify_limit_origin(
    *,
    limiting_constraint: str,
    timing_coherence: float | None,
    information_coupling: float | None,
    integration_pressure: float | None,
    history_length: int,
) -> tuple[str, list[str]]:
    warnings = _history_warning(history_length)

    if limiting_constraint != "bridge":
        return "not_bridge_limited", warnings

    if (
        timing_coherence is None
        or information_coupling is None
        or integration_pressure is None
    ):
        warnings.append("insufficient evidence: missing v5 bridge diagnostics")
        return "insufficient_evidence", warnings

    timing = _clamp01(timing_coherence)
    coupling = _clamp01(information_coupling)
    pressure = _clamp01(integration_pressure)

    if history_length < 3:
        return "insufficient_evidence", warnings

    if timing < 0.35 and coupling >= 0.55 and pressure >= 0.55:
        warnings.append(
            "sparse timing detected, but coupling/integration remain strong; "
            "do not treat this as automatic Bridge failure"
        )
        return "formula_made_possible", warnings

    if timing < 0.35 and coupling < 0.35 and pressure < 0.35:
        return "substrate_made_possible", warnings

    warnings.append("mixed evidence: Bridge limit requires more history")
    return "uncertain", warnings


def build_diagnostic_report(
    *,
    observer: float,
    actor: float,
    bridge: float,
    cube_corner: str,
    heartbeat_phase: str,
    coherence_score: float,
    action: str,
    habit_signatures: dict[str, int] | None = None,
    history_length: int = 0,
    timing_coherence: float | None = None,
    information_coupling: float | None = None,
    integration_pressure: float | None = None,
) -> DiagnosticReport:
    observer = _clamp01(observer)
    actor = _clamp01(actor)
    bridge = _clamp01(bridge)
    coherence_score = _clamp01(coherence_score)

    habits = habit_signatures or {}
    limit = _limiting_constraint(observer, actor, bridge)

    limit_origin, warnings = classify_limit_origin(
        limiting_constraint=limit,
        timing_coherence=timing_coherence,
        information_coupling=information_coupling,
        integration_pressure=integration_pressure,
        history_length=history_length,
    )

    evidence = {
        "history_length": history_length,
        "timing_coherence": timing_coherence,
        "information_coupling": information_coupling,
        "integration_pressure": integration_pressure,
        "claim_guardrail": (
            "diagnostic report only; does not prove consciousness or completed AGI"
        ),
    }

    return DiagnosticReport(
        observer=observer,
        actor=actor,
        bridge=bridge,
        cube_corner=cube_corner,
        heartbeat_phase=heartbeat_phase,
        coherence_score=coherence_score,
        action=action,
        habit_signatures=habits,
        limiting_constraint=limit,
        limit_origin=limit_origin,
        warnings=warnings,
        evidence=evidence,
    )


def demo() -> None:
    report = build_diagnostic_report(
        observer=0.72,
        actor=0.68,
        bridge=0.22,
        cube_corner="110",
        heartbeat_phase="bridge_pause",
        coherence_score=0.41,
        action="bounce",
        habit_signatures={"24113320": 5},
        history_length=8,
        timing_coherence=0.18,
        information_coupling=0.72,
        integration_pressure=0.69,
    )
    print(report.to_json())


if __name__ == "__main__":
    demo()
