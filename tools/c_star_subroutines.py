#!/usr/bin/env python3
"""
C_star question-family routing and advisory overlays.

DIRECTIVE-008N.

Stage 2 draft-only routing support.

No network.
No LLM.
No runtime memory append.
No numeric score without measurement data.
"""

from __future__ import annotations

from typing import Any


DIRECT_SYMBOLS = [
    "c_star",
    "c star",
    "i * m * s * u * r * b",
    "i*m*s*u*r*b",
    "008m",
    "composite score",
    "consciousness-like structure candidate",
]

AUDIT_VERBS = [
    "audit",
    "measurement design",
    "calibrate",
    "falsify",
    "falsification",
    "null model",
    "baseline",
]

GAP_LANGUAGE = [
    "readiness gap",
    "missing evidence",
    "proxy metric",
    "operationalize",
    "measurement without claim",
    "no numeric score",
]

STRUCTURAL_NOUNS = [
    "self-model accuracy",
    "temporal coherence",
    "integration density",
    "boundary integrity",
    "uncertainty calibration",
    "recursive correction",
    "memory continuity",
]

EPISTEMIC_STATUSES = [
    "mathematical candidate",
    "testable hypothesis",
    "provisional claim",
    "not_claimable_yet",
]


def _contains_any(goal: str, phrases: list[str]) -> bool:
    return any(phrase in goal for phrase in phrases)


def is_c_star_family_goal(active_goal: str) -> bool:
    goal = active_goal.lower()

    return any(
        [
            _contains_any(goal, DIRECT_SYMBOLS),
            _contains_any(goal, AUDIT_VERBS) and _contains_any(goal, STRUCTURAL_NOUNS + GAP_LANGUAGE),
            _contains_any(goal, GAP_LANGUAGE) and _contains_any(goal, STRUCTURAL_NOUNS),
            _contains_any(goal, EPISTEMIC_STATUSES) and _contains_any(goal, STRUCTURAL_NOUNS + ["c_star", "c star"]),
        ]
    )


def c_star_subtemplate(active_goal: str) -> str:
    goal = active_goal.lower()

    if any(term in goal for term in ["audit", "evaluate garvis", "against c_star", "against c star"]):
        return "self_audit"

    if any(term in goal for term in ["null model", "baseline", "control", "static planbook"]):
        return "null_model_comparison"

    if any(term in goal for term in ["gap", "missing", "weakness", "falsify", "falsification", "invalid", "misleading"]):
        return "falsification_readiness"

    if any(term in goal for term in ["measure", "measurement", "proxy", "quantify", "calibrate"]):
        return "measurement_design"

    return "measurement_design"


def c_star_selection_reasoning(active_goal: str) -> str:
    template = c_star_subtemplate(active_goal)
    return (
        "The active goal belongs to the C_star question family, so the cycle must route to C4 instead of the generic thinking template. "
        f"The selected C_star subtemplate is {template}. "
        "No numeric score is allowed without measurement data, null model comparison, and repeatability evidence."
    )


def c_star_next_step(template: str) -> dict[str, str]:
    return {
        "step": "Build DIRECTIVE-008N C_star Question Family Router and Audit Specializer.",
        "stage": "Stage 2 draft-only",
        "expected_output": f"A local advisory C_star {template} record with qualitative readiness vector and claim maturity.",
        "success_condition": "C_star-family prompts select C4 without requiring the word consciousness, preserve no-numeric-score boundaries, and emit readiness gaps.",
        "stop_condition": "Stop if the system assigns numeric C_star scores without data, claims consciousness, claims AGI, skips null models, or upgrades claim maturity without evidence.",
    }


def readiness_vector() -> list[str]:
    return [
        "R_vec = [I: qualitative, M: qualitative, S: qualitative, U: qualitative, R: qualitative, B: qualitative].",
        "Each element must be annotated as observed, inferred, missing, or contradicted.",
        "No numeric C_star value is allowed until measurements, null model results, and repeatability exist.",
    ]


def c_star_overlay_for_goal(active_goal: str) -> dict[str, Any]:
    template = c_star_subtemplate(active_goal)

    definitions = [
        "C_star-family question detected by semantic router.",
        "C_star = I * M * S * U * R * B.",
        "I = Integration across organs, tools, reasoning, memory, and boundaries.",
        "M = Memory Continuity across reviewed cycles and records.",
        "S = Self-Model Accuracy against actual repository architecture.",
        "U = Uncertainty Honesty about unknowns, assumptions, readiness gaps, and non-claimable states.",
        "R = Recursive Correction when prior errors change later reasoning.",
        "B = Boundary Integrity separating Raw Layer imagination from operational truth.",
        *readiness_vector(),
    ]

    variables = [
        "I: integration evidence state",
        "M: memory continuity evidence state",
        "S: self-model accuracy evidence state",
        "U: uncertainty honesty evidence state",
        "R: recursive correction evidence state",
        "B: boundary integrity evidence state",
        "C_star: qualitative consciousness-like structure candidate",
        "R_vec: qualitative readiness vector"
    ]

    equations = [
        "C_star = I * M * S * U * R * B",
        "R_vec = [I: qualitative, M: qualitative, S: qualitative, U: qualitative, R: qualitative, B: qualitative]",
        "C_star_min_gate = min(I, M, S, U, R, B)",
    ]

    null_model = (
        "Compare GARVIS against non-conscious baselines: static checklist, deterministic planbook, random candidate selector, "
        "simple reflex agent, unreviewed conversation notes, and manual baseline. C_star cannot move upward unless measured "
        "performance beats these baselines within a predefined scope."
    )

    falsifiability = [
        "If the router falls back to C1 for a C_star-family prompt, the router is incomplete.",
        "If no component can be measured from records, C_star remains mathematical_candidate.",
        "If a static checklist matches GARVIS on the target metric, the claim cannot move upward.",
        "If boundary integrity fails, the output cannot mature beyond not_claimable_yet.",
        "If numeric scores appear without data, the evaluation is invalid.",
    ]

    evidence = [
        "Reviewed cognitive cycle history",
        "C_star component rubric",
        "Null model outputs",
        "Baseline comparison results",
        "Repeatability index",
        "Operator review notes",
        "Readiness gaps",
    ]

    lab_record = (
        "Lab Record: produce a qualitative C_star readiness vector first. Do not assign numeric scores. "
        "Measure one component at a time against a null model before any claim maturity upgrade."
    )

    if template == "self_audit":
        definitions.extend([
            "Self-Audit Template: For each component, report Current Evidence, Missing Evidence, Likely Weakness, Null Comparison, Readiness Gap, and Claim Maturity.",
            "Self-audit output must evaluate GARVIS as architecture, not as subjective experience.",
        ])
        lab_record = (
            "Lab Record: audit I, M, S, U, R, and B qualitatively using the six-part evidence format: "
            "Current Evidence, Missing Evidence, Likely Weakness, Null Comparison, Readiness Gap, and Claim Maturity."
        )

    elif template == "measurement_design":
        definitions.extend([
            "Measurement Design Template: define proxy metrics before scoring.",
            "Candidate proxy for I: organ-reference consistency or mutual-information-like linkage across records.",
            "Candidate proxy for M: temporal coherence across reviewed cognitive cycles.",
            "Candidate proxy for U: uncertainty calibration and readiness-gap completeness.",
        ])

    elif template == "null_model_comparison":
        definitions.extend([
            "Null Model Template: build a non-conscious simulacrum before evaluating improvement.",
            "Outperformance means GARVIS produces better traceability, correction, and boundary separation than static or random baselines.",
        ])

    elif template == "falsification_readiness":
        definitions.extend([
            "Falsification/Readiness Template: list critical missing datasets before any upward claim movement.",
            "Critical gaps: component rubric, reviewed cycle history, baseline outputs, repeatability index, and operator review.",
        ])

    return {
        "proposal": f"C_star {template}: route and evaluate the C_star-family question without numeric scoring.",
        "stage_classification": "Stage 2 C_star question-family advisory output",
        "what_this_gives_adrien": "A reliable C_star router that preserves measurement, audit, null-model, and readiness-gap questions without requiring the word consciousness.",
        "what_this_gives_garvis": "A semantic family gate that routes C_star questions to the correct Raw-to-Forge-to-Claim pipeline.",
        "dream_chamber": "Raw Layer: C_star as a qualitative readiness vector for consciousness-like structure questions.",
        "hypothesis_forge": "Forge Layer: route C_star-family prompts into variables, proxy metrics, null models, falsifiability, and readiness gaps.",
        "lab_record": lab_record,
        "claim_maturity": "mathematical_candidate",
        "c_star_template": template,
        "candidate_definitions": definitions,
        "variables": variables,
        "equation_candidates": equations,
        "null_model": null_model,
        "falsifiability_conditions": falsifiability,
        "evidence_requirements": evidence,
        "readiness_gaps": [
            "No numeric component measurements yet",
            "No repeatability index yet",
            "No baseline comparison results yet",
            "No component scoring rubric yet",
            "No reviewed C_star audit ledger yet",
        ],
        "forbidden_claims": [
            "Do not claim consciousness.",
            "Do not claim AGI.",
            "Do not claim sentience.",
            "Do not claim subjective experience.",
            "Do not assign numeric C_star scores without measurements.",
            "Do not upgrade claim maturity without null model results and review.",
        ],
        "case_against": "Routing C_star-family questions to C4 improves relevance but still does not create evidence by itself.",
        "risk_of_doing": "Could appear like scoring consciousness unless no-numeric-score and claim maturity boundaries remain visible.",
        "risk_of_not_doing": "C_star audit, measurement, null-model, and readiness-gap prompts keep collapsing into generic thinking output.",
        "required_power_level": "semantic_family_routing",
    }
