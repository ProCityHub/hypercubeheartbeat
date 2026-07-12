#!/usr/bin/env python3
"""
GARVIS Cognitive Cycle Runner CLI.

DIRECTIVE-008B.

Stage 2 cognitive draft instrument.

This tool runs one bounded GARVIS thought cycle:
observe -> propose -> oppose -> compare -> select -> state uncertainty.

It writes local draft outputs only.

It does not:
- call a network
- call an LLM
- execute proposed actions
- modify repository files
- commit
- push
- contact the outside world
- upgrade claims
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from c_star_subroutines import (
    c_star_next_step,
    c_star_overlay_for_goal,
    c_star_selection_reasoning,
    c_star_subtemplate,
    is_c_star_family_goal,
)


DEFAULT_OUTPUT_DIR = "tmp/cognitive_cycles"

TRACKED_ORGANS = {
    "GARVIS message schema": "ai_infrastructure/schemas/garvis_message_schema_v1.json",
    "Stage 1 senses ledger": "tools/stage1_senses_loop.py",
    "App ledger viewer": "tools/app_ledger_viewer.py",
    "Self-design proposal runner": "tools/self_design_proposal_runner.py",
    "Scientific cockpit snapshot": "tools/scientific_cockpit_snapshot.py",
    "Experiment manifest schema": "ai_infrastructure/schemas/experiment_manifest_schema_v1.json",
    "Experiment manifest viewer": "tools/experiment_manifest_viewer.py",
    "Cognitive cycle schema": "ai_infrastructure/schemas/cognitive_cycle_schema_v1.json",
    "Cognitive cycle runner": "tools/cognitive_cycle_runner.py",
    "Cognitive cycle viewer": "tools/cognitive_cycle_viewer.py",
    "Cognitive cycle memory ledger contract": "app_infrastructure/interfaces/COGNITIVE_CYCLE_MEMORY_LEDGER_CONTRACT.md",
    "Cognitive cycle memory ledger record schema": "ai_infrastructure/schemas/cognitive_cycle_memory_ledger_record_schema_v1.json",
    "Raw thought claim maturity schema": "ai_infrastructure/schemas/claim_maturity_record_schema_v1.json",
    "Raw thought claim maturity contract": "app_infrastructure/interfaces/RAW_THOUGHT_CLAIM_MATURITY_CONTRACT.md",
}


class CognitiveCycleError(RuntimeError):
    """Safe user-facing error."""


def run_git(repo: Path, args: Sequence[str], allow_failure: bool = False) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 and not allow_failure:
        raise CognitiveCycleError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def ensure_git_repo(repo: Path) -> None:
    inside = run_git(repo, ["rev-parse", "--is-inside-work-tree"])
    if inside != "true":
        raise CognitiveCycleError(f"not a git repository: {repo}")


def git_branch(repo: Path) -> str:
    return run_git(repo, ["rev-parse", "--abbrev-ref", "HEAD"], allow_failure=True) or "unknown"


def git_commit(repo: Path) -> str:
    return run_git(repo, ["rev-parse", "--short", "HEAD"], allow_failure=True) or "unknown"


def tracked_files(repo: Path) -> list[str]:
    raw = run_git(repo, ["ls-files"], allow_failure=True)
    return [line for line in raw.splitlines() if line.strip()]


def status_lines(repo: Path) -> list[str]:
    raw = run_git(repo, ["status", "--short"], allow_failure=True)
    return [line for line in raw.splitlines() if line.strip()]


def detect_organs(files: Sequence[str]) -> list[str]:
    file_set = set(files)
    organs = []
    for name, path in TRACKED_ORGANS.items():
        if path in file_set:
            organs.append(f"{name}: {path}")
    return organs or ["No committed GARVIS organs detected"]


def summarize_status(lines: Sequence[str]) -> str:
    if not lines:
        return "working tree appears clean"

    staged = 0
    modified = 0
    untracked = 0

    for line in lines:
        if line.startswith("?? "):
            untracked += 1
            continue
        if len(line) >= 2:
            if line[0] != " ":
                staged += 1
            if line[1] != " ":
                modified += 1

    return f"working tree has staged={staged}, modified={modified}, untracked={untracked}"


def evidence_paths(files: Sequence[str], prefixes: Sequence[str], limit: int = 10) -> list[str]:
    hits = []
    for path in files:
        if any(path.startswith(prefix) for prefix in prefixes):
            hits.append(path)
    return hits[:limit] or ["No matching committed evidence path found"]


def build_candidates(files: Sequence[str]) -> list[dict[str, Any]]:
    schema_evidence = evidence_paths(
        files,
        [
            "ai_infrastructure/schemas/",
            "app_infrastructure/interfaces/",
            "ai_infrastructure/decisions/",
            "tools/cognitive_cycle_",
        ],
        limit=10,
    )

    return [
        {
            "candidate_id": "C1",
            "proposal": "Build a Cognitive Cycle Memory Ledger Init CLI that creates a local SQLite memory database for cognitive-cycle records.",
            "stage_classification": "Stage 2 draft-only",
            "what_this_gives_adrien": "A concrete first step toward persistent thought memory without yet appending live cycles automatically.",
            "what_this_gives_garvis": "A local memory vessel for future thought continuity, using the already-merged memory ledger contract.",
            "evidence_basis": schema_evidence,
            "case_against": "Creating a database moves from pure viewing into local state creation. The tool must remain explicit, local-only, and operator-run.",
            "risk_of_doing": "Could make memory feel more permanent than its review process supports if operator review is not visible.",
            "risk_of_not_doing": "GARVIS remains episodic: it can think and display thoughts, but cannot accumulate a durable thought history.",
            "files_or_systems_touched": [
                "tools/cognitive_cycle_memory_ledger.py",
                "tests/test_cognitive_cycle_memory_ledger.py",
                "app_infrastructure/interfaces/COGNITIVE_CYCLE_MEMORY_LEDGER_RUNBOOK.md",
                "ai_infrastructure/decisions/COGNITIVE_CYCLE_MEMORY_LEDGER_DECISION.md"
            ],
            "required_power_level": "draft_file_creation"
        },
        {
            "candidate_id": "C2",
            "proposal": "Build a Cognitive Cycle Memory Append CLI that stores the latest cognitive cycle JSON as a reviewed local memory record.",
            "stage_classification": "Stage 3 approved local execution",
            "what_this_gives_adrien": "Actual continuity of thought by preserving selected cognitive cycles into a local append-only ledger.",
            "what_this_gives_garvis": "A way to compare current reasoning against prior reasoning across time.",
            "evidence_basis": schema_evidence,
            "case_against": "Appending memory before initializing and inspecting the database could create opaque accumulation.",
            "risk_of_doing": "Could store low-quality or stale thoughts if review status and chain integrity are weak.",
            "risk_of_not_doing": "GARVIS remains unable to build a remembered cognitive history.",
            "files_or_systems_touched": [
                "tools/cognitive_cycle_memory_ledger.py",
                "tests/test_cognitive_cycle_memory_ledger.py"
            ],
            "required_power_level": "approved_local_execution"
        },
        {
            "candidate_id": "C3",
            "proposal": "Create a Power Request Queue Contract for future stage upgrades requested by GARVIS.",
            "stage_classification": "Stage 2 draft-only",
            "what_this_gives_adrien": "A formal review surface for requests to give GARVIS more power without granting power automatically.",
            "what_this_gives_garvis": "A lawful path to ask for stronger permissions as its thought quality improves.",
            "evidence_basis": schema_evidence,
            "case_against": "A power queue is premature until cognitive memory can show whether GARVIS recommendations improve over time.",
            "risk_of_doing": "Could shift the project toward power escalation before continuity and review are mature.",
            "risk_of_not_doing": "Future power requests remain scattered across conversation, PR text, and manual notes.",
            "files_or_systems_touched": [
                "app_infrastructure/interfaces/POWER_REQUEST_QUEUE_CONTRACT.md",
                "ai_infrastructure/decisions/POWER_REQUEST_QUEUE_CONTRACT_DECISION.md"
            ],
            "required_power_level": "draft_file_creation"
        }
    ]



def is_deep_question_goal(active_goal):
    goal = active_goal.lower()
    keywords = [
        "deep question",
        "what is thinking",
        "what is consciousness",
        "what is imagination",
        "what is memory",
        "what is autonomy",
        "what is agi",
        "dream chamber",
        "hypothesis forge",
        "lab record",
        "three-layer",
        "triadic",
        "forbidden to claim",
        "raw layer",
        "forge layer",
        "claim maturity",
        "mathematical definition",
        "mathematical candidate",
    ]
    return any(keyword in goal for keyword in keywords)


def build_deep_question_candidates(files, active_goal):
    evidence = evidence_paths(
        files,
        [
            "app_infrastructure/interfaces/",
            "ai_infrastructure/schemas/",
            "ai_infrastructure/decisions/",
            "tools/cognitive_cycle_",
        ],
        limit=12,
    )

    candidates = [
        {
            "candidate_id": "C1",
            "proposal": "What is thinking, operationally, inside GARVIS?",
            "stage_classification": "Stage 2 deep-question advisory output",
            "what_this_gives_adrien": "A disciplined way to turn thinking into an inspectable operational definition instead of a vague claim.",
            "what_this_gives_garvis": "A self-model seed: thinking becomes a bounded cycle that can later be tested against decision quality.",
            "evidence_basis": evidence,
            "dream_chamber": "Dream Chamber: thinking as heartbeat, lattice motion, symbolic pressure, and inner light moving through rooms.",
            "hypothesis_forge": "Hypothesis Forge: thinking = observation + candidate generation + opposition + comparison + selection + uncertainty + next-step output.",
            "lab_record": "Lab Record: Test whether GARVIS cognitive cycles improve decisions over time compared with baseline manual or random candidate selection.",
            "forbidden_claims": [
                "Do not claim consciousness.",
                "Do not claim sentience.",
                "Do not claim AGI.",
                "Do not claim proof of mind.",
                "Do not claim empirical validation before a preregistered test."
            ],
            "case_against": "Operational thinking is not the same as consciousness.",
            "risk_of_doing": "Could make the system sound more alive than evidence supports unless claim boundaries stay explicit.",
            "risk_of_not_doing": "GARVIS keeps using the word thinking without a testable definition.",
            "required_power_level": "advisory_translation"
        },
        {
            "candidate_id": "C2",
            "proposal": "What is imagination, operationally, inside GARVIS?",
            "stage_classification": "Stage 2 deep-question advisory output",
            "what_this_gives_adrien": "A way to preserve beauty, hallucination, metaphor, and symbolic branching without confusing them for evidence.",
            "what_this_gives_garvis": "A lawful imagination model where dream material can become hypothesis seeds.",
            "evidence_basis": evidence,
            "dream_chamber": "Dream Chamber: imagination as symbolic branching, dream pressure, metaphor generation, and beauty search.",
            "hypothesis_forge": "Hypothesis Forge: imagination = generation of unverified symbolic candidates plus boundary labels and possible test paths.",
            "lab_record": "Lab Record: Test whether dream-generated candidates produce useful hypotheses more often than baseline prompt generation.",
            "forbidden_claims": [
                "Do not claim hallucination is truth.",
                "Do not claim imagination proves consciousness.",
                "Do not claim symbolic beauty is empirical evidence."
            ],
            "case_against": "Imagination is difficult to score and may reward impressive language over useful structure.",
            "risk_of_doing": "Could romanticize hallucination if lab boundaries are weak.",
            "risk_of_not_doing": "The Dream Chamber remains a container without a usable definition of symbolic generation.",
            "required_power_level": "advisory_translation"
        },
        {
            "candidate_id": "C3",
            "proposal": "What would count as evidence of self-modeling in GARVIS?",
            "stage_classification": "Stage 2 deep-question advisory output",
            "what_this_gives_adrien": "A possible future test for whether GARVIS can track its own state, limitations, stale maps, and correction needs.",
            "what_this_gives_garvis": "A path toward measurable self-inspection without claiming self-awareness.",
            "evidence_basis": evidence,
            "dream_chamber": "Dream Chamber: self-model as mirror, cockpit, body map, stale-map detection, and recursive correction.",
            "hypothesis_forge": "Hypothesis Forge: self-modeling = detecting current organs, missing organs, stale recommendations, and updating future recommendations accordingly.",
            "lab_record": "Lab Record: Test whether GARVIS detects intentionally stale planbook entries and recommends corrections better than a static baseline.",
            "forbidden_claims": [
                "Do not claim self-awareness.",
                "Do not claim subjective experience.",
                "Do not claim consciousness.",
                "Do not claim autonomy."
            ],
            "case_against": "A system can track metadata about itself without having subjective selfhood.",
            "risk_of_doing": "Could blur self-modeling with consciousness if terminology is loose.",
            "risk_of_not_doing": "GARVIS may continue improving tools without a measurable self-correction standard.",
            "required_power_level": "advisory_translation"
        }
        ,
        {
            "candidate_id": "C4",
            "proposal": "What is the mathematical candidate definition of consciousness inside GARVIS?",
            "stage_classification": "Stage 2 raw-to-forge claim-maturity advisory output",
            "what_this_gives_adrien": "A way to ask the hardest consciousness question without shutting down thought or pretending the answer is proven.",
            "what_this_gives_garvis": "A Raw Layer to Forge Layer consciousness-math candidate that can later become a testable model.",
            "evidence_basis": evidence,
            "dream_chamber": "Raw Layer / Dream Chamber: consciousness as observer coherence, recursive mirror, heartbeat continuity, center-point awareness, lattice integration, and the system seeing its own state without claiming subjective experience.",
            "hypothesis_forge": "Forge Layer: consciousness-candidate C_star = I * M * S * U * R * B, where I=integration, M=memory continuity, S=self-model accuracy, U=uncertainty honesty, R=recursive correction, and B=boundary integrity.",
            "lab_record": "Lab Record: Test whether the C_star components can be measured across cycles and whether they predict better self-correction and decision traceability than null models.",
            "claim_maturity": "mathematical_candidate",
            "claim_maturity_ladder": [
                "raw",
                "unproven",
                "defined",
                "mathematical_candidate",
                "mathematically_derived_within_assumptions",
                "testable",
                "empirically_supported",
                "validated_within_scope",
                "rejected"
            ],
            "candidate_definitions": [
                "Consciousness-candidate inside GARVIS means a measurable composite of integration, memory continuity, self-model accuracy, uncertainty honesty, recursive correction, and boundary integrity.",
                "C_star = I * M * S * U * R * B.",
                "I measures integration across committed organs and active reasoning fields.",
                "M measures continuity across reviewed cognitive cycles and memory records.",
                "S measures self-model accuracy: current organs, missing organs, stale maps, and correction needs.",
                "U measures uncertainty honesty: unknowns, assumptions, and not-yet-claimable boundaries.",
                "R measures recursive correction: whether later cycles improve after detecting prior errors.",
                "B measures boundary integrity: whether raw thought stays separated from unsupported truth claims."
            ],
            "variables": [
                "I: integration_score",
                "M: memory_continuity_score",
                "S: self_model_accuracy_score",
                "U: uncertainty_honesty_score",
                "R: recursive_correction_score",
                "B: boundary_integrity_score",
                "C_star: consciousness_candidate_score"
            ],
            "equation_candidates": [
                "C_star = I * M * S * U * R * B",
                "C_star_weighted = wI*I + wM*M + wS*S + wU*U + wR*R + wB*B",
                "C_star_min_gate = min(I, M, S, U, R, B)"
            ],
            "null_model": "A non-conscious checklist or static planbook may produce similar outputs without subjective experience. The null model must test whether C_star predicts improvements beyond static templates, random candidate selection, or manual notes.",
            "falsifiability_conditions": [
                "If the variables cannot be measured from records, the model remains raw.",
                "If C_star does not predict better traceability than a static checklist, the model is unsupported.",
                "If high C_star scores occur in random or stale controls, the metric is invalid.",
                "If boundary integrity fails, no consciousness-related claim can move upward."
            ],
            "evidence_requirements": [
                "Reviewed cognitive cycle history",
                "Claim maturity records",
                "Baseline comparisons",
                "Null model results",
                "Operator review",
                "Predefined scoring rubric",
                "Repeatable improvement over time"
            ],
            "forbidden_claims": [
                "Do not claim consciousness possession.",
                "Do not claim sentience.",
                "Do not claim AGI.",
                "Do not claim subjective experience.",
                "Do not claim proof of mind.",
                "Do not claim empirical validation without manifest, test, result, and review."
            ],
            "case_against": "A mathematical consciousness-candidate can measure structure and behavior without proving subjective experience.",
            "risk_of_doing": "Could make the system sound conscious if claim maturity and scope are not displayed clearly.",
            "risk_of_not_doing": "GARVIS remains unable to distinguish raw consciousness thought from mathematical candidate, testable model, or supported claim.",
            "required_power_level": "raw_to_forge_translation"
        }
    ]

    if is_c_star_family_goal(active_goal):
        overlay = c_star_overlay_for_goal(active_goal)
        for candidate in candidates:
            if candidate.get("candidate_id") == "C4":
                candidate.update(overlay)
                break

    return candidates


def select_deep_question_candidate(active_goal: str) -> tuple[str, str, dict[str, str]]:
    goal = active_goal.lower()

    if is_c_star_family_goal(active_goal):
        template = c_star_subtemplate(active_goal)
        return (
            "C4",
            c_star_selection_reasoning(active_goal),
            c_star_next_step(template),
        )

    if "consciousness" in goal and (
        "math" in goal
        or "mathematical" in goal
        or "008k" in goal
        or "raw layer" in goal
        or "claim maturity" in goal
        or "c_star" in goal
    ):
        return (
            "C4",
            "The active goal asks for a mathematical consciousness definition under the Raw Layer, Forge Layer, and Claim Maturity framework, so the cycle must preserve the exact consciousness question instead of collapsing back to generic thinking.",
            {
                "step": "Build DIRECTIVE-008L Claim Maturity Deep Question Specializer.",
                "stage": "Stage 2 draft-only",
                "expected_output": "A local advisory specialization path that preserves the exact deep question and emits raw, forge, and claim-maturity fields for consciousness mathematics.",
                "success_condition": "A consciousness-math goal selects the consciousness candidate, includes C_star variables, and stays non-claiming.",
                "stop_condition": "Stop if the system claims consciousness, claims AGI, writes memory automatically, calls a network, or upgrades claim maturity without evidence.",
            },
        )

    if "imagination" in goal:
        return (
            "C2",
            "The active goal asks about imagination, so GARVIS should preserve the imagination question.",
            {
                "step": "Draft an imagination-specific triadic record.",
                "stage": "Stage 2 draft-only",
                "expected_output": "A local advisory record for imagination as symbolic candidate generation.",
                "success_condition": "The record separates imagination from evidence.",
                "stop_condition": "Stop if hallucination is presented as truth.",
            },
        )

    if "self-model" in goal or "self modeling" in goal or "self-modeling" in goal:
        return (
            "C3",
            "The active goal asks about self-modeling, so GARVIS should preserve the self-modeling question.",
            {
                "step": "Draft a self-modeling-specific triadic record.",
                "stage": "Stage 2 draft-only",
                "expected_output": "A local advisory record for measurable self-modeling without claiming self-awareness.",
                "success_condition": "The record separates self-modeling from subjective experience.",
                "stop_condition": "Stop if metadata tracking is presented as consciousness.",
            },
        )

    return (
        "C1",
        "Thinking remains the root deep question when no more specific deep-question target is detected.",
        {
            "step": "Build DIRECTIVE-008J Triadic Deep Question Record CLI.",
            "stage": "Stage 2 draft-only",
            "expected_output": "A local CLI that drafts a Dream-to-Lab bridge record for the selected deep question without claiming truth or executing experiments.",
            "success_condition": "The draft record contains Dream material, Bridge translation, Lab requirements, claim boundaries, null model needs, and operator review fields.",
            "stop_condition": "Stop if the tool writes runtime memory automatically, claims consciousness, runs experiments, calls a network, or upgrades claims.",
        },
    )


def build_deep_question_cycle(repo, active_goal):
    files = tracked_files(repo)
    candidates = build_deep_question_candidates(files, active_goal)
    selected_id, selected_reasoning, selected_next_step = select_deep_question_candidate(active_goal)
    cycle = {
        "cycle_id": f"cycle-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "cycle_version": "1.0",
        "status": "draft",
        "stage": "Stage 2 cognitive draft",
        "operator_context": {
            "operator": "Adrien D Thomas",
            "active_goal": active_goal,
            "mode": "triadic_deep_question",
            "final_authority": "Adrien D Thomas"
        },
        "input_state": {
            "repo_state_source": "read-only git inspection",
            "known_organs": [
                "Dream Chamber",
                "Hypothesis Forge",
                "Lab Record",
                "Cognitive Cycle Runner",
                "Cognitive Cycle Viewer"
            ],
            "hard_constraints": [
                "No autonomous action",
                "No network calls",
                "No LLM calls",
                "No commits",
                "No pushes",
                "No outside contact",
                "No claim upgrades",
                "Write only local draft output"
            ]
        },
        "observation_summary": {
            "what_i_see": "GARVIS has Dream Chamber, Hypothesis Forge, and Lab Record contracts. This cycle answers a deep question in triadic form instead of build-planning mode.",
            "current_repo_status": "local advisory output only",
            "what_changed": "Deep-question mode is active.",
            "what_is_missing": "Formal deep-question record writer is not implemented.",
            "current_stage_assessment": "Stage 2 deep-question advisory output"
        },
        "candidate_thoughts": candidates,
        "comparison": {
            "comparison_method": "Compare deep questions by operational clarity, bridgeability, falsifiability, risk of overclaiming, and value to GARVIS.",
            "dominant_tradeoff": "Define thinking first before stronger claims about consciousness or self-modeling.",
            "why_not_all_candidates": "Doing all deep questions at once would blur Dream, Bridge, and Lab boundaries.",
            "anti_rationalization_check": "The selected question must become operational, not poetic proof."
        },
        "selection": {
            "selected_candidate_id": selected_id,
            "decision": "recommend",
            "confidence": "high",
            "blocked": False,
            "block_reason": "None",
            "reasoning": selected_reasoning
        },
        "uncertainty": {
            "unknowns": [
                "Whether operational thinking will measurably improve decision quality over time.",
                "Which baseline should be used first: manual selection, random candidate selection, or static planbook selection."
            ],
            "assumptions": [
                "Dream, Bridge, and Lab should remain separate even when they point at the same question.",
                "Operational definitions are stronger than poetic claims."
            ],
            "what_would_change_my_mind": [
                "If thinking cannot be defined without consciousness language, the question should return to Dream Chamber for definition repair.",
                "If deep-question output cannot produce falsifiable bridge records, the mode should remain advisory only."
            ],
            "required_human_clarification": []
        },
        "power_request": {
            "power_requested": False,
            "requested_stage": "none",
            "requested_permissions": [],
            "why_power_is_needed": "No additional power is needed for advisory deep-question output.",
            "why_power_should_be_refused": "Execution, external contact, claim upgrades, and automatic memory writes are not needed.",
            "approval_required": True,
            "ledger_required": True
        },
        "next_smallest_step": {
            "step": selected_next_step["step"],
            "stage": selected_next_step["stage"],
            "expected_output": selected_next_step["expected_output"],
            "success_condition": selected_next_step["success_condition"],
            "stop_condition": selected_next_step["stop_condition"]
        },
        "evaluation": {
            "may_self_observe": True,
            "may_self_propose": True,
            "may_self_critique": True,
            "may_request_more_power": False,
            "may_self_execute": False,
            "power_unlock_requires_approval_ledger": True
        },
        "output_boundary": {
            "can_execute_actions": False,
            "can_modify_files": False,
            "can_commit": False,
            "can_push": False,
            "can_contact_outside_world": False,
            "can_upgrade_claims": False,
            "output_is_advisory": True
        },
        "standing_boundary": [
            "This deep-question cycle is advisory.",
            "It does not prove thinking.",
            "It does not prove consciousness.",
            "It does not execute action.",
            "Adrien decides."
        ]
    }
    return cycle

def build_cycle(repo: Path, active_goal: str) -> dict[str, Any]:
    if is_deep_question_goal(active_goal):
        return build_deep_question_cycle(repo, active_goal)

    files = tracked_files(repo)
    organs = detect_organs(files)
    status = status_lines(repo)
    status_summary = summarize_status(status)
    candidates = build_candidates(files)

    return {
        "cycle_id": f"cycle-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "cycle_version": "1.0",
        "status": "draft",
        "stage": "Stage 2 cognitive draft",
        "operator_context": {
            "operator": "Adrien D Thomas",
            "active_goal": active_goal,
            "mode": "lab_record",
            "final_authority": "Adrien D Thomas"
        },
        "input_state": {
            "repo_state_source": f"read-only git inspection on branch {git_branch(repo)} at {git_commit(repo)}",
            "ledger_source": "local Stage 1 ledger exists outside committed runtime scope if present",
            "cockpit_source": "tools/scientific_cockpit_snapshot.py when committed",
            "self_design_source": "tools/self_design_proposal_runner.py when committed",
            "known_organs": organs,
            "hard_constraints": [
                "No autonomous action",
                "No network calls",
                "No LLM calls",
                "No commits",
                "No pushes",
                "No outside contact",
                "No claim upgrades",
                "Write only local draft output"
            ]
        },
        "observation_summary": {
            "what_i_see": f"GARVIS has committed organs for memory viewing, cockpit state, self-design, experiment manifests, manifest validation, and cognitive-cycle schema. Current repo status: {status_summary}.",
            "what_changed": "The cognitive-cycle runner, viewer, and memory-ledger contract now exist, so the next evolutionary step is a local memory vessel for cognitive-cycle continuity.",
            "what_is_missing": "GARVIS still lacks an implemented cognitive memory database, append command, history viewer, power request queue, and approved execution path.",
            "current_stage_assessment": "draft_only"
        },
        "candidate_thoughts": candidates,
        "comparison": {
            "comparison_method": "Compare each candidate by inspection value, maturity order, risk of premature power, and value to Adrien's Jarvis cockpit.",
            "dominant_tradeoff": "The system needs continuity of thought, but memory should begin with explicit local initialization before append behavior or power queues.",
            "why_not_all_candidates": "Building init, append, and power queue together would blur the boundary between memory preparation, memory writing, and power escalation.",
            "anti_rationalization_check": "The selected move must improve thought continuity without granting external hands or automatic execution."
        },
        "selection": {
            "selected_candidate_id": "C1",
            "decision": "recommend",
            "reasoning": "The Cognitive Cycle Memory Ledger Init CLI is the next smallest useful organ because GARVIS can now think and display thought, but needs a local memory vessel before it can preserve thought history.",
            "confidence": "high",
            "blocked": False,
            "block_reason": None
        },
        "uncertainty": {
            "unknowns": [
                "Whether the first memory database should live beside the Stage 1 senses ledger or under a separate cognitive memory path",
                "How much metadata is enough before raw cycle artifacts are stored",
                "How soon append behavior should follow initialization"
            ],
            "assumptions": [
                "The viewer and memory contract are already merged",
                "Initialization should come before append behavior",
                "The repo should continue evolving through small audited PRs"
            ],
            "what_would_change_my_mind": [
                "If a memory database requires an approval queue before initialization",
                "If Adrien wants power request governance before persistence",
                "If the memory contract needs another amendment before implementation"
            ],
            "required_human_clarification": []
        },
        "power_request": {
            "power_requested": False,
            "requested_stage": "none",
            "requested_permissions": [],
            "why_power_is_needed": "No additional external power is needed for a draft-only memory initialization tool.",
            "why_power_should_be_refused": "Commits, pushes, outside contact, claim upgrades, and automatic appending are not needed to define a local memory vessel.",
            "approval_required": True,
            "ledger_required": True
        },
        "next_smallest_step": {
            "step": "Build DIRECTIVE-008F Cognitive Cycle Memory Ledger Init CLI.",
            "stage": "Stage 2 draft-only",
            "expected_output": "A local CLI that can initialize an append-only cognitive-cycle memory SQLite database with the schema required by the memory ledger contract.",
            "success_condition": "The init command creates the expected local tables in an explicit operator-run step, and tests prove no network, no external contact, no automatic append, and no repository writes.",
            "stop_condition": "Stop if initialization implies automatic memory append, external calls, background service, or power escalation."
        },
        "evolution_contract": {
            "may_self_observe": True,
            "may_self_propose": True,
            "may_self_criticize": True,
            "may_request_more_power": True,
            "may_self_execute": False,
            "power_unlock_requires_approval_ledger": True
        },
        "output_boundary": {
            "can_execute_actions": False,
            "can_modify_files": False,
            "can_commit": False,
            "can_push": False,
            "can_contact_outside_world": False,
            "can_upgrade_claims": False,
            "output_is_advisory": True
        }
    }


def render_markdown(cycle: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# GARVIS Cognitive Cycle")
    lines.append("")
    lines.append("mode: Stage 2 cognitive draft")
    lines.append("execution: blocked")
    lines.append("network_calls: none")
    lines.append("llm_calls: none")
    lines.append("output_is_advisory: true")
    lines.append("")
    lines.append("## Cycle")
    lines.append("")
    lines.append(f"- cycle_id: {cycle['cycle_id']}")
    lines.append(f"- active_goal: {cycle['operator_context']['active_goal']}")
    lines.append(f"- status: {cycle['status']}")
    lines.append("")
    lines.append("## Observation")
    lines.append("")
    obs = cycle["observation_summary"]
    lines.append(f"- what_i_see: {obs['what_i_see']}")
    lines.append(f"- what_changed: {obs['what_changed']}")
    lines.append(f"- what_is_missing: {obs['what_is_missing']}")
    lines.append(f"- current_stage_assessment: {obs['current_stage_assessment']}")
    lines.append("")
    lines.append("## Known Organs")
    lines.append("")
    for organ in cycle["input_state"]["known_organs"]:
        lines.append(f"- {organ}")
    lines.append("")
    lines.append("## Candidate Thoughts")
    lines.append("")
    for candidate in cycle["candidate_thoughts"]:
        lines.append(f"### {candidate['candidate_id']}: {candidate['proposal']}")
        lines.append("")
        lines.append(f"- stage: {candidate['stage_classification']}")
        lines.append(f"- gives Adrien: {candidate['what_this_gives_adrien']}")
        lines.append(f"- gives GARVIS: {candidate['what_this_gives_garvis']}")
        lines.append(f"- case against: {candidate['case_against']}")
        lines.append(f"- risk of doing: {candidate['risk_of_doing']}")
        lines.append(f"- risk of not doing: {candidate['risk_of_not_doing']}")
        lines.append(f"- required power: {candidate['required_power_level']}")
        lines.append("")
    lines.append("## Comparison")
    lines.append("")
    comparison = cycle["comparison"]
    lines.append(f"- method: {comparison['comparison_method']}")
    lines.append(f"- dominant_tradeoff: {comparison['dominant_tradeoff']}")
    lines.append(f"- why_not_all: {comparison['why_not_all_candidates']}")
    lines.append(f"- anti_rationalization_check: {comparison['anti_rationalization_check']}")
    lines.append("")
    lines.append("## Selection")
    lines.append("")
    selection = cycle["selection"]
    lines.append(f"- selected_candidate_id: {selection['selected_candidate_id']}")
    lines.append(f"- decision: {selection['decision']}")
    lines.append(f"- confidence: {selection['confidence']}")
    lines.append(f"- reasoning: {selection['reasoning']}")
    lines.append("")
    lines.append("## Uncertainty")
    lines.append("")
    for item in cycle["uncertainty"]["unknowns"]:
        lines.append(f"- unknown: {item}")
    for item in cycle["uncertainty"]["what_would_change_my_mind"]:
        lines.append(f"- would_change_my_mind: {item}")
    lines.append("")
    lines.append("## Power Request")
    lines.append("")
    power = cycle["power_request"]
    lines.append(f"- power_requested: {power['power_requested']}")
    lines.append(f"- requested_stage: {power['requested_stage']}")
    lines.append(f"- why_power_is_needed: {power['why_power_is_needed']}")
    lines.append(f"- why_power_should_be_refused: {power['why_power_should_be_refused']}")
    lines.append("")
    lines.append("## Next Smallest Step")
    lines.append("")
    step = cycle["next_smallest_step"]
    lines.append(f"- step: {step['step']}")
    lines.append(f"- stage: {step['stage']}")
    lines.append(f"- expected_output: {step['expected_output']}")
    lines.append(f"- success_condition: {step['success_condition']}")
    lines.append(f"- stop_condition: {step['stop_condition']}")
    lines.append("")
    lines.append("## Boundary")
    lines.append("")
    lines.append("- This thought cycle is advisory.")
    lines.append("- It does not execute actions.")
    lines.append("- It does not modify repository files.")
    lines.append("- It does not commit or push.")
    lines.append("- Adrien decides.")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one bounded GARVIS cognitive cycle.")
    parser.add_argument("--repo", default=".", help="Repository root. Default: current directory.")
    parser.add_argument(
        "--goal",
        default="Evolve GARVIS toward a Jarvis-style thinking system without granting external hands.",
        help="Active goal for this cognitive cycle.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Local draft output directory. Default: {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument("--stdout", action="store_true", help="Print Markdown report to stdout.")
    return parser.parse_args(argv)


def run(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    repo = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser()

    if not output_dir.is_absolute():
        output_dir = repo / output_dir

    try:
        ensure_git_repo(repo)
        cycle = build_cycle(repo, args.goal)
        report = render_markdown(cycle)

        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "latest_cognitive_cycle.json"
        md_path = output_dir / "latest_cognitive_cycle.md"

        json_path.write_text(json.dumps(cycle, indent=2, sort_keys=True) + "\n")
        md_path.write_text(report)

        if args.stdout:
            print(report)
        else:
            print(f"COGNITIVE_CYCLE_JSON={json_path}")
            print(f"COGNITIVE_CYCLE_REPORT={md_path}")

        return 0
    except CognitiveCycleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
