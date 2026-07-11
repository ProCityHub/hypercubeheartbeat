# GARVIS Message Schema v1

## Status

This is a machine-readable communication schema.

It does not create consciousness.

It does not claim AGI.

It does not record an empirical outcome.

It does not upgrade any claim.

## Purpose

This document explains `ai_infrastructure/schemas/garvis_message_schema_v1.json`.

The schema turns the GARVIS-to-hypercubeheartbeat communication protocol into a structured message format.

## Message types

The schema allows eight message types:

- `status`
- `intake`
- `manifest`
- `proposal`
- `decision_request`
- `action_directive`
- `report`
- `refusal`

## Required fields

Every message must include:

- `schema_version`
- `message_id`
- `message_type`
- `created_utc`
- `created_by`
- `summary`
- `allowed_actions`
- `forbidden_actions`
- `human_approval_required`
- `claim_boundary`
- `evidence_boundary`
- `stop_conditions`

## Boundaries

Every message must state both a claim boundary and an evidence boundary.

This prevents GARVIS messages from silently becoming claim upgrades or empirical outcomes.

## Claim boundary values

Allowed claim boundary values:

- `no_claim_change`
- `claims_ledger_reference_required`
- `unapproved_claim_language_stop`

## Evidence boundary values

Allowed evidence boundary values:

- `not_evidence`
- `sandbox_only`
- `exploratory_only`
- `registered_analysis_required`
- `manifest_only`

## Why this matters

A Jarvis-like body needs structured messages.

The body may route and propose.

The guarded lab brain may record and test.

The human may decide.

The schema keeps those roles separate.

## Example message

```json
{
  "schema_version": "garvis_message_schema_v1",
  "message_id": "example-status-001",
  "message_type": "status",
  "created_utc": "2026-07-11T00:00:00Z",
  "created_by": "GARVIS",
  "summary": "Report current repository state without changing claims.",
  "source_files": [],
  "destination_files": [],
  "related_prs": ["#43"],
  "allowed_actions": [
    "summarize merged PRs",
    "identify next allowed infrastructure step"
  ],
  "forbidden_actions": [
    "upgrade claims",
    "edit outcome logs",
    "run registered analyses"
  ],
  "human_approval_required": false,
  "claim_boundary": "no_claim_change",
  "evidence_boundary": "not_evidence",
  "stop_conditions": [
    "stop if asked to change claims",
    "stop if asked to run empirical scoring"
  ],
  "notes": "Status message only."
}

---

## COPY BLOCK 4 — create decision file

```bash id="6rhbki"
cat > ai_infrastructure/decisions/GARVIS_MESSAGE_SCHEMA_DECISION.md <<'EOF'
# GARVIS Message Schema Decision

## Decision

Human-approved creation of machine-readable GARVIS message schema v1.

## Purpose

This decision allows the repository to define structured JSON messages for communication between the GARVIS body layer and the hypercubeheartbeat guarded lab brain.

## What this PR may add

This PR may add:

- `ai_infrastructure/schemas/garvis_message_schema_v1.json`
- `docs/GARVIS_MESSAGE_SCHEMA_V1.md`
- `ai_infrastructure/decisions/GARVIS_MESSAGE_SCHEMA_DECISION.md`
- `tests/test_garvis_message_schema.py`

## What this PR may not do

This PR may not:

- create consciousness claims
- create AGI claims
- record an empirical outcome
- run LL-07
- score LL-06
- touch Kubota holdout data
- edit outcome logs
- edit frozen preregistration files
- change formulas
- change datasets
- commit synthetic outputs
- add real participant data
- add new quantum job data
- upgrade any claim

## Future path

A later PR may add example message files.

A later PR may add a message validator script.

A later PR may add tests for valid and invalid message examples.

## Standing boundary

The schema structures communication.

The schema does not authorize sensitive action by itself.
