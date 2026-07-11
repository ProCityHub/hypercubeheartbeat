# Experiment Manifest Schema Decision

## Decision

Human operator approved DIRECTIVE-007A: Experiment Manifest Schema.

## Date

2026-07-11T22:03:01Z

## Human operator

Adrien D Thomas

## Purpose

This decision records creation of the first Scientific Method Engine schema.

The schema defines how future experiment thoughts must be structured before any run is allowed.

## Files authorized

This decision authorizes:

- ai_infrastructure/schemas/experiment_manifest_schema_v1.json
- app_infrastructure/interfaces/EXPERIMENT_MANIFEST_SCHEMA_RUNBOOK.md
- ai_infrastructure/decisions/EXPERIMENT_MANIFEST_SCHEMA_DECISION.md
- tests/test_experiment_manifest_schema.py

## Required amendments

This directive includes four required amendments:

1. Pre-registration hash
2. Mandatory null model
3. Fixed claim vocabulary
4. Dry-run means cannot-run

## What this PR may do

This PR may create:

- a machine-readable experiment manifest schema
- required hypothesis and prediction fields
- required counter-prediction fields
- required null model fields
- required claim boundary fields
- required dry-run cannot-execute boundary fields
- required ledger chain fields
- schema tests
- runbook
- decision record

## What this PR may not do

This PR may not create:

- experiment runner
- dry-run harness CLI
- local execution engine
- approval engine
- action queue
- graphical app
- background service
- LLM call
- network call
- GitHub write action
- automatic commit
- automatic pull request
- external contact
- camera read
- microphone read
- raw runtime data read
- runtime data commit
- secret access
- autonomous action
- empirical result
- claim upgrade

## Stage classification

This is Stage 2 draft-only scientific method infrastructure.

It defines the shape of experiment thought.

It does not execute experiments.

## Standing sentence

Thinking becomes scientific only when prediction, null model, failure, and claim boundary are locked before the run.
