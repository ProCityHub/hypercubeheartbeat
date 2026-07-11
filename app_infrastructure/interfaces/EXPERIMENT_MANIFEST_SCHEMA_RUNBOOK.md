# Experiment Manifest Schema Runbook

## Status

Implementation runbook for DIRECTIVE-007A.

This is schema-only scientific method infrastructure.

It is not an experiment runner.

It is not an action engine.

## Tooling status

This directive defines:

`ai_infrastructure/schemas/experiment_manifest_schema_v1.json`

It does not create a CLI harness.

It does not execute experiments.

It does not run method commands.

## Purpose

The Experiment Manifest Schema defines the minimum structure required before any future GARVIS scientific run can be considered valid.

It converts a research thought into a pre-registered experimental object.

The schema forces each proposed experiment to state:

- hypothesis
- prediction
- counter-prediction
- null model
- data needed
- method
- failure conditions
- claim boundary
- dry-run boundary
- approval status
- ledger chain
- safety restrictions

## Amendment 1: pre-registration hash

The manifest must be committed before any run.

Every result must cite the manifest commit SHA.

A result whose manifest SHA postdates the run is invalid on its face.

This blocks post-hoc ratio selection and rationalization.

## Amendment 2: mandatory null model

Every manifest must define what pure noise would produce on the test.

No null model means no valid run.

The null model must include expected noise behavior, false-positive risk, comparison method, and sample size or trials.

## Amendment 3: fixed claim vocabulary

Every result must use only this closed claim vocabulary:

- exploratory
- suggestive
- supported
- retracted

Free-text claim upgrades are not permitted.

Claim upgrades require a new manifest, new approval, and new run.

## Amendment 4: dry-run means cannot-run

A future dry-run harness may parse, validate, and print the exact command it would run.

It must not execute that command.

Execution is a separate Stage 3 approved event.

The ledger chain must connect:

manifest commit SHA → approval ledger ID → run ID → result ID → claim record ID

## Stage classification

This directive is Stage 2 draft-only preparation.

It defines experimental thought structure.

It creates no experiment execution capability.

## No-network law

This schema makes no network calls.

This directive creates no LLM calls.

This directive creates no outside contact.

This directive creates no cloud export.

## Standing boundary

A manifest is not proof.

A prediction is not evidence.

A dry-run is not execution.

A result without a valid manifest chain is not a valid scientific result.
