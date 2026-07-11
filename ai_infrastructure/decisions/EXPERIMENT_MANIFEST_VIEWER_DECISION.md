# Experiment Manifest Viewer Decision

## Decision

Human operator approved DIRECTIVE-007B: Experiment Manifest Viewer / Validator CLI.

## Date

2026-07-11T22:17:05Z

## Human operator

Adrien D Thomas

## Purpose

This decision records implementation of a Stage 2 dry-run validation instrument for GARVIS experiment manifests.

The viewer reads a manifest JSON, validates required scientific-method boundaries, displays the hypothesis, prediction, counter-prediction, null model, claim boundary, approval status, ledger chain warnings, and would-run command.

## Files authorized

This decision authorizes:

- tools/experiment_manifest_viewer.py
- tests/test_experiment_manifest_viewer.py
- app_infrastructure/interfaces/EXPERIMENT_MANIFEST_VIEWER_RUNBOOK.md
- ai_infrastructure/decisions/EXPERIMENT_MANIFEST_VIEWER_DECISION.md

## What this PR may do

This PR may create:

- a local manifest viewer CLI
- manifest completeness checks
- null model checks
- claim boundary checks
- dry-run cannot-execute checks
- safety restriction checks
- pre-registration SHA warnings
- approval warnings
- ledger chain warnings
- would-run command display
- runbook
- tests

## What this PR may not do

This PR may not create:

- experiment execution
- method command execution
- subprocess execution
- shell execution
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

This is Stage 2 dry-run validation.

It validates experiment thought structure.

It does not execute experiments.

## Standing sentence

The viewer lets Adrien see the experiment before the experiment is allowed to move.
