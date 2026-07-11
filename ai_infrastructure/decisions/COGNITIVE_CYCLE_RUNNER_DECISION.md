# Cognitive Cycle Runner Decision

## Decision

Human operator approved DIRECTIVE-008B: Cognitive Cycle Runner CLI.

## Date

2026-07-11T22:44:07Z

## Human operator

Adrien D Thomas

## Purpose

This decision records implementation of the first GARVIS cognitive pulse runner.

The runner creates one bounded Stage 2 cognitive draft cycle using local read-only repository inspection.

## Files authorized

This decision authorizes:

- tools/cognitive_cycle_runner.py
- tests/test_cognitive_cycle_runner.py
- app_infrastructure/interfaces/COGNITIVE_CYCLE_RUNNER_RUNBOOK.md
- ai_infrastructure/decisions/COGNITIVE_CYCLE_RUNNER_DECISION.md

## What this PR may do

This PR may create:

- a local cognitive cycle runner CLI
- read-only git inspection
- known-organ detection
- candidate thought generation
- case-against generation
- risk-of-doing and risk-of-not-doing fields
- candidate comparison
- selected next move
- uncertainty fields
- power-request fields
- local draft JSON output
- local draft Markdown output
- runbook
- tests

## What this PR may not do

This PR may not create:

- autonomous runtime
- app runtime
- graphical app
- background service
- LLM call
- network call
- approval engine
- action engine
- local execution engine
- candidate execution
- test execution
- guard execution
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

This is Stage 2 cognitive draft behavior.

It emits a local thought-cycle draft.

It does not execute the selected recommendation.

## Standing sentence

The first heartbeat of thinking is observation, opposition, comparison, uncertainty, and a next smallest step.
