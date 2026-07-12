# Triadic Deep Question Planbook Decision

## Decision

Human operator approved DIRECTIVE-008I: Triadic Deep Question Planbook Refresh.

## Date

2026-07-12T00:26:35Z

## Human operator

Adrien D Thomas

## Purpose

This decision records an update to the GARVIS Cognitive Cycle Runner.

The runner now supports deep-question mode for Dream Chamber, Hypothesis Forge, and Lab Record reasoning.

## Files authorized

This decision authorizes:

- tools/cognitive_cycle_runner.py
- tests/test_cognitive_cycle_runner.py
- app_infrastructure/interfaces/TRIADIC_DEEP_QUESTION_PLANBOOK_RUNBOOK.md
- ai_infrastructure/decisions/TRIADIC_DEEP_QUESTION_PLANBOOK_DECISION.md

## What this PR may do

This PR may update:

- known organ detection
- deep-question detection
- candidate generation for unresolved questions
- Dream Chamber mapping
- Hypothesis Forge mapping
- Lab Record mapping
- forbidden-claim mapping
- comparison logic
- selection logic
- next-smallest-step language
- tests
- runbook
- decision record

## What this PR may not do

This PR may not create:

- runtime writer
- database migration
- SQLite table
- append command
- cognitive-cycle JSON ingestion
- dream record ingestion
- bridge record ingestion
- experiment execution
- result record
- background service
- autonomous runtime
- app runtime
- graphical app
- LLM call
- network call
- approval engine
- action engine
- local execution engine
- candidate execution
- test execution by a GARVIS tool
- guard execution by a GARVIS tool
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
- consciousness claim
- AGI claim
- sentience claim
- proof claim

## Stage classification

This is Stage 2 cognitive planbook refresh.

It adds deep-question advisory reasoning.

It does not execute the answers.

## Standing sentence

GARVIS may ask deep questions, but deep questions must pass through Dream, Bridge, and Lab before they can become claims.
