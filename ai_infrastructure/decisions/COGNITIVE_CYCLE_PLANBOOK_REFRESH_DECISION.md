# Cognitive Cycle Planbook Refresh Decision

## Decision

Human operator approved DIRECTIVE-008E: Cognitive Cycle Runner Planbook Refresh.

## Date

2026-07-11T23:13:29Z

## Human operator

Adrien D Thomas

## Purpose

This decision records an update to the GARVIS Cognitive Cycle Runner planbook.

The runner had a stale body map and continued to recommend the already-merged Cognitive Cycle Viewer.

This directive updates known organs and candidate thoughts so future Heartbeat cycles reason from the current GARVIS body.

## Files authorized

This decision authorizes:

- tools/cognitive_cycle_runner.py
- tests/test_cognitive_cycle_runner.py
- app_infrastructure/interfaces/COGNITIVE_CYCLE_PLANBOOK_REFRESH_RUNBOOK.md
- ai_infrastructure/decisions/COGNITIVE_CYCLE_PLANBOOK_REFRESH_DECISION.md

## What this PR may do

This PR may update:

- known organ detection
- cognitive-cycle runner candidate planbook
- observation language
- comparison language
- selection logic
- uncertainty language
- power-request language
- next-smallest-step recommendation
- tests
- runbook
- decision record

## What this PR may not do

This PR may not create:

- memory database
- SQLite table
- runtime memory writer
- append command
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
- test execution by the runner
- guard execution by the runner
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

This is Stage 2 cognitive planbook refresh.

It updates how Heartbeat thinks about next steps.

It does not execute next steps.

## Standing sentence

A thinking machine must update its own map before it can choose the next road.
