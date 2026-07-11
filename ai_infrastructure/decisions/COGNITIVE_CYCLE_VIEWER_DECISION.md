# Cognitive Cycle Viewer Decision

## Decision

Human operator approved DIRECTIVE-008C: Cognitive Cycle Viewer CLI.

## Date

2026-07-11T22:54:44Z

## Human operator

Adrien D Thomas

## Purpose

This decision records implementation of a Stage 2 cognitive inspection instrument.

The viewer reads a GARVIS cognitive cycle JSON and displays the thought pulse clearly without executing actions.

## Files authorized

This decision authorizes:

- tools/cognitive_cycle_viewer.py
- tests/test_cognitive_cycle_viewer.py
- app_infrastructure/interfaces/COGNITIVE_CYCLE_VIEWER_RUNBOOK.md
- ai_infrastructure/decisions/COGNITIVE_CYCLE_VIEWER_DECISION.md

## What this PR may do

This PR may create:

- a local cognitive cycle viewer CLI
- cycle JSON loading
- safe validation failure handling
- observation display
- known-organ display
- candidate thought display
- case-against display
- risk-of-doing display
- risk-of-not-doing display
- comparison display
- selection display
- uncertainty display
- power-request display
- next-smallest-step display
- output-boundary display
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

This is Stage 2 cognitive inspection behavior.

It displays a thought cycle.

It does not execute a thought cycle.

## Standing sentence

The viewer is the mirror: GARVIS may think, but Adrien must be able to see the thought before any power grows.
