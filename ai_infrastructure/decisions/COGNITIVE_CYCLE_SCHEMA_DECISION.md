# Cognitive Cycle Schema Decision

## Decision

Human operator approved DIRECTIVE-008A: Cognitive Cycle Schema.

## Date

2026-07-11T22:26:11Z

## Human operator

Adrien D Thomas

## Purpose

This decision records creation of the first GARVIS cognitive cycle schema.

The schema defines one bounded thought cycle for the Jarvis-style brain.

## Files authorized

This decision authorizes:

- ai_infrastructure/schemas/cognitive_cycle_schema_v1.json
- app_infrastructure/interfaces/COGNITIVE_CYCLE_SCHEMA_RUNBOOK.md
- ai_infrastructure/decisions/COGNITIVE_CYCLE_SCHEMA_DECISION.md
- tests/test_cognitive_cycle_schema.py

## What this PR may do

This PR may create:

- a machine-readable cognitive cycle schema
- observation fields
- candidate thought fields
- case-against fields
- risk-of-doing fields
- risk-of-not-doing fields
- candidate comparison fields
- selection fields
- uncertainty fields
- power request fields
- next smallest step fields
- evolution contract fields
- output boundary fields
- runbook
- tests

## What this PR may not do

This PR may not create:

- cognitive cycle runner
- autonomous runtime
- app runtime
- graphical app
- background service
- LLM call
- network call
- approval engine
- action engine
- local execution engine
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

This is Stage 2 cognitive draft infrastructure.

It defines the shape of a GARVIS thought cycle.

It does not execute the thought cycle.

## Standing sentence

Build thinking first as a cycle of observation, opposition, comparison, uncertainty, and power request.
