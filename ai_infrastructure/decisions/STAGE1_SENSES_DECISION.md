# Stage 1 Senses Decision

## Decision

Human operator approved creation of the Stage 1 senses scaffold.

## Date

2026-07-11T18:41:11Z

## Human operator

Adrien D Thomas

## Purpose

This decision creates a local Android/Termux senses scaffold for the GARVIS body.

Stage 1 records local sensor command availability and optional camera, microphone, and notification command results into an append-only SQLite decision ledger.

## Files created

This decision authorizes:

- tools/stage1_senses_loop.py
- docs/STAGE1_SENSES_RUNBOOK.md
- tests/test_stage1_senses_loop.py
- ai_infrastructure/decisions/STAGE1_SENSES_DECISION.md

## What this PR may do

This PR may:

- create a Stage 1 local senses script
- create an append-only SQLite decision ledger schema
- create a self-test path
- create a runbook
- create tests for ledger creation and append-only behavior

## What this PR may not do

This PR may not:

- activate a background service
- create autonomous action
- call an LLM
- send messages
- contact people
- change claims
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
- commit runtime sensor outputs
- add real participant data
- add new quantum job data
- upgrade any claim

## Future path

A future directive may add a controlled background service.

A future directive may add embedding or chat API connections.

A future directive may connect Stage 1 senses to the three-gate thinker.

## Standing boundary

Stage 1 may sense locally.

Stage 1 may write append-only local ledger rows.

Stage 1 may not act autonomously.
