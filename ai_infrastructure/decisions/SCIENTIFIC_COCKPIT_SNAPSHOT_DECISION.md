# Scientific Cockpit Snapshot Decision

## Decision

Human operator approved implementation of the GARVIS Scientific Cockpit Snapshot CLI.

## Date

2026-07-11T21:10:23Z

## Human operator

Adrien D Thomas

## Purpose

This decision records implementation of a Stage 0 view-only cockpit instrument.

The snapshot gives Adrien a fast local readout of repository state, tools, tests, decisions, app interface runbooks, untracked risk paths, and local Stage 1 ledger metadata.

## Files authorized

This decision authorizes:

- tools/scientific_cockpit_snapshot.py
- tests/test_scientific_cockpit_snapshot.py
- app_infrastructure/interfaces/SCIENTIFIC_COCKPIT_SNAPSHOT_RUNBOOK.md
- ai_infrastructure/decisions/SCIENTIFIC_COCKPIT_SNAPSHOT_DECISION.md

## What this PR may do

This PR may create:

- a local read-only cockpit snapshot CLI
- git branch and commit display
- working-tree status summary
- risky untracked path warnings
- tool listing
- test listing
- decision record listing
- app interface runbook listing
- safe ledger metadata display
- runbook
- tests

## What this PR may not do

This PR may not create:

- app runtime
- graphical app
- background service
- LLM call
- network call
- export-to-cloud feature
- approval engine
- action engine
- command execution engine
- GitHub write action
- automatic commit
- automatic pull request
- test runner execution
- guard runner execution
- camera read
- microphone read
- raw runtime data read
- runtime data commit
- secret commit
- autonomous action
- empirical outcome
- claim upgrade

## Stage classification

This is Stage 0 view-only behavior.

It creates visibility.

It does not create authority.

## Standing sentence

The cockpit gives Adrien clearer eyes.

It gives GARVIS no hidden hands.
