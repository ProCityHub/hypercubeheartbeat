# App Ledger Viewer CLI Runbook

## Status

Implementation runbook for the first read-only GARVIS ledger viewer.

This is a cockpit instrument.

It is a window, not a hand.

## Tool

`tools/app_ledger_viewer.py`

## Purpose

The App Ledger Viewer CLI gives Adrien a safe read-only view into the local Stage 1 decision ledger.

It displays safe metadata only.

It does not display raw JSON payloads by default.

It does not write to the ledger.

It does not export files.

It does not call a network.

## Example

```bash
python tools/app_ledger_viewer.py \
  --db data/stage1_senses/decision_ledger.sqlite3 \
  --limit 10

---

# COPY BLOCK 5 — create decision record

```bash id="vfg5sn"
RUN_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

cat > ai_infrastructure/decisions/APP_LEDGER_VIEWER_CLI_DECISION.md <<EOF
# App Ledger Viewer CLI Decision

## Decision

Human operator approved implementation of the first App Ledger Viewer CLI.

## Date

$RUN_DATE

## Human operator

Adrien D Thomas

## Purpose

This decision records the implementation of a read-only command-line viewer for the local Stage 1 decision ledger.

The viewer gives Adrien a safe cockpit instrument for inspecting GARVIS local ledger metadata.

## Files authorized

This decision authorizes:

- tools/app_ledger_viewer.py
- tests/test_app_ledger_viewer.py
- app_infrastructure/interfaces/APP_LEDGER_VIEWER_CLI_RUNBOOK.md
- ai_infrastructure/decisions/APP_LEDGER_VIEWER_CLI_DECISION.md

## What this PR may do

This PR may create:

- a read-only SQLite ledger viewer CLI
- safe metadata display
- limit handling
- schema validation
- missing database error handling
- tests proving raw JSON values are hidden by default
- tests proving read-only write attempts fail
- tests proving missing databases fail safely
- a runbook for local use

## What this PR may not do

This PR may not create:

- app runtime
- graphical app
- background service
- LLM call
- network call
- export feature
- approval engine
- action engine
- proposal engine
- self-design runner
- camera read
- microphone read
- runtime data commit
- secret commit
- autonomous action
- empirical outcome
- claim upgrade

## Stage classification

This is Stage 0 view-only behavior.

It is not Stage 3 execution beyond the approved local command used to run it.

It is not Stage 4 repository action except for the human-approved PR carrying this implementation.

## Standing sentence

The viewer gives Adrien eyes.

It gives GARVIS no hands.
