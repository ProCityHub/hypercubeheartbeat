# Scientific Cockpit Snapshot Runbook

## Status

Implementation runbook for the GARVIS Scientific Cockpit Snapshot CLI.

This is a Stage 0 view-only cockpit instrument.

It is not an action engine.

## Tool

`tools/scientific_cockpit_snapshot.py`

## Purpose

The Scientific Cockpit Snapshot CLI gives Adrien a fast local readout of the GARVIS repository, tools, tests, decisions, app interface runbooks, untracked risk paths, and local Stage 1 ledger metadata.

It is designed to answer:

- What branch is active?
- What commit is active?
- Is the working tree clean?
- What tools exist?
- What tests exist?
- What decision records exist?
- What app interface runbooks exist?
- What local ledger memory exists?
- What should not be committed?

## Example

    python tools/scientific_cockpit_snapshot.py \
      --repo . \
      --ledger-db data/stage1_senses/decision_ledger.sqlite3 \
      --limit 8

## Stage classification

This tool is Stage 0 view-only behavior.

It reads and displays local metadata.

It does not run tests.

It does not run guard checks.

It does not execute proposed actions.

It does not approve actions.

It does not write files.

## Ledger boundary

The tool may read safe ledger metadata if a local ledger path is provided or available at the default path.

It displays:

- row counts
- approved row counts
- latest row id
- latest timestamp
- latest decision
- latest approval flag
- shortened latest input hash

It does not display:

- raw candidates_json
- raw gate_scores_json
- raw camera paths
- raw microphone paths
- full input hashes
- raw sensor payloads

## Do Not Commit Watch

The snapshot warns about risky untracked paths such as:

- `AGI/`
- `brain.py`
- `data/stage1_senses/`
- `tmp/self_design_proposals/`

These paths are local runtime or private development material unless explicitly authorized by a future directive.

## No-network law

The snapshot makes no network calls.

It performs no HTTP requests.

It performs no LLM API calls.

It performs no cloud sync.

It performs no telemetry.

It performs no outside contact.

## Standing boundary

Seeing is not approving.

Reporting is not executing.

A cockpit instrument is not a hand.

Adrien remains final authority.
