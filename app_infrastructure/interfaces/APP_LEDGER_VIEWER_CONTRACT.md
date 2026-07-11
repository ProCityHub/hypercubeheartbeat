# App Ledger Viewer Contract

## Status

Doc-only.

No viewer implementation.

No script.

No app runtime.

No background service.

No LLM call.

No network call.

No export feature.

No action engine.

No runtime data committed.

No secrets committed.

## Purpose

The App Ledger Viewer Contract defines how future GARVIS app infrastructure may read and display the local Stage 1 SQLite decision ledger.

The ledger is local body memory.

The viewer is a window into that memory.

The viewer is not a hand.

The viewer may show what happened.

The viewer may not change what happened.

The viewer may not approve actions.

The viewer may not execute actions.

The viewer may not export, transform, upload, or act on ledger rows by default.

## Design principle

Simple surface.

Hard boundary.

Beautiful control.

No hidden hands.

The viewer should feel like a cockpit instrument panel:

- clear
- local
- read-only
- approval-aware
- audit-friendly
- impossible to confuse with an action engine

The app should make GARVIS easier to see.

The app should not make GARVIS harder to govern.

## Ledger object

The Stage 1 ledger is the local SQLite database created by:

`tools/stage1_senses_loop.py`

Default path:

`data/stage1_senses/decision_ledger.sqlite3`

The ledger records local sense snapshots and approval state.

Important fields include:

- `id`
- `timestamp`
- `input_hash`
- `candidates_json`
- `gate_scores_json`
- `decision`
- `approved_by_human`

The ledger is runtime body memory.

Raw runtime data must remain local unless a future directive explicitly exports a sanitized report.

## Read-only law

All future ledger viewers must open the SQLite database in mechanical read-only mode.

Required SQLite pattern:

`file:<path>?mode=ro`

The connection must use SQLite URI mode.

Read-only access must be enforced at the database connection level, not merely by developer promise.

A future implementation must include a regression test proving attempted write fails through the viewer path.

## No-network law

The ledger viewer path must make no network calls.

Forbidden in the viewer path:

- HTTP requests
- LLM API calls
- cloud sync
- telemetry
- remote logging
- analytics
- background upload
- webhook calls
- GitHub writes
- message sending

The viewer may read the local SQLite ledger.

The viewer may print or display local rows.

The viewer may not contact the outside world.

## Display-only law

The default viewer behavior must be display-only.

Default allowed display fields:

- `id`
- `timestamp`
- `decision`
- `approved_by_human`
- `input_hash`
- sensor/status summary

The viewer may show whether a row was approved.

The viewer may show whether the row is a snapshot, self-test, or sensor attempt.

The viewer may show high-level metadata.

The viewer must not display raw camera files, raw microphone files, private payloads, or full JSON blobs by default.

## Raw payload boundary

Runtime sensor payloads must be treated as sensitive.

Default view:

`metadata only`

Raw payloads may only be shown if a future directive adds an explicit per-row inspection mode.

That future mode must define:

- what row is being inspected
- what field is being opened
- whether raw data is displayed
- whether anything is copied or exported
- whether the user explicitly approved the inspection

No raw blob display by default.

No raw blob export by default.

No automatic summarization of raw payloads.

## No-write law

The viewer itself must write nothing.

Forbidden:

- no cache file
- no local viewer log
- no telemetry
- no modified database rows
- no export file by default
- no hidden marker files
- no last-viewed state
- no automatic report generation

Reading must leave no trace except normal operating-system/runtime effects outside the repository's control.

## No-export law

The viewer must not export rows by default.

No CSV export.

No JSON export.

No Markdown report export.

No GitHub-ready report generation.

No clipboard or copy automation.

A future export feature requires a separate directive and must be sanitized by design.

## Approval visibility

The viewer must make approval state visible.

Every displayed row should make clear whether:

- `approved_by_human = 0`
- `approved_by_human = 1`

The viewer must not treat unapproved rows as approved.

The viewer must not hide unapproved rows.

The viewer must not rewrite approval status.

## Human authority

The viewer is not an approval engine.

It may display approval state.

It may not approve a row.

It may not change approval state.

It may not infer approval from context.

It may not turn a displayed row into an action.

Future approval controls require a separate directive.

## Action Engine Reservation Clause

This contract does not prohibit future GARVIS action engines.

It only states that the App Ledger Viewer is not itself an action engine.

Future action capability must be added as a separate organ through separate directives.

No action capability may be smuggled into a viewer, dashboard, ledger reader, status display, report generator, or inspection tool.

A future action engine must define, at minimum:

1. action type
2. approval requirement
3. approval ledger requirement: every approval event must be written to the ledger with a unique approval ID before execution, and the action audit log must reference that ID
4. scope of allowed execution
5. files or systems touched
6. whether the outside world is affected
7. whether network access is used
8. whether secrets are required
9. whether sensor data is read
10. rollback or stop condition
11. audit log destination
12. expiration of approval
13. test proving unapproved execution fails

No ledger approval row, no execution.

One approval authorizes only the specific action described.

Approval does not create standing permission.

Approval does not authorize a whole category of future actions.

Approval does not authorize hidden sub-actions.

Approval does not authorize future actions unless explicitly renewed.

The viewer is a window.

A future action engine will be a separate hand.

The hand must be built later, named clearly, tested separately, and governed by Adrien.

## Staged action ladder boundary

The full Stage 0-7 action ladder is not part of this viewer contract.

The ladder is growth law, not viewer law.

It belongs in a separate Living Constitution amendment.

Planned future directive:

`DIRECTIVE-003A: Living Constitution Amendment - Staged Action Ladder`

This prevents constitutional growth law from being buried inside a component document.

## Hypercube self-design boundary

The hypercube brain may later be allowed to inspect its committed infrastructure and propose its own next designs.

That is not part of this viewer contract.

Self-design proposals are design speech, not action.

A future self-design organ must be created separately.

Planned future directive:

`DIRECTIVE-003B: Hypercube Brain Self-Design Proposal Contract`

That future contract may define how the repository brain can summarize itself, identify missing infrastructure, propose future PRs, and route proposals to Adrien for approval.

It may not execute those proposals without separate approval.

## Test requirements for future viewer implementation

The future implementation PR must include tests proving:

1. The viewer opens the SQLite ledger in read-only URI mode.
2. Attempted insert, update, or delete through the viewer connection fails.
3. The default output excludes raw JSON blobs unless explicitly requested by a future approved mode.
4. The viewer performs no network calls.
5. The viewer writes no output file by default.
6. The viewer preserves approval state exactly as stored.
7. Missing ledger file fails safely with a clear message.
8. Corrupt or unreadable ledger fails safely without modifying anything.

## Future implementation path

This contract authorizes no implementation.

The next possible implementation directive may be:

`DIRECTIVE-004: App Ledger Viewer CLI`

That future PR may add a tiny read-only CLI viewer.

It must obey this contract.

It must be local-only.

It must be display-only by default.

It must include the read-only write-failure regression test.

## Standing law

No consciousness claim.

No AGI claim.

No empirical outcome.

No autonomous action.

No background service activation.

No LLM call.

No network call.

No message sending.

No outside contact.

No camera output read.

No microphone output read.

No runtime sensor data committed.

No secrets committed.

No claim change.

No LL-07 run.

No LL-06 scoring.

No Kubota holdout contact.

No outcome-log edit.

No frozen preregistration edit.

No formula or dataset change.

No real participant data.

No new quantum job data.

No claim upgrade.

## Standing sentence

The App Ledger Viewer may become GARVIS's window into local memory.

It must not become GARVIS's hand.

The hand may be built later, but only as a separate named organ with ledger-bound approval, tests, audit trail, and human authority.
