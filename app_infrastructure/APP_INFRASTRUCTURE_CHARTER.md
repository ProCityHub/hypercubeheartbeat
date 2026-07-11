# App Infrastructure Charter

## Status

This charter governs future GARVIS app-facing work.

It is doc-only.

It does not create an app.

It does not activate a service.

It does not create autonomous action.

It does not call an LLM.

It does not touch external users.

## Purpose

The app infrastructure layer exists to make GARVIS usable without weakening the constitutional structure.

It must preserve:

- human authority
- GitHub memory
- local ledger integrity
- approval gates
- evidence boundaries
- claim boundaries
- sensor-data privacy
- no-secrets-in-repo rule
- no autonomous action

## Layer responsibilities

The app infrastructure layer may later provide:

- a mobile-first dashboard
- local status display
- local ledger display
- approval buttons
- refusal display
- notification display
- sensor availability display
- runbook links
- local-only runtime indicators
- future interface contracts for Lattice Thinker
- future interface contracts for LLM calls

## Layer non-responsibilities

The app infrastructure layer may not:

- decide claims
- upgrade claims
- score LL-06
- run LL-07
- touch Kubota holdout data
- bypass GitHub
- bypass guard checks
- bypass Adrien
- send messages without approval
- contact people without approval
- run background services without a directive
- commit runtime sensor files
- commit secrets
- hide actions from the ledger

## Interface law

Every future interface must make clear:

- what is being proposed
- what will change if approved
- what data will be read
- what data will be written
- whether the action touches the outside world
- whether the action writes to the local ledger
- whether the action writes to GitHub
- whether the action requires human approval

## Local-first law

The app infrastructure must prefer local-first operation.

Local-first means:

- local SQLite ledger first
- local Termux scripts first
- explicit network calls only when approved
- no hidden cloud dependency
- no background upload of sensor data
- no automatic sync of private runtime data

## Secrets law

No secrets may be committed.

Secrets include:

- API keys
- tokens
- passwords
- private authorization phrases
- private contact details
- raw camera captures
- raw microphone recordings
- private sensor outputs
- private relationship notes
- private identity material not approved for the repo

Future secret handling must use local environment variables, Android secure storage, or another approved secret path.

## Runtime data law

Runtime data stays local unless a future directive explicitly exports a sanitized report.

Runtime data includes:

- SQLite ledger database
- camera files
- microphone files
- notification results
- local status snapshots
- local debug outputs
- local app logs

The repository may contain schemas, runbooks, templates, and sanitized reports.

The repository must not accidentally absorb raw runtime body memory.

## Approval gate law

The app infrastructure may expose approval controls.

Approval controls must be explicit.

Approval controls must not be hidden behind ambiguous language.

High-stakes actions require stronger confirmation.

Examples of high-stakes actions:

- sending a message
- contacting a person
- publishing a claim
- editing a frozen file
- scoring holdout data
- running registered analysis
- using camera or microphone in a new context
- uploading sensor outputs
- changing a constitution or charter
- changing formulas or datasets

## Future app surfaces

Future app surfaces may include:

- Termux command wrapper
- local web dashboard
- Android notification interface
- lightweight mobile UI
- local-only API server
- ledger viewer
- approval queue
- GARVIS status panel
- Lattice Thinker candidate contest viewer

Each surface requires a separate directive before implementation.

## Standing law

No consciousness claim.

No AGI claim.

No empirical outcome.

No LL-07 run.

No LL-06 scoring.

No Kubota holdout contact.

No outcome-log edit.

No frozen preregistration edit.

No formula or dataset change.

No participant data.

No new quantum job data.

No synthetic output.

No claim upgrade.

No manipulation engine.

No coercive-use framing.

No autonomous action.

No committed secrets.

No committed runtime sensor data.

## Standing sentence

App Infrastructure may make GARVIS visible, usable, and auditable.

App Infrastructure may not make GARVIS autonomous, hidden, or harder for Adrien to govern.
