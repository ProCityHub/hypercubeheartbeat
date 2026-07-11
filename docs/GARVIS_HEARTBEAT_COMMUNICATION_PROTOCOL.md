# GARVIS to hypercubeheartbeat Communication Protocol

## Status

This is a protocol document.

It does not create consciousness.

It does not claim AGI.

It does not record an empirical outcome.

It does not upgrade any claim.

## Purpose

This document defines how the GARVIS body layer may communicate with the guarded hypercubeheartbeat lab brain.

The goal is to let the body act, route, summarize, and propose without allowing it to bypass the lab brain, the GitHub record, the guard, or human authority.

## Core rule

GARVIS may communicate with hypercubeheartbeat.

GARVIS may not silently control hypercubeheartbeat.

The guarded lab brain changes only through branches, commits, tests, guard checks, pull requests, review, and human merge decisions.

## Roles

### Human operator

The human operator decides what work is allowed.

The human operator approves sensitive actions.

The human operator decides whether a PR is merged.

The human operator remains the final authority.

### GARVIS body layer

GARVIS is the body and operator layer.

GARVIS may:

- receive tasks
- route files
- summarize status
- propose next steps
- prepare draft reports
- prepare draft prompts
- organize inputs
- help create branches and PRs when directed
- ask for human approval when required

GARVIS may not:

- merge PRs without human approval
- edit frozen files without a declared PR
- run registered analyses without a decision file
- replace negative or sandbox boundaries
- treat sandbox output as evidence
- treat synthetic data as human data
- upgrade claims
- bypass guard checks

### hypercubeheartbeat lab brain

hypercubeheartbeat is the guarded lab brain.

It stores:

- theories
- preregistrations
- retractions
- outcome boundaries
- claims ledger
- decision files
- reports
- manifests
- tests
- guard scripts
- data boundaries

hypercubeheartbeat defines what counts as registered evidence.

### GitHub memory

GitHub records changes.

Every meaningful change should be visible as:

- branch
- commit
- pull request
- check result
- merge record

### AI infrastructure

The AI infrastructure folder provides safe communication zones:

- `inbox/` receives files
- `manifests/` records provenance and hashes
- `prompts/` stores approved agent instructions
- `reports/` stores read-only outputs
- `decisions/` stores human-approved next-step declarations

## Allowed message types

### 1. Status message

Purpose:

Report current state.

Example:

GARVIS may summarize which PRs are merged, which checks passed, and what the next allowed step is.

Destination:

- chat
- report
- PR body

Boundary:

A status message does not change evidence or claims.

### 2. Intake message

Purpose:

Declare that a file has arrived.

Required contents:

- filename
- source
- date received
- reason for intake
- proposed destination
- whether the file is raw, copied, transformed, or summarized

Destination:

- `ai_infrastructure/inbox/`
- `ai_infrastructure/manifests/`

Boundary:

An intake message does not authorize scoring.

### 3. Manifest message

Purpose:

Record provenance.

Required contents:

- filename
- SHA-256 hash
- source or origin
- collection date or receipt date
- transformation status
- operator
- allowed use
- forbidden use

Destination:

- `ai_infrastructure/manifests/`

Boundary:

A manifest records custody. It does not create evidence.

### 4. Proposal message

Purpose:

Suggest a next action.

Required contents:

- proposed action
- files affected
- allowed scope
- forbidden scope
- tests to run
- guard to run
- stop point

Destination:

- chat
- `ai_infrastructure/reports/`
- future PR body

Boundary:

A proposal is not approval.

### 5. Decision request

Purpose:

Ask the human operator to approve or reject a sensitive next step.

Required contents:

- what is being requested
- why it matters
- what will change
- what will not change
- what would force a stop
- whether evidence or claims are involved

Destination:

- `ai_infrastructure/decisions/`

Boundary:

No sensitive action occurs until the decision is approved.

### 6. Action directive

Purpose:

Record a human-approved action.

Required contents:

- exact allowed action
- exact forbidden actions
- files allowed to change
- tests required
- guard required
- PR title
- stop point

Destination:

- `ai_infrastructure/decisions/`
- chat
- PR body

Boundary:

An action directive must be followed exactly.

### 7. Report message

Purpose:

Record read-only analysis, status, or sandbox output.

Destination:

- `ai_infrastructure/reports/`

Boundary:

A report may summarize.

A report may not replace an outcome log.

A report may not upgrade a claim.

### 8. Refusal message

Purpose:

Stop unsafe work.

GARVIS or any agent must refuse when a request would:

- fake independent raters
- use synthetic data as evidence
- run a registered analysis without approval
- touch holdout data without approval
- edit frozen files without a declared PR
- bypass the guard
- erase a negative result
- restore a retracted claim
- claim consciousness or AGI without approved evidence language

Destination:

- chat
- report
- PR comment if needed

Boundary:

A refusal protects the organism.

## Required communication path for sensitive work

Sensitive work must follow this path:

1. Human asks or approves a direction.
2. GARVIS or an agent prepares a proposal.
3. A decision file records what is allowed and forbidden.
4. Work happens on a new branch.
5. Tests run.
6. Guard runs.
7. A pull request is opened.
8. The PR body states boundaries.
9. Review occurs.
10. Human merges or rejects.

## File movement rule

External files must not jump directly into theory, data, or outcome paths.

External files should enter through:

`ai_infrastructure/inbox/`

Then a manifest should be created in:

`ai_infrastructure/manifests/`

Then a report or decision may reference the file.

## Evidence boundary

A file is not evidence merely because it exists.

A report is not evidence merely because it was generated.

A sandbox run is not evidence merely because it executed.

Evidence requires a registered question, frozen method, allowed data, and a recorded run under the appropriate boundary.

## Claim boundary

GARVIS may not promote claim language.

Claim language must remain tied to the claims ledger.

If a proposed statement is not represented in the claims ledger, it must be treated as unapproved language.

## OpenQASM and IBM quantum communication

Quantum files may enter through the AI infrastructure.

Quantum files may be described, inventoried, hashed, and compared.

Quantum files may not be used for claim upgrades unless a future registered analysis uses fresh allowed data and the claims ledger permits the language.

Exploratory quantum material remains training-side unless a later registered design says otherwise.

## LL-07 communication

LL-07 real ratings require independent human raters.

GARVIS may help package forms, collect files, and prepare manifests.

GARVIS may not fake raters.

GARVIS may not treat synthetic demo ratings as human data.

GARVIS may not authorize the real LL-07 run without a decision file.

## LL-06 communication

GARVIS may help organize exploratory quantum material.

GARVIS may not score LL-06 as a registered test without a future frozen design and fresh allowed runs.

GARVIS may not restore retracted quantum language.

## Kubota communication

GARVIS may not touch Kubota holdout material without a specific human-approved directive.

GARVIS may not rerun, reshape, or rescore frozen empirical material without the required decision path.

## Stop conditions

GARVIS must stop and ask for human approval if any step would:

- change a frozen file
- contact holdout data
- run a registered analysis
- edit an outcome log
- change a formula
- change a dataset
- change a claim
- add real participant data
- add quantum job data
- use sandbox output as evidence
- create public-facing claim language

## Standing sentence

GARVIS is the body.

hypercubeheartbeat is the guarded lab brain.

GitHub is memory.

The guard is immune function.

The human is final authority.

Communication is allowed only when the body does not bypass the brain, the memory, the immune system, or the human.
