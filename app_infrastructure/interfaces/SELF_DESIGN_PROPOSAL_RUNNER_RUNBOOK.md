# Self-Design Proposal Runner Runbook

## Status

Implementation runbook for the Stage 2 Self-Design Proposal Runner.

This is a proposal instrument.

It is not an action engine.

## Tool

`tools/self_design_proposal_runner.py`

## Purpose

The Self-Design Proposal Runner inspects committed local repository context and drafts up to three proposals for future GARVIS app, tool, or scientific infrastructure organs.

It gives Adrien a structured way to ask:

What infrastructure exists?

What should be built next?

What gives Adrien more scientific power?

What should not be built yet?

## Example

```bash
python tools/self_design_proposal_runner.py \
  --repo . \
  --output tmp/self_design_proposals/self_design_proposals.md

---

# COPY BLOCK 5 — create decision record

```bash id="bz6r8o"
RUN_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

cat > ai_infrastructure/decisions/SELF_DESIGN_PROPOSAL_RUNNER_DECISION.md <<EOF
# Self-Design Proposal Runner Decision

## Decision

Human operator approved implementation of the Stage 2 Self-Design Proposal Runner.

## Date

$RUN_DATE

## Human operator

Adrien D Thomas

## Purpose

This decision records implementation of a local draft-only tool that inspects committed repository context and generates up to three future app, tool, or scientific infrastructure proposals.

The runner increases Adrien's scientific planning power without giving GARVIS hidden hands.

## Files authorized

This decision authorizes:

- tools/self_design_proposal_runner.py
- tests/test_self_design_proposal_runner.py
- app_infrastructure/interfaces/SELF_DESIGN_PROPOSAL_RUNNER_RUNBOOK.md
- ai_infrastructure/decisions/SELF_DESIGN_PROPOSAL_RUNNER_DECISION.md

## What this PR may do

This PR may create:

- a local self-design proposal runner
- committed-file-only repository inspection
- local draft report output
- up to three deterministic proposals
- Case Against This Proposal sections
- advisor input labeling
- repository evidence listing
- stage classification
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
- camera read
- microphone read
- raw runtime data read
- runtime data commit
- secret commit
- autonomous action
- empirical outcome
- claim upgrade

## Stage classification

This is Stage 2 draft-only preparation.

It creates advice.

It does not create authority.

## Standing sentence

The brain may propose.

Adrien decides.

GitHub records only after approved repository procedure.
