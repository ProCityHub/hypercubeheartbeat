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

- What infrastructure exists?
- What should be built next?
- What gives Adrien more scientific power?
- What should not be built yet?

## Example

```bash
python tools/self_design_proposal_runner.py \
  --repo . \
  --output tmp/self_design_proposals/self_design_proposals.md
