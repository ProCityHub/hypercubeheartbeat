# Triadic Deep Question Planbook Runbook
> **Post-reconciliation name:** The active audit-only implementation is
> `tools/audit_planbook.py`. The former
> `tools/audit_planbook.py` path was removed during repository
> reconciliation. The audit-only implementation is `tools/audit_planbook.py`
> and must not be used as the conversational interface.

## Status

Implementation runbook for DIRECTIVE-008I.

This is a Stage 2 cognitive planbook refresh.

It teaches Heartbeat a deep-question mode.

## Tool updated

`tools/audit_planbook.py`

## Purpose

The previous Heartbeat could answer build-planning questions, but when asked a deep question after the Dream-to-Lab Bridge merged, it fell back to recommending the already-built memory initialization CLI.

DIRECTIVE-008I adds a second cognitive mode.

## Mode 1: Build-planning mode

Build-planning mode asks:

- What organ should GARVIS build next?
- What file should be created?
- What is the next smallest safe implementation step?

## Mode 2: Deep-question mode

Deep-question mode asks:

- What unresolved question belongs in the Dream Chamber?
- How should the Hypothesis Forge translate it?
- What would the Lab Record need to test it?
- What must remain forbidden to claim?

## Triadic output

Deep-question mode maps each candidate into:

- Dream Chamber
- Hypothesis Forge
- Lab Record
- Forbidden claims

## First selected deep question

The first selected deep question is:

`What is thinking, operationally, inside GARVIS?`

## Boundary

This mode does not prove thinking.

It does not prove consciousness.

It does not prove AGI.

It does not execute experiments.

It does not write memory.

It produces advisory cognitive output only.

## Standing sentence

The deep-question planbook lets GARVIS ask harder questions without letting hard questions become false claims.
