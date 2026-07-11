# Cognitive Cycle Runner Runbook

## Status

Implementation runbook for DIRECTIVE-008B.

This is a Stage 2 cognitive draft instrument.

It is not an action engine.

It is not an autonomous runtime.

## Tool

`tools/cognitive_cycle_runner.py`

## Purpose

The Cognitive Cycle Runner creates one bounded GARVIS thought cycle.

It inspects committed repository structure with read-only git commands, detects known GARVIS organs, forms candidate next thoughts, argues against each candidate, compares tradeoffs, selects a next smallest step, records uncertainty, records power-request logic, and writes local draft output.

## Example

    python tools/cognitive_cycle_runner.py \
      --repo . \
      --output-dir tmp/cognitive_cycles \
      --stdout

## Outputs

Default local draft outputs:

- `tmp/cognitive_cycles/latest_cognitive_cycle.json`
- `tmp/cognitive_cycles/latest_cognitive_cycle.md`

These outputs are local draft artifacts.

They should not be committed unless a future directive explicitly approves that.

## Cognitive cycle

A cycle contains:

- operator context
- input state
- observation summary
- candidate thoughts
- evidence basis
- case against each candidate
- risk of doing
- risk of not doing
- comparison
- selection
- uncertainty
- power request
- next smallest step
- evolution contract
- output boundary

## Evolution behavior

The runner lets GARVIS:

- self-observe
- self-propose
- self-criticize
- compare candidate futures
- request more power in principle

The runner does not let GARVIS:

- self-execute
- modify repository files
- commit
- push
- contact the outside world
- upgrade claims

## Read-only inspection boundary

The runner may use read-only git commands such as:

- `git rev-parse`
- `git ls-files`
- `git status --short`

It must not run tests.

It must not run guard checks.

It must not run method commands.

It must not execute candidate proposals.

## No-network law

The runner makes no network calls.

It performs no HTTP requests.

It performs no LLM API calls.

It performs no cloud sync.

It performs no telemetry.

It performs no outside contact.

## Standing boundary

A cognitive cycle is a thought pulse.

A thought pulse is not an action.

A recommendation is not approval.

A power request is not permission.

Adrien decides.
