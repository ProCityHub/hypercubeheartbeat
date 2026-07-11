# Cognitive Cycle Viewer Runbook

## Status

Implementation runbook for DIRECTIVE-008C.

This is a Stage 2 cognitive inspection instrument.

It is not an action engine.

It is not an autonomous runtime.

## Tool

`tools/cognitive_cycle_viewer.py`

## Purpose

The Cognitive Cycle Viewer reads a GARVIS cognitive cycle JSON and displays the thought pulse clearly.

It lets Adrien inspect:

- observation
- known organs
- hard constraints
- candidate thoughts
- case against each candidate
- risk of doing
- risk of not doing
- comparison
- selection
- uncertainty
- power request
- next smallest step
- output boundary

## Example

    python tools/cognitive_cycle_viewer.py \
      --repo . \
      --cycle tmp/cognitive_cycles/latest_cognitive_cycle.json

## Stage classification

This tool is Stage 2 cognitive inspection.

It reads a cognitive cycle.

It validates the cognitive cycle boundary.

It displays the cognitive cycle.

It writes nothing.

It executes nothing.

## No-execution law

The viewer does not execute candidate proposals.

It does not execute shell commands.

It does not call subprocess.

It does not run tests.

It does not run guard checks.

It does not commit.

It does not push.

## No-network law

The viewer makes no network calls.

It performs no HTTP requests.

It performs no LLM API calls.

It performs no cloud sync.

It performs no telemetry.

It performs no outside contact.

## Power boundary

A power request is displayed only.

A power request is not approval.

A recommendation is displayed only.

A recommendation is not action.

## Standing boundary

The viewer is a mirror.

It lets Adrien see GARVIS thought.

It does not move the world.

Adrien decides.
