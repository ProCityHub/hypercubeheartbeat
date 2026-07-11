# Experiment Manifest Viewer Runbook

## Status

Implementation runbook for DIRECTIVE-007B.

This is a Stage 2 dry-run validation instrument.

It is not an experiment runner.

It is not an execution engine.

## Tool

`tools/experiment_manifest_viewer.py`

## Purpose

The Experiment Manifest Viewer reads a GARVIS experiment manifest JSON, validates the required scientific-method boundaries, and displays the manifest in human-readable form.

It is designed to answer:

- Is the manifest structurally complete?
- Is the null model present?
- Is the claim boundary fixed?
- Is the dry-run boundary intact?
- Is the manifest commit SHA locked?
- Is approval missing?
- What command would a future approved run execute?
- What claims are forbidden?

## Example

    python tools/experiment_manifest_viewer.py \
      --repo . \
      --schema ai_infrastructure/schemas/experiment_manifest_schema_v1.json \
      --manifest path/to/manifest.json

## Stage classification

This tool is Stage 2 dry-run validation.

It reads a manifest.

It validates a manifest.

It prints a would-run command.

It cannot execute the would-run command.

## Pre-registration boundary

The viewer warns if the manifest commit SHA is not locked.

A future result cannot be valid unless it cites a manifest SHA that predates the run.

## Null model boundary

The viewer requires the null model section.

No null model means no valid run.

## Claim boundary

The viewer requires the fixed claim vocabulary:

- exploratory
- suggestive
- supported
- retracted

Free-text claim upgrades are not allowed.

Claim upgrades require a new manifest, new approval, and new run.

## Dry-run means cannot-run

The viewer displays the exact command a future approved execution may run.

It does not run that command.

It does not import subprocess.

It does not call shell execution.

It does not run tests.

It does not run guard checks.

## No-network law

The viewer makes no network calls.

It performs no HTTP requests.

It performs no LLM API calls.

It performs no cloud sync.

It performs no telemetry.

It performs no outside contact.

## Standing boundary

A manifest is not proof.

A validation pass is not evidence.

A would-run command is not execution.

A dry-run is not a result.

Execution requires a separate Stage 3 approved event.
