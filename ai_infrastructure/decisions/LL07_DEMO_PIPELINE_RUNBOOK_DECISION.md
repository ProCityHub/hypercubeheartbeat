# LL-07 Demo Pipeline Runbook Decision

## Decision

Human-approved creation of a demo pipeline runbook and sandbox report template for LL-07.

## Purpose

This decision allows the repo to document how to test the LL-07 machinery without treating synthetic demo output as evidence.

## What this PR may add

This PR may add:

- docs/LL07_DEMO_PIPELINE_RUNBOOK.md
- ai_infrastructure/reports/LL07_DEMO_PIPELINE_REPORT_TEMPLATE.md
- ai_infrastructure/decisions/LL07_DEMO_PIPELINE_RUNBOOK_DECISION.md

## What this PR may not do

This PR may not:

- collect real rating data
- add real completed ratings
- run a real LL-07 empirical analysis
- commit synthetic ratings
- commit synthetic analyzer output
- record an LL-07 outcome
- edit the outcome log
- use synthetic data as evidence
- edit the frozen LL-07 stimulus list
- edit the frozen LL-07 preregistration
- edit the frozen LL-07 analysis script
- touch Kubota holdout data
- score LL-06 material
- upgrade any claim

## Future path

A later sandbox step may run the demo command locally.

A future empirical step still requires real independent raters, a ratings manifest, and a separate human-approved analysis directive.
