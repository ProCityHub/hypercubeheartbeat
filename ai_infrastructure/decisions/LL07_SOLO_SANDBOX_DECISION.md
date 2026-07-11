# LL-07 Solo Sandbox Decision

## Decision

Human-approved creation of a solo sandbox and synthetic demo pipeline for LL-07.

## Purpose

This decision allows single-operator development while preserving the preregistered requirement for independent raters.

## What this PR may add

This PR may add:

- `docs/LL07_SOLO_SANDBOX.md`
- `tools/ll07_generate_demo_ratings.py`
- `tests/test_ll07_demo_generator.py`
- `ai_infrastructure/decisions/LL07_SOLO_SANDBOX_DECISION.md`

## What this PR may do

This PR may:

- create a synthetic demo rating generator
- test the generator
- test that synthetic demo ratings can flow through the existing analyzer
- clarify that solo work is sandbox-only

## What this PR may not do

This PR may not:

- collect real rater data
- add real completed ratings
- run a real LL-07 empirical analysis
- record an LL-07 outcome
- edit the outcome log
- count one person as multiple raters
- use synthetic data as evidence
- edit the frozen LL-07 stimulus list
- edit the frozen LL-07 preregistration
- edit Kubota holdout data
- score LL-06 material
- upgrade any claim

## Future path

When real independent raters exist, a separate PR may intake completed ratings.

Until then, solo work remains sandbox work.
