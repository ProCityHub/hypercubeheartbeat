# LL-07 Demo Pipeline Runbook

## Status

This is a sandbox runbook.

It does not collect real LL-07 rating data.

It does not authorize the empirical LL-07 run.

It does not create an outcome.

It does not upgrade any claim.

## Purpose

This runbook explains how to test the LL-07 pipeline using synthetic demo ratings.

The purpose is to verify that:

- the frozen stimulus list loads
- synthetic demo rows can be generated
- attention-check rows are present
- the analyzer can process a correctly shaped ratings file
- the pipeline can produce a JSON output file

## Boundary

Synthetic demo ratings are not human data.

Synthetic demo ratings are not evidence.

Synthetic demo ratings must not enter the outcome log.

Synthetic demo ratings must not be described as an empirical LL-07 result.

## Recommended sandbox command

Run this only for pipeline testing:

mkdir -p /tmp/ll07_demo

python tools/ll07_generate_demo_ratings.py \
  --stimuli data/ll07_word_stimuli.csv \
  --output /tmp/ll07_demo/ll07_demo_ratings.csv \
  --n-raters 12

python tools/ll07_analyze_ratings.py \
  --stimuli data/ll07_word_stimuli.csv \
  --ratings /tmp/ll07_demo/ll07_demo_ratings.csv \
  --output /tmp/ll07_demo/ll07_demo_output.json \
  --permutations 19 \
  --seed 7707

cat /tmp/ll07_demo/ll07_demo_output.json

## Why the demo uses fewer permutations

The real preregistered LL-07 analysis uses 10000 permutations.

The demo run may use a small number such as 19 because the purpose is pipeline testing only.

The demo output is not evidence.

## What to inspect

After the sandbox command, check that:

- the generator prints a warning that synthetic data is not evidence
- the output ratings file exists
- the output JSON exists
- the JSON includes n_valid_raters
- the JSON includes n_words
- the JSON includes dimensions
- the JSON includes pc1_variance_ratio
- the JSON includes criterion_met

## Forbidden use

Do not:

- commit /tmp/ll07_demo/ll07_demo_ratings.csv
- commit /tmp/ll07_demo/ll07_demo_output.json
- copy synthetic results into the outcome log
- call the demo an empirical result
- use the demo to update claims
- change the frozen LL-07 preregistration from the demo output
- change the frozen LL-07 analysis script from the demo output
- touch Kubota holdout data
- score LL-06 material

## Safe report language

Allowed:

- The synthetic demo pipeline executed.
- The generator produced correctly shaped demo rows.
- The analyzer accepted the demo file.
- The pipeline is technically runnable.

Forbidden:

- LL-07 succeeded.
- LL-07 failed.
- The lattice model is supported.
- The demo validates the theory.
- The demo is evidence.
- The demo is human data.

## Standing sentence

The LL-07 demo pipeline may test machinery, but only real independent raters can test the preregistered hypothesis.
