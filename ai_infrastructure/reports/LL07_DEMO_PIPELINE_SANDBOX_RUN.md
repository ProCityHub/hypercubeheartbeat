# LL-07 Demo Pipeline Sandbox Run

SANDBOX ONLY.

Synthetic demo ratings are not human data.

Synthetic demo ratings are not evidence.

This report does not record an empirical LL-07 result.

## Run information

Date:

2026-07-11T16:33:29Z

Operator:

Adrien D Thomas

Repository commit:

f8a462e17af2ce2690cd534d1702da0be8dac7ef

## Commands run

The sandbox demo generator was run locally in /tmp.

The sandbox analyzer was run locally in /tmp.

The demo used:

- 12 synthetic demo raters
- 100 frozen LL-07 stimulus words
- 3 attention-check rows per synthetic demo rater
- 19 demo permutations
- seed 7707

## Observed machinery checks

The sandbox run confirmed:

- generator completed
- analyzer completed
- demo ratings file existed in /tmp
- demo output JSON existed in /tmp
- JSON included n_valid_raters
- JSON included n_words
- JSON included dimensions
- JSON included pc1_variance_ratio
- JSON included criterion_met

## Non-committed files

The following files were generated locally only and were not committed:

- /tmp/ll07_demo/ll07_demo_ratings.csv
- /tmp/ll07_demo/ll07_demo_output.json

## Boundary statement

This was a sandbox machinery check only.

It did not test the LL-07 hypothesis.

It did not collect independent human ratings.

It did not create an empirical outcome.

It did not edit the outcome log.

It did not upgrade any claim.

## Standing sentence

The LL-07 demo pipeline can test machinery, but only real independent raters can test the preregistered hypothesis.
