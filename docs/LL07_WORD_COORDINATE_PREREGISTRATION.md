# LL-07 Word-Coordinate Preregistration

## Status

This document preregisters the LL-07 word-coordinate study.

No rating data has been collected in this PR.

No empirical result is claimed in this PR.

No theory claim is upgraded in this PR.

## Purpose

LL-07 tests whether independent human raters can reliably place words into three semantic coordinates:

- O: Observer
- A: Actor
- B: Bridge

This is the H1 test for the language layer.

It tests whether O/A/B coordinates can be measured reliably.

It does not test whether the phi-exponent formula predicts an external target.

It does not test AGI.

It does not test consciousness.

It does not test physics.

## Hypothesis structure

### H1: Coordinate reliability

Words can be rated by independent raters on O/A/B dimensions with reliable agreement.

### H2: Exponent weighting

Given frozen O/A/B coordinates, a later study may test whether the exponent vector 1, 1/phi, 1/phi^2 predicts an external target better than alternatives.

H2 is not tested in LL-07.

LL-07 tests H1 only.

## Stimulus list

The stimulus list is frozen in:

`data/ll07_word_stimuli.csv`

It contains 100 words:

- 25 mystical words
- 25 fantasy words
- 25 technical words
- 25 mundane words

The list avoids theory-keyword loading.

It does not include the words observer, actor, bridge, lattice, phi, consciousness, or AGI.

## Raters

Minimum valid rater count:

N = 12

Raters must be independent.

Raters must not be told the theory name, formula, expected direction, or desired result.

## Rating task

Each rater assigns every word three independent ratings from 0 to 100:

- O: Observer
- A: Actor
- B: Bridge

The three ratings do not need to add to 100.

## Attention checks

The rating form must include attention-check rows.

Attention-check rows are not part of the 100 stimulus words.

A rater is excluded if any attention-check value misses its target by more than 5 points.

## Exclusion rules

Exclude a rater if:

- the rater fails any attention check
- the rater does not rate all 100 stimulus words
- any O, A, or B rating is missing
- any O, A, or B rating is outside 0 to 100

No word is excluded after data collection.

## Primary statistic

For each dimension O, A, and B, compute Kendall's W across raters.

Kendall's W measures agreement among raters in ranking the words on that dimension.

## Permutation test

For each dimension, compute a permutation p-value using 10000 permutations.

Permutation method:

Within each rater, shuffle the ratings across words for that dimension.

This preserves each rater's rating distribution but destroys word-level agreement.

Use seed 7707.

## Primary criterion

The LL-07 criterion is met only if all of the following are true:

- O dimension Kendall's W >= 0.50
- A dimension Kendall's W >= 0.50
- B dimension Kendall's W >= 0.50
- O permutation p < 0.01
- A permutation p < 0.01
- B permutation p < 0.01
- PC1 variance ratio < 0.80

## Redundancy check

Compute mean O/A/B coordinates per word.

Run a principal component analysis on the 100 by 3 mean-coordinate matrix.

If the first principal component explains 80 percent or more of the variance, the triadic coordinate structure is treated as collapsed into a generic rating dimension.

In that case, the LL-07 criterion is not met even if Kendall's W is high.

## Secondary exploratory analysis

The four word strata may be compared in O/A/B space.

This is exploratory only.

It does not affect the primary criterion.

## One-run rule

After valid rating data is collected and manifested, the frozen analysis script must be run once.

The output must be recorded the same day.

No rerun may replace the first valid output.

## Files frozen by this preregistration PR

- `data/ll07_word_stimuli.csv`
- `docs/LL07_RATER_INSTRUCTIONS.md`
- `docs/LL07_WORD_COORDINATE_PREREGISTRATION.md`
- `tools/ll07_analyze_ratings.py`
- `tests/test_ll07_word_coordinate_analysis.py`

## Forbidden

This PR does not collect data.

This PR does not run the empirical LL-07 analysis.

This PR does not upgrade any claim.

This PR does not edit outcome logs.

This PR does not touch Kubota holdout data.

This PR does not score LL-06.

## Plain-language boundary

LL-07 asks whether people can reliably rate words on O/A/B semantic dimensions.

It does not ask whether mystical words are true.

It does not ask whether the full Lattice Law is true.

It is a coordinate-reliability test only.
