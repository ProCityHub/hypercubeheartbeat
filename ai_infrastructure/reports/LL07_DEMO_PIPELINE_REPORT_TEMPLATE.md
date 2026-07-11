# LL-07 Demo Pipeline Report Template

SANDBOX ONLY.

Synthetic demo ratings are not human data.

Synthetic demo ratings are not evidence.

This report template does not record an empirical LL-07 result.

## Run information

Date:

TO_BE_FILLED

Operator:

TO_BE_FILLED

Repository commit:

TO_BE_FILLED

## Commands run

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

## Expected checks

- generator completed
- analyzer completed
- demo ratings file existed in /tmp
- demo output JSON existed in /tmp
- no demo CSV was committed
- no demo JSON was committed
- no outcome log was edited
- no claim was upgraded

## Boundary statement

This was a sandbox machinery check only.

It did not test the LL-07 hypothesis.

It did not collect independent human ratings.

It did not create an empirical outcome.

## Notes

TO_BE_FILLED
