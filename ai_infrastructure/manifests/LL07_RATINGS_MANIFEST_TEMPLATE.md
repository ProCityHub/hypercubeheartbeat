# LL-07 Ratings Manifest Template

## Status

Template only.

No rating data is included in this file.

No analysis is run from this file.

## Ratings file

Filename:

`ai_infrastructure/inbox/ll07_ratings/ll07_ratings_raw.csv`

SHA-256:

`TO_BE_FILLED_AFTER_FILE_EXISTS`

File size:

`TO_BE_FILLED_AFTER_FILE_EXISTS`

## Collection information

Collection method:

`TO_BE_FILLED`

Collection date range:

`TO_BE_FILLED`

Number of submitted raters:

`TO_BE_FILLED`

Number of expected rows:

`submitted_raters * 103 rows, excluding header`

Anonymization method:

`TO_BE_FILLED`

Rater independence statement:

`TO_BE_FILLED`

Rater blinding statement:

`TO_BE_FILLED`

## Frozen source files

The collection must use:

- `data/ll07_word_stimuli.csv`
- `data/ll07_rating_template.csv`
- `docs/LL07_RATER_INSTRUCTIONS.md`
- `docs/LL07_WORD_COORDINATE_PREREGISTRATION.md`

## Required columns

The ratings CSV must contain:

- rater_id
- item_id
- word
- O
- A
- B

## Attention-check rows required per rater

Each rater must include:

- `ATTN_O`
- `ATTN_A`
- `ATTN_B`

## Preregistered analysis script

The frozen analysis script is:

`tools/ll07_analyze_ratings.py`

Do not run it during intake unless a separate decision file explicitly authorizes the single preregistered run.

## Approval

Human approval for intake:

`TO_BE_FILLED`

Human approval for analysis run:

`NOT_APPROVED_IN_THIS_TEMPLATE`

## Notes

This manifest template prepares future LL-07 data intake.

It does not authorize analysis.

It does not record an outcome.

It does not upgrade any claim.
