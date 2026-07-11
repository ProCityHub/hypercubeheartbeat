# LL-07 Data Collection Decision

## Decision

Human-approved preparation for LL-07 data collection infrastructure.

This decision authorizes creation of:

- a rating template
- collection instructions
- a ratings manifest template
- this decision file

## What is allowed

This PR may:

- add `data/ll07_rating_template.csv`
- add `docs/LL07_DATA_COLLECTION_INSTRUCTIONS.md`
- add `ai_infrastructure/manifests/LL07_RATINGS_MANIFEST_TEMPLATE.md`
- add `ai_infrastructure/decisions/LL07_DATA_COLLECTION_DECISION.md`

## What is not allowed

This PR may not:

- collect rating data
- add real rater responses
- run `tools/ll07_analyze_ratings.py`
- edit the LL-07 stimulus list
- edit the LL-07 rater instructions
- edit the LL-07 preregistration
- edit the LL-07 analysis script
- edit outcome logs
- touch Kubota holdout data
- score LL-06 material
- upgrade any claim

## Future required PRs

A future PR may intake collected ratings into:

`ai_infrastructure/inbox/ll07_ratings/`

A later separate PR may authorize the single preregistered LL-07 run after:

- ratings are collected
- ratings are manifested
- hashes are recorded
- the human operator explicitly approves the run

## Stop rule

After this infrastructure PR merges, the next step is human data collection outside the repo or a ratings-intake PR.

No empirical LL-07 output exists until the frozen script is run once under a separate directive.
