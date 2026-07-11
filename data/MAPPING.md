# Phase 4 Step 2a — Declared Mapping Before Run

Source URL: https://osf.io/8cj69

Raw files inspected:

- data/Task2_Export.csv
- data/codebook.pdf

This file declares the mapping decision before any scoring run.

No preregistration_test.py execution was performed in this PR.

## Required preregistered format

The repository data intake rule requires:

offer, stake, rt_ms, accept

The converter requires a reaction-time source column through --rt-col.

## Raw Task2_Export.csv structure

Task2_Export.csv contains five numeric columns and no header row.

Observed first rows:

1,1,5,1,1
1,2,5,1,1
1,3,5,1,1
1,4,5,1,1
1,5,5,1,1

## Column assessment

| Raw column | Interpreted meaning | Target field | Decision |
|---|---|---|---|
| Column 1 | Participant / subject index | none | Not a preregistered target field |
| Column 2 | Trial index within participant/task | none | Not a preregistered target field |
| Column 3 | Offer value / offer size candidate | offer candidate | Usable only if the dataset were otherwise eligible |
| Column 4 | Accept/reject decision candidate | accept candidate | Usable only if the dataset were otherwise eligible |
| Column 5 | Feedback / facial-expression / condition code candidate | none | Not a preregistered target field |

## Missing required field

No usable reaction-time column was identified.

Because the frozen preregistered pilot requires rt_ms, this dataset is ineligible as frozen.

## Converter invocation

No converter invocation is declared.

Reason: data/convert_probe.py requires --rt-col, but no reaction-time source column exists in Task2_Export.csv.

This PR does not run python data/convert_probe.py.

This PR does not create data/ug_probe.csv.

## Decision branch

INELIGIBLE.

This dataset must not be bent to fit the preregistered pilot.

No proxy column may be substituted for reaction time.

No reaction-time value may be generated, inferred, fabricated, copied from EEG time-window text, or replaced with trial number, condition code, offer value, or decision code.

## Candidate next-attempt data sources

Search for an open Ultimatum Game behavioral dataset that explicitly includes:

offer, stake, responder reaction time, accept/reject decision

Candidate search targets:

1. OSF: Ultimatum Game reaction time accept reject offer
2. Harvard Dataverse: Ultimatum Game response time behavioral data
3. OpenNeuro/BIDS: ultimatum game events.tsv response_time

Any candidate dataset must be rejected unless its codebook or event file explicitly provides responder reaction time.
