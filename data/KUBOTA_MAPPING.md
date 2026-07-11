# Phase 4 Step 2a-3 — Kubota Dataset Mapping Declaration

Source URL: https://osf.io/ugdcz

Dataset: Kubota et al. 2013 — The price of racial bias.

This file declares the planned mapping before any conversion, scoring, or preregistered test execution.

No converter was run in this PR.
No data/ug_probe.csv file was created in this PR.
No preregistration_test.py execution was performed in this PR.

## Documentation screened

Documentation-only screening found:

- Variables.docx
- IATRawData.zip
- Ultimatum_manuscript_2.sav
- UltimatumRawData.zip

Raw data was not opened during the eligibility-search PR.

The next conversion PR may open the raw data only to apply the mapping declared here.

## Required preregistered format

The frozen preregistered pilot requires:

offer, stake, rt_ms, accept

## Documented source variables

From Variables.docx:

- pTOffer: [time, actualOffer]
- actualOffer: offer amount
- utDecision: [time, actualDecision]
- time = decisionTime - startTime
- actualDecision: 1 = Accept and 2 = Reject
- NaN for no response
- TOutcome: proposer payout outcome in dollars
- trialOutcome: responder payout outcome in dollars
- origOffer: trial offer drawn from the distribution of offers
- futureOffer: participant future offers
- five splits of $10

## Declared mapping

| Preregistered field | Declared source | Transformation |
|---|---|---|
| offer | actualOffer | Use documented offer amount directly. |
| stake | fixed documented stake | Use constant 10, based on five splits of $10. |
| rt_ms | utDecision.time - pTOffer.time | Compute responder decision latency from documented decision time minus documented offer time. Convert to milliseconds if source timing is stored in seconds. |
| accept | actualDecision | Normalize 1 = Accept to 1. Normalize 2 = Reject to 0. Exclude NaN or no-response trials. |

## Eligibility decision

ELIGIBLE FOR CONVERSION PR.

Reason:

The documentation contains responder-side offer, decision, stake context, and timing fields sufficient to declare a preregistered mapping before conversion.

## Conversion plan for the next PR

The next PR may create a dataset-specific conversion script if the existing converter cannot express derived reaction time.

The next PR must do only this:

1. Open the raw Kubota data.
2. Apply this mapping exactly.
3. Produce data/ug_probe.csv with columns: offer, stake, rt_ms, accept.
4. Commit data/ug_probe.csv before any test run.
5. Include source URL https://osf.io/ugdcz in the commit message.

## Forbidden in this PR

This PR does not:

- open raw .sav or .zip data,
- create data/ug_probe.csv,
- run any converter,
- run preregistration_test.py,
- score the Lattice Law,
- report SUPPORTED or NOT_SUPPORTED.

## Claim discipline

This mapping declaration is not evidence for the Lattice Law.

This mapping declaration is not a result.

This mapping declaration does not prove consciousness.

This mapping declaration does not prove AGI.

The only purpose of this file is to freeze the mapping decision before conversion and before the single preregistered run.
