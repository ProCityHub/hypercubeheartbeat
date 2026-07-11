# Phase 4 Step 2a-2 — Eligibility Search Log

This log records dataset eligibility screening before any conversion, scoring, or preregistered test execution.

No converter was run in this PR.

No `data/ug_probe.csv` file was created in this PR.

No `preregistration_test.py` execution was performed in this PR.

## Eligibility criteria

These criteria are fixed before any candidate mapping or scoring:

1. Trial-level rows.
2. Responder-side data.
3. Offer amount.
4. Stake amount, or a documented fixed stake.
5. Responder reaction time in milliseconds, or documented timing fields that can be declared before conversion and converted to milliseconds.
6. Accept/reject outcome.
7. Openly licensed or openly accessible research data.
8. Participant count and trial count recorded or recoverable from documentation/data structure.

## Screening rule

Screening reads codebooks, variable documentation, READMEs, and project documentation only.

No scoring contact with any candidate data is allowed before a mapping is declared.

No conversion is allowed before a mapping is declared.

No proxy reaction-time column may be substituted.

No reaction-time value may be generated, inferred without documentation, fabricated, or replaced with trial number, condition code, offer value, EEG window, or decision code.

## Log format

Each candidate entry records:

- Source URL
- License/access status
- Documentation screened
- Columns or variables found
- Verdict: ELIGIBLE or INELIGIBLE
- Reason

---

## Candidate 1 — OSF Task2 archive

Source URL: https://osf.io/8cj69

License/access status: open OSF archive.

Documentation screened:

- `data/codebook.pdf`
- `data/Task2_Export.csv`
- archive CSV structure already inspected in prior mapping PR

Columns or variables found:

- Five numeric columns in `Task2_Export.csv`, no header row.
- Offer and decision-like fields are present in the task export structure.
- No usable responder reaction-time column was identified.
- Prior inspection determined that the project files do not provide the required reaction-time field for the frozen pilot.

Verdict: INELIGIBLE.

Reason:

The frozen preregistered pilot requires `offer, stake, rt_ms, accept`. This archive does not provide a usable responder reaction-time column. The dataset must not be bent with a proxy timing field.

---

## Candidate 2 — Kubota et al. 2013, The price of racial bias

Source URL: https://osf.io/ugdcz

License/access status: open OSF archive.

Documentation screened:

- `Variables.docx`

Files observed from documentation-only screening:

- `Variables.docx`
- `IATRawData.zip`
- `Ultimatum_manuscript_2.sav`
- `UltimatumRawData.zip`

Raw data was not opened in this screening PR.

Columns or variables found in documentation:

- `pTOffer: [time, actualOffer]`
- `actualOffer: offer amount`
- `utDecision: [time, actualDecision]`
- `time = decisionTime - startTime`
- `actualDecision: 1 = Accept and 2 = Reject`
- `NaN for no response`
- `TOutcome: [TOutcome, trialOutcome]`
- `TOutcome: proposer payout outcome in dollars`
- `trialOutcome: responder payout outcome in dollars`
- `origOffer: trial offer drawn from the distribution of offers`
- `futureOffer: participant future offers`
- `five splits of $10`

Screening term check:

- Offer: found.
- Stake/payout: found.
- Decision: found.
- Reaction/decision timing: found through documented offer and decision time fields.
- Responder/participant fields: found.

Verdict: ELIGIBLE.

Reason:

Documentation identifies offer amount, decision outcome, payout/stake context, and timing fields for offer and decision. The next PR may declare a dataset-specific mapping before any data conversion. The likely reaction-time mapping must be declared before conversion as the documented decision time minus documented offer time, with any unit conversion to milliseconds stated explicitly.

---

## Candidate 3 — Cognitive reflection and unfair Ultimatum Game offers

Source URL: https://osf.io/4njwp

License/access status: not openly screenable through OSF API during this pass.

Documentation screened:

- None readable.

Columns or variables found:

- None.

Verdict: INELIGIBLE.

Reason:

OSF returned HTTP 401 Unauthorized during documentation screening. No readable documentation could be inspected, so eligibility cannot be established under the screening rule.

---

## Candidate 4 — Ultimatum giving / proposer behavior

Source URL: https://osf.io/cy9e6

License/access status: open OSF archive with readable supplemental documentation.

Documentation screened:

- Supplemental documentation PDF files.

Columns or variables found:

- Offer-related terms found.
- Decision and response-time related terms found in documentation.
- The project is documented as proposer behavior / ultimatum giving rather than responder-side accept/reject behavior.

Verdict: INELIGIBLE.

Reason:

The frozen pilot requires responder-side accept/reject data. This candidate appears to concern proposer-side giving behavior, so it does not match the responder-side preregistered pilot target.

---

## Candidate 5 — Values and conflict psychology

Source URL: https://osf.io/3cb8a

License/access status: OSF node accessible, but no files were found during screening.

Documentation screened:

- None.

Columns or variables found:

- None.

Verdict: INELIGIBLE.

Reason:

No accessible files or documentation were found, so eligibility cannot be established.

---

## Candidate 6 — Ultimatum game with 2 rounds

Source URL: https://osf.io/8vcfr

License/access status: inaccessible OSF storage during screening.

Documentation screened:

- None.

Columns or variables found:

- None.

Verdict: INELIGIBLE.

Reason:

OSF storage returned HTTP 404 Not Found during documentation screening. No readable documentation could be inspected, so eligibility cannot be established.

---

## Current next step after this PR

If this PR is merged and independently verified, the next PR should be a dataset-specific mapping declaration for:

https://osf.io/ugdcz

That next PR must declare the exact mapping before conversion, including:

- offer source
- stake or fixed-stake source
- reaction-time calculation and unit conversion to milliseconds
- accept/reject normalization
- converter invocation

Only after that mapping PR is verified may conversion occur.

Only after converted data is committed and verified may `preregistration_test.py` be run exactly once.
