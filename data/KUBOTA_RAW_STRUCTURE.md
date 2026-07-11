# Phase 4 Step 2b-1A — Kubota Raw Structure Inspection

Source URL: https://osf.io/ugdcz

This file records why the Phase 4 Step 2b-1 conversion did not proceed.

No conversion was performed.

No `data/ug_probe.csv` file was created.

No `preregistration_test.py` execution was performed.

No SUPPORTED or NOT_SUPPORTED verdict was produced.

## Raw file inspected

Raw archive:

`UltimatumRawData.zip`

The archive contains per-participant MATLAB `.mat` files.

A representative file was inspected for raw structure:

`012701.mat`

The file loaded successfully with SciPy and contained a `behavior` structure.

## Fields observed

Observed `behavior` fields included:

- `pTOffer`
- `utDecision`
- `TOutcome`
- `trialOutcome`
- `origOffer`

The frozen mapping expected standalone fields:

- `actualOffer`
- `actualDecision`

Those standalone fields were not present in the inspected `.mat` structure.

## Important structure finding

The inspected `.mat` structure appears to store the documented values as paired arrays:

- `pTOffer` has shape `(160, 2)`
- `utDecision` has shape `(160, 2)`
- `TOutcome` has shape `(160, 2)`

This suggests the documented values may be embedded as:

- `pTOffer[:, 0]` = offer presentation time
- `pTOffer[:, 1]` = actual offer
- `utDecision[:, 0]` = decision time
- `utDecision[:, 1]` = actual decision

However, the currently frozen `data/KUBOTA_MAPPING.md` does not explicitly declare second-column extraction.

## Decision

CONVERSION BLOCKED.

Reason:

The raw `.mat` structure does not expose standalone `actualOffer` and `actualDecision` fields. Applying the intended extraction would require a mapping clarification before conversion.

## Required next step

Create a new mapping declaration PR that explicitly declares the raw `.mat` extraction:

- `offer = pTOffer[:, 1]`
- `stake = 10`
- `rt_ms = utDecision[:, 0] - pTOffer[:, 0]`, converted to milliseconds if source timing is seconds
- `accept = utDecision[:, 1]`, normalized as `1 = accept`, `2 = reject`

Only after that corrected mapping PR is merged and verified may conversion proceed.

## Claim discipline

This inspection is not a result.

This inspection does not score the Lattice Law.

This inspection does not prove consciousness.

This inspection does not prove AGI.

This inspection preserves the rule that raw-data deviations must be declared before conversion.
