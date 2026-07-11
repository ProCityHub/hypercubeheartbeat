# Phase 4 Step 2b-1B — Corrected Kubota MAT Mapping Declaration

Source URL: https://osf.io/ugdcz

Dataset: Kubota et al. 2013, "The price of racial bias"

This file declares the corrected raw `.mat` extraction before any conversion.

No conversion is performed in this PR.

No `data/ug_probe.csv` file is created in this PR.

No scoring is performed in this PR.

No `preregistration_test.py` execution is performed in this PR.

## Reason for corrected declaration

The earlier Kubota mapping expected standalone fields:

- `actualOffer`
- `actualDecision`

Raw structure inspection showed that the per-participant `.mat` files do not expose those as standalone fields.

Instead, the inspected `behavior` structure stores the relevant values inside paired arrays:

- `pTOffer`
- `utDecision`
- `TOutcome`
- `trialOutcome`
- `origOffer`

The correction below makes the column extraction explicit before conversion.

## Raw source for next conversion PR

The next conversion PR may use:

`UltimatumRawData.zip`

from:

https://osf.io/ugdcz

The raw archive contains per-participant MATLAB `.mat` files.

## Corrected declared mapping

The preregistered converter target remains:

```text
offer,stake,rt_ms,accept
The corrected raw extraction is:

| Target field | Raw extraction | Meaning |
|---|---|---|
| `offer` | `pTOffer[:, 1]` | actual offer value shown on the trial |
| `stake` | `10` | fixed documented stake: five splits of $10 |
| `rt_ms` | `utDecision[:, 0] - pTOffer[:, 0]` | responder decision time minus offer presentation time |
| `accept` | `utDecision[:, 1]` | raw decision code |

## Decision normalization

For `accept`:

| Raw value | Converted value |
|---|---|
| `1` | `1` |
| `2` | `0` |
| NaN / no response | exclude |

No other decision values may be invented or inferred.

## Timing-unit rule

The next conversion PR may inspect timing magnitudes only to determine whether the raw timing difference is in seconds or milliseconds.

Allowed unit decision:

- If the median raw timing difference is below `200`, treat raw timing as seconds and multiply by `1000`.
- Otherwise, treat raw timing as already milliseconds.

Forbidden during unit decision:

- no offer-vs-RT analysis
- no accept-vs-RT analysis
- no offer-vs-accept analysis
- no model scoring
- no preregistered test execution

## Exclusions allowed during conversion

Only the following exclusions are allowed:

1. NaN / no-response trials.
2. Rows where the declared raw fields cannot be parsed.
3. Rows removed by the converter-compatible `rt_ms < 200` floor.

Every excluded row must be counted by reason in the next conversion notes.

## Required converter output

The next conversion PR must create:

`data/ug_probe.csv`

with exactly:

```text
offer,stake,rt_ms,accept
```

The next conversion PR must also create conversion notes recording:

- raw filename
- raw file sha256
- converter command
- timing-unit decision
- timing-unit evidence based on magnitude only
- total row count
- excluded row count by reason
- final row count
- output `data/ug_probe.csv` sha256

## Claim discipline

This mapping declaration is not an empirical result.

This mapping declaration does not score the Lattice Law.

This mapping declaration does not prove consciousness.

This mapping declaration does not prove AGI.

The only purpose of this file is to freeze the corrected raw `.mat` extraction before conversion.
