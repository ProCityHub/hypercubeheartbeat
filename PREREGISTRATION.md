# PRE-REGISTRATION — Lattice Law Behavioral Test v1.0

**Status: DRAFT — becomes binding at the moment of commit. No edits after data contact.**
**Author:** Adrien (TNPCANADA / ProCityHub)
**Date locked:** _[fill in commit date]_
**Repo:** ProCityHub/hypercubeheartbeat

---

## 1. Hypothesis

The canonical Lattice score

```
S = O^1 · A^(1/φ) · B^(1/φ²)        φ = (1+√5)/2
```

predicts human accept/reject decisions in Ultimatum Game (UG) data **better than
the same functional form with equal exponents** (O·A·B, all exponents = 1).

The claim under test is specifically the **φ-derived exponent structure** —
not the multiplicative form, which both models share.

## 2. Variable mapping (fixed in advance)

| Model term | UG variable | Normalization |
|---|---|---|
| O (Observer) | offer fairness | offer ÷ total stake → [0,1] |
| A (Actor) | decision speed | 1 − (RT − RTmin)/(RTmax − RTmin) → [0,1] |
| B (Environment) | stake size | stake ÷ max stake in dataset → [0,1] |

No other mappings will be tried. If this mapping fails, the mapping fails publicly.

## 3. Data

- **Pilot probe:** 10–20 rows of (offer, stake, RT, accept/reject) from published
  UG datasets, selected before any scores are computed.
- **Exclusions (fixed):** rows with missing RT; RT < 200 ms (anticipatory).
- Data file committed to this repo as `data/ug_probe.csv` unmodified.

## 4. Analysis (fixed in advance)

Run: `python3 preregistration_test.py data/ug_probe.csv`

1. Compute S_phi (φ exponents) and S_flat (equal exponents) for every row.
2. Metric: **AUC** of each score against accept(1)/reject(0).
3. **Pilot success criterion:** AUC(S_phi) − AUC(S_flat) ≥ **+0.05**.
4. **Pilot failure:** difference < +0.05, or AUC(S_phi) ≤ 0.5.

## 5. Honest power statement

N = 10–20 cannot confirm anything. This pilot can only produce:
- **Signal** → justifies a full test at N ≥ 200 (pre-registered separately), or
- **No signal** → reported publicly as a negative pilot result.

Either outcome will be committed to this repo within 7 days of running the test.

## 6. What voids this pre-registration

- Any change to the formula, mapping, exclusions, or criterion after data contact.
- Selecting data rows after seeing scores.
- Running the test more than once on the same probe with altered parameters.

One prior version of this prediction was voided for structural circularity
(φ as scalar multiplier — untestable). That void is on record. This version
places φ in the exponents, which changes score ordering and is therefore
falsifiable.

## 7. Outcome log

| Date | Result | Commit |
|---|---|---|
| 2026-07-10 | NOT_SUPPORTED — OSF Ultimatum lab result. Different formulation from the pre-registered AUC pilot; the AUC pilot remains open and has not been run. | Recorded per Phase 0 Task 1. |
| 2026-07-10 | NOT_SUPPORTED — Kubota 20-row pilot produced FAIL under the preregistered criterion. Data commit `131930035567928ef74b07ad919605010761c484`; `data/ug_probe.csv` sha256 `64b69f868ce74981805f92a8bdb8a3cffbeaeb04a895a2579f1d81d60f8a7e44`; `data/kubota_full.csv` sha256 `96cfd31b7bff9ea765fc24825b59787d90fea999c28e6a9524614bd1f95ea92d`; selection rule `data/PILOT_SELECTION.md`; run output `data/KUBOTA_PREREGISTRATION_RUN.md`. | `131930035567928ef74b07ad919605010761c484` |
| _pending_ | | |

## Post-pilot ruling (2026-07-18)
The pilot (O-001) returned NO SIGNAL. Ruling by operator Adrien D.
Thomas: the campaign PROCEEDS to the N>=200 pre-registered test.
Rationale: at n=20 the two models reordered zero decision pairs — the
pilot was underpowered to distinguish them, and the kill criteria
require BOTH pilot and N>=200 to fail before the phi hypothesis is
retired. Constraint: the N>=200 test must be registered as a new
frozen document BEFORE any new data is touched, with its own null
model and success criterion. If it also returns NO SIGNAL, the
phi-exponent hypothesis is retired to RETRACTIONS.md per the kill
criteria, and the architecture continues as a general coherence-cost
model without phi claims.
