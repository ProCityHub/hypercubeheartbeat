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
| _pending_ | | |
