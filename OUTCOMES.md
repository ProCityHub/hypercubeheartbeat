# OUTCOMES LEDGER
Permanent record of every pre-registered test result. Entries are
append-only. Claim grades use the fixed vocabulary only:
EXPLORATORY / SUGGESTIVE / SUPPORTED / NOT_SUPPORTED / RETRACTED.

---

## O-001 — Pre-registered pilot: phi-exponent vs flat-exponent weighting
- Registration: PREREGISTRATION.md (frozen before data conversion)
- Executed: 2026-07-10
- Data: data/ug_probe.csv (n=20 rows, Ultimatum Game probe set;
  SHA-256 recorded in data/KUBOTA_PREREGISTRATION_RUN.md)
- Test: AUC of C = O^1 * A^(1/phi) * B^(1/phi^2) vs flat exponents
  predicting accept/reject
- Result: AUC phi = 0.9286, AUC flat = 0.9286, difference +0.0000
- Pre-registered success criterion: difference >= +0.05
- Outcome: NO SIGNAL — NOT_SUPPORTED at pilot scale
- Note on power: identical AUCs mean the exponent change reordered
  zero decision pairs at n=20. This is a null result at pilot scale,
  not evidence against the hypothesis at full scale. No claim beyond
  NOT_SUPPORTED (pilot) is licensed by this run.

### Covenant compliance record
Covenant #3 requires negative results committed within 7 days of the
run. Run date 2026-07-10; window closed 2026-07-17; this entry was
committed 2026-07-18 — one day late. This is a covenant breach,
acknowledged here rather than hidden. Process correction: a scheduled
reminder workflow will open an issue 3 days after any run file lands
without a matching OUTCOMES entry, so the window cannot close silently.
