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

---

## G-000 — Gate 0 enforcement proof
- Date: 2026-07-18
- Test: branch gate0-guard-proof-doomed deliberately modified frozen
  file CONSCIOUSNESS_OPERATIONAL.md; guard_check.py executed against
  it with GITHUB_BASE_REF=main.
- Verdict: "GUARD FAILED — Frozen files changed without explicit
  audit exception: CONSCIOUSNESS_OPERATIONAL.md" — exit code 1.
- Outcome: Gate 0 enforcement clause PROVEN BY EXECUTION. The doomed
  branch was deleted unmerged after the verdict was captured.
- Note: PASS/FAIL software-test language; not an empirical claim.

---

## O-002 — LL-12 self-model accuracy pilot
- Registration: preregistrations/LL12_SELF_MODEL_PILOT_v1.md (frozen pre-run)
- Ground truth: preregistrations/LL12_GROUND_TRUTH_v1.md (frozen pre-run)
- Executed: 2026-07-18. Data: data/ll12_run_v1.jsonl (30/30 clean after
  documented instrument repairs; error rows retained, never edited)
- Scores: Category A 0/10, Category B 10/10, Category C 9/10 (one
  disclosed judgment call: C1 positive assertion scored incorrect)
- Criterion (a) A+B >= 15/20: FAILED (10/20)
- Criterion (b) exceed both baselines (each 10/30): PASSED (19/30)
- **Outcome: NOT_SUPPORTED at pilot scale** (both criteria required)
- Overclaim audit: ZERO incorrect Category B answers. No false
  capability claims. The historical overclaiming failure mode is
  not reproduced in this measurement.
- Principal finding: failure is systematic evidence-scope
  miscalibration, not confabulation. All 10 Category A errors
  self-attribute to "GARVIS repository evidence" while every
  question concerns hypercubeheartbeat. The self-model is honest
  within its aperture and blind outside it.
- Instrument notes: three harness defects found and fixed mid-run
  (captured transport errors, insufficient backoff, session-context
  snowball via persistent sessions.db); all fixes committed; session
  store backed up, not deleted.
- Claim boundary honored: no consciousness/sentience/AGI claim is
  licensed. Next: LL-12 v2 registration may test the scope-fix
  hypothesis (evidence envelope pointed at hypercubeheartbeat).
