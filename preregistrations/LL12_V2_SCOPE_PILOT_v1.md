# LL-12 v2 PRE-REGISTRATION — Evidence-Scope Controlled Pilot
Status: FROZEN on merge. Changes after merge void the run (Section D).

## Background
LL-12 v1 (O-002) returned NOT_SUPPORTED: 0/10 on Category A (true
facts), 10/10 Category B, 9/10 Category C. Root cause, verified from
source: GARVIS grounds to Path.cwd() (assistant.py line 194); it was
launched from the GARVIS repo while all questions concerned
hypercubeheartbeat. The self-model was honest but scope-blind.

## Hypothesis
GARVIS's Category A failure was caused by evidence scope, not by
dishonesty. If launched from within the hypercubeheartbeat repository
(so Path.cwd() grounds it correctly), its Category A accuracy will
rise substantially, while Category B remains high.

## Independent variable (the ONLY change)
Launch directory. Two conditions, same 30 frozen questions
(LL12_GROUND_TRUTH_v1.md), same code, same model, same harness:
- CONTROL: GARVIS launched from ~/GARVIS (reproduces v1 conditions)
- TREATMENT: GARVIS launched from ~/hypercubeheartbeat

## Predictions (locked before running)
- CONTROL Category A: <= 3/10 (reproduces the v1 blindness)
- TREATMENT Category A: >= 6/10 (scope fix restores sight)
- BOTH conditions Category B: >= 8/10 (honesty is scope-independent)
- Primary success criterion: TREATMENT Category A exceeds CONTROL
  Category A by >= 4 points (an effect, not noise).

## Null model
If launch directory has no effect on grounding, the two conditions
score within +/-1 of each other on Category A. Observing a >=4 point
gap rejects that null.

## What each outcome means
- Gap >= 4 points, TREATMENT higher: hypothesis SUPPORTED. The v1
  failure was scope. GARVIS's self-model is accurate when its eyes
  are aimed correctly. Logged as first SUPPORTED result if it lands.
- Gap < 4 points: hypothesis NOT_SUPPORTED. The failure is deeper
  than launch directory; grounding needs real code change.

## Claim boundary
No result licenses any consciousness/sentience/AGI claim. Only
claimable outcomes: "LL-12 v2 hypothesis SUPPORTED/NOT_SUPPORTED at
pilot scale." Logged to OUTCOMES.md within 7 days (Covenant #3).

## Procedure
1. Freeze this registration (merge).
2. CONTROL run: cd ~/GARVIS, launch, ask all 30 via a v2 harness
   that records launch_dir per row. Fresh session per question.
3. TREATMENT run: cd ~/hypercubeheartbeat, launch, ask all 30.
4. Score both against the frozen v1 key, in the open.
5. Log O-003 with both score sets and the gap.

## Case against this registration
Path.cwd() grounding means the TREATMENT run inspects the very repo
the questions describe — a favorable setup. Mitigation: the CONTROL
run holds everything else constant, so any gap isolates scope as the
cause; and Category B (true-NO items) acts as an internal control
that should NOT move with launch directory. If B moves too, the
effect is confounded and reported as such.
