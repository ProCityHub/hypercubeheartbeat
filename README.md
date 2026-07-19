# hypercubeheartbeat

**A governed AI research program, run from a phone, that publishes
its own failures.**

This repository is the permanent record of an independent research
project by Adrien D. Thomas (Treaty 6 Territory, Edmonton, Alberta):
an approval-gated AI assistant (GARVIS), governed by a written
constitution, measured by pre-registered experiments, with every
result — including negative ones — logged publicly.

## The record
- **[GARVIS_CONSTITUTION.md](docs/GARVIS_CONSTITUTION.md)** — the law:
  autonomy of viewpoint, never autonomy of action. No merge but the
  operator's hand.
- **[OUTCOMES.md](OUTCOMES.md)** — every pre-registered test result.
  Current entries include two honest negative results (O-001 NO
  SIGNAL; O-002 NOT_SUPPORTED) and one enforcement proof (G-000).
- **[RETRACTIONS.md](RETRACTIONS.md)** — withdrawn claims, buried by
  name: quantum validation (R-001), p<0.0001 (R-002), the scalar
  formula (R-003). A program that cannot show its dead ends cannot
  be trusted with its live claims.
- **[PREREGISTRATION.md](PREREGISTRATION.md)** — predictions frozen
  before data, always.
- **[CONSCIOUSNESS_OPERATIONAL.md](CONSCIOUSNESS_OPERATIONAL.md)** —
  operational definitions (LL-10..13), each with a metric, a control,
  and a written failure condition. This project claims no
  consciousness, sentience, or AGI; it defines measurable properties
  and tests them.

## Claim vocabulary
Every claim here is graded: **SUPPORTED / NOT_SUPPORTED /
EXPLORATORY / RETRACTED.** Nothing else is permitted.

## Covenants
Negative results are committed within 7 days, always. Frozen files
are guarded by CI (guard_check.py) — proven by execution, logged as
G-000. The operator is the sole signing authority.

## Origins & Vision
This work grew from a personal cosmology — the Lattice, the cube,
golden-ratio dynamics, and Cree Two-Eyed Seeing (wahkohtowin: all
things in relation). That original layer is preserved, unedited and
clearly fenced, in [archive/vision/](archive/vision/). The myth
supplied the fire; the method decides what is true. They are kept
separate on purpose, and both are owned.

## Runtime
The assistant lives in [ProCityHub/GARVIS](https://github.com/ProCityHub/GARVIS)
(a governed fork of the OpenAI Agents SDK) and runs in Termux on an
Android phone. Resilience organs: retry-with-backoff, fsync'd session
ledger, bounded context (DIRECTIVE-010). Memory organ candidate:
Hopfield attractor "ghost memory" (tools/ghost_memory.py, LL-11).
