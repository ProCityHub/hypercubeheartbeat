# QASM Thesis Reviewer

**Author and concept origin:** Adrien D. Thomas / ProCityHub

## Purpose

`qasm_thesis_reviewer.py` gives Hypercube Heartbeat a deterministic scientific
inspection layer for OpenQASM and Lattice-theory documents.

It does not replace GARVIS conversation. It creates structured context that a
future GARVIS language layer can read without pretending the current
O/A/B heartbeat instrument already has a semantic opinion.

## What it inspects

- OpenQASM register sizes
- gate and measurement counts
- parameterized rotation values
- approximate source-level dependency depth
- coupling graph and topology classification
- concepts named in file names or comments
- gate-level structure versus intended semantic meaning
- required null models and controls
- explicit falsification conditions
- scalar-phi versus exponent-weighted formula conflict

## Scientific boundary

The reviewer may identify circuit proxies for coupling, phase, recurrence,
measurement, or topology. It does not treat comments such as Observer, Memory,
Witness, Consciousness, Heartbeat, or Attractor as physical evidence.

A positive angle-comparison result could support a circuit-specific parameter
effect. It would not prove consciousness, AGI, sentience, or new physics.

## Run inside Hypercube Heartbeat

```bash
python qasm_thesis_reviewer.py \
  --qasm-root ai_infrastructure/inbox/ll06_exploratory_2026_04 \
  --theory README.md \
  --theory CONSCIOUSNESS_OPERATIONAL.md \
  --theory RETRACTIONS.md \
  --theory claims/CLAIMS.json \
  --output-dir reports/qasm_thesis_review \
  --stdout
```

To include Adrien D. Thomas's additional circuit from a neighboring AGI clone:

```bash
python qasm_thesis_reviewer.py \
  "$HOME/AGI/lattice_cube_engine/quantum/adrien_double_slit.qasm" \
  --qasm-root ai_infrastructure/inbox/ll06_exploratory_2026_04 \
  --theory README.md \
  --theory CONSCIOUSNESS_OPERATIONAL.md \
  --theory RETRACTIONS.md \
  --theory claims/CLAIMS.json \
  --output-dir reports/qasm_thesis_review
```

## Outputs

- `review.json`: machine-readable GARVIS/Hypercube context
- `review.md`: human-readable scientific assessment

## Current limitation

This reviewer is deterministic and rule-based. It is not an LLM, does not run
QASM, does not contact quantum hardware, and does not upgrade claims. GARVIS
still needs a separate tested language-model adapter to answer open questions
semantically from this structured context.
