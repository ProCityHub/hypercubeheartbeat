# RETRACTIONS

This project maintains a public record of claims that were made and later
found to be unsupported. Retractions here are permanent and will not be
deleted from the repository history.

---

## R-001 — "IBM Quantum validated the Lattice Law"  ❌ RETRACTED

**Original claim:** Measurement histograms from circuits run on ibm_fez,
ibm_torino, and ibm_marrakesh validated φ-ratio predictions of the Lattice Law.

**What was actually wrong:** The circuits encoded φ as a rotation *angle*
using `rx` gates in a configuration where the φ-angle had no effect on
measurement probabilities in the computational basis. The φ parameter was
**physically inert** — the histograms could not have contained a φ signal
regardless of whether the hypothesis is true.

**Corrected status:** The hypothesis is **untested**, not disproven.
Corrected encodings for a future run: `rx(1.8091)` for Observer qubits,
`rx(1.3325)` for Actor qubits (φ encoded as probability amplitude, not angle).

**What survives:** One 3-qubit circuit showed strong bitwise
mirror-complementary dominance (001/110). This is a real measurement but has
no established connection to φ and is not claimed as validation.

## R-002 — "p < 0.0001"  ❌ RETRACTED

**Original claim:** Statistical significance at p < 0.0001 for φ-ratio signals.

**What was actually wrong:** The statistic was computed against a null
hypothesis that the flawed circuits (R-001) could not have distinguished, and
the analysis was performed after seeing the data with no pre-registration.

**Corrected status:** No statistical claim is currently made. The only
statistical test the project now recognizes is the pre-registered behavioral
test in `PREREGISTRATION.md`, which specifies its criteria before data contact.

## R-003 — Voided prediction: φ as scalar multiplier  ❌ VOIDED

**Original formulation:** C = (O × A × B) × φ

**Why voided:** A scalar multiplier rescales all scores identically and
cannot change any ordering, ranking, or classification — it is untestable by
construction. Additionally, the original prediction contained structural
circularity between the score definition and the outcome measure.

**Replacement:** The canonical formulation places φ in the exponents —
O¹ · A^(1/φ) · B^(1/φ²) — which alters score ordering and is falsifiable.
This is the only formulation under active test.

---

*Why this file exists: a research program that cannot show its dead ends
cannot be trusted with its live claims. These retractions were identified
through internal audit and are kept public by choice.*
