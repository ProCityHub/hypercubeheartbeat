# Lattice Validation Laboratory

**Author and theory originator:** Adrien D. Thomas  
**Repository target:** ProCityHub/hypercubeheartbeat  
**Status:** Exploratory, reproducible research scaffold

## Purpose

This laboratory converts the Lattice Law from a broad interpretive framework into a sequence of tests that can fail.

The laboratory does not assume that the Lattice Law is correct. It compares lattice-derived predictions against ordinary statistical baselines, ablations, shuffled controls, and alternative constants. Each result is assigned one of five verdicts:

- `SUPPORTED`
- `PARTIALLY_SUPPORTED`
- `NOT_SUPPORTED`
- `INCONCLUSIVE`
- `INVALID_TEST`

## Core discipline

Theory → operational definition → preregistered prediction → baseline → lattice model → hostile control → held-out test → verdict.

The framework must not grade itself. Variables cannot be defined from the outcome they are supposed to predict. Phi is not treated as evidence merely because it was inserted into a formula.

## First experiment

The first study uses the supplied OSF archive for the Ultimatum-game EEG experiment. It performs:

1. Data integrity and schema checks.
2. A basic reproduction of key descriptive quantities.
3. Trial alignment between offer-locked EEG (`M2_single.csv`) and feedback-locked EEG (`M3_single.csv`).
4. A forward-only Observer–Actor–Bridge feature construction.
5. Grouped cross-validation by participant.
6. Comparisons against ordinary baselines and destructive controls.
7. A formal identifiability check for the role of φ.

## Run

```bash
python -m pip install pandas numpy scipy scikit-learn
python experiments/osf_ultimatum/run_experiment.py
python -m unittest discover -s tests -v
```

## Honest framing

This laboratory tests measurable models inspired by the Lattice Law. It does not test subjective consciousness and cannot establish awareness, sentience, spirit, or metaphysical truth.
