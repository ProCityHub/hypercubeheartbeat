# Phase 4 Step 2b-2A — Pilot Probe Selection

Source URL: https://osf.io/ugdcz

This file declares the pilot probe selection rule before any scoring.

No preregistration_test.py execution was performed in this PR.

No scoring was performed in this PR.

## Preserved full dataset

The full converted Kubota dataset is preserved as:

- `data/kubota_full.csv`
- sha256: `96cfd31b7bff9ea765fc24825b59787d90fea999c28e6a9524614bd1f95ea92d`

This full file is the candidate dataset for a later confirmatory test only if the pilot shows signal.

## Pilot selection rule

The pilot probe is the FIRST 20 DATA ROWS of `data/kubota_full.csv` in committed order.

This rule is deterministic, was declared before any score computation, and involves no inspection of row contents.

The selected pilot probe is written to:

- `data/ug_probe.csv`
- sha256: `64b69f868ce74981805f92a8bdb8a3cffbeaeb04a895a2579f1d81d60f8a7e44`

## Claim discipline

This pilot-selection declaration is not an empirical result.

This pilot-selection declaration does not score the Lattice Law.

This pilot-selection declaration does not prove consciousness.

This pilot-selection declaration does not prove AGI.

The only purpose of this file is to freeze the pilot probe before the single preregistered run.
