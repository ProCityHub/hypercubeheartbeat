# Phase 4 Step 2b-2B — Single Preregistration Run

Date: 2026-07-10

Data commit before run: `131930035567928ef74b07ad919605010761c484`

Pilot file: `data/ug_probe.csv`

Pilot sha256: `64b69f868ce74981805f92a8bdb8a3cffbeaeb04a895a2579f1d81d60f8a7e44`

Full Kubota file: `data/kubota_full.csv`

Full Kubota sha256: `96cfd31b7bff9ea765fc24825b59787d90fea999c28e6a9524614bd1f95ea92d`

Selection rule file: `data/PILOT_SELECTION.md`

Command run exactly once:

    python3 preregistration_test.py data/ug_probe.csv

## Complete unedited output

    Loaded 20 rows (0 excluded per pre-registration)
    AUC  phi-exponents  (O^1 A^0.618 B^0.382): 0.9286
    AUC  flat-exponents (O^1 A^1     B^1    ): 0.9286
    Difference: +0.0000   (criterion: >= +0.05)
    RESULT: FAIL — phi exponents do not beat flat exponents by the pre-registered margin. Report this negative result publicly.
