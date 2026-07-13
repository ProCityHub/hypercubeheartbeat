# Unified Directive Reconciliation Record

**Authority:** Adrien D. Thomas
**Status:** Requires human decisions before PR1 merge

## Decision 1 — Compatibility shim

Original directive:

- Delete `tools/cognitive_cycle_runner.py`.

Current implementation:

- Logic was moved to `tools/audit_planbook.py`.
- A temporary compatibility shim currently remains at the old path.

Required Adrien decision:

- **A:** Delete the old path exactly as originally directed.
- **B:** Keep the shim for one release, then delete it in a later PR.

No decision is implied by this document.

## Decision 2 — Issue #8

Original directive:

- Close stale issue #8.

Current live record:

- Issue #8 is titled `data/README.md`.
- It documents Probe Data Intake and the required external behavioral-data CSV format.

Required Adrien decision:

- **A:** Close issue #8 because the original directive identifies it as stale.
- **B:** Keep issue #8 open because it is an active scientific intake record.
- **C:** Close issue #8 only after replacing it with a correctly scoped intake issue.

No automated closure is authorized by this document.

## Decision 3 — 008P manifest identity

Original directive names:

`ai_infrastructure/manifests/008P_BOUNDARY_INTEGRITY_MANIFEST.md`

That exact path has not been confirmed in the current tree.

Required action:

1. Locate all committed 008P Boundary Integrity records.
2. Identify the authoritative manifest.
3. Mark only that verified record `SUPERSEDED`.
4. Preserve its historical contents and audit chain.

## Decision 4 — Seven-PR ordering

Original directive order remains:

1. Reconciliation
2. GARVIS packaging
3. Bridge
4. Approval gates
5. CI
6. LL-10 preparation
7. Lattice Thinker

Any proposed reordering is an amendment and requires Adrien's explicit approval.

## Decision 5 — LL-10 approval

Original directive states that the LL-10 approval chain is approved by the directive.

Implementation must still preserve:

- exact deterministic parameters;
- immutable input and result hashes;
- preregistration;
- declared controls;
- failure conditions;
- permanent negative results.

The directive authorizes preparation. The exact execution command and resulting record must remain visible and auditable.

## Recorded PR1 decisions — 2026-07-13

- Decision 1: Original Option A selected.
  `tools/cognitive_cycle_runner.py` is removed.
- The active audit-only implementation is `tools/audit_planbook.py`.
- Decision 3: The authoritative 008P manifest was identified as:
  `ai_infrastructure/experiments/C_STAR_BOUNDARY_INTEGRITY_FIRST_MEASUREMENT_MANIFEST_008P.json`
- Its status is changed from `draft` to `superseded`.
- No experiment, measurement, approval, result, or claim upgrade occurred.
- Issue #8 remains an unresolved human decision because the live issue is a
  Probe Data Intake record rather than an obviously stale 008 implementation issue.

## Issue #8 decision — approved amendment

Adrien D. Thomas approves keeping GitHub issue #8 open.

Reason:

- The live issue is the Probe Data Intake record.
- It documents required external behavioral-data intake.
- It is not the stale 008 implementation issue contemplated by the original directive.

This is an explicit amendment to the original PR1 instruction to close issue #8.
