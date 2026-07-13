# Architecture Decisions — Post-Reconciliation

**Date:** 2026-07-13  
**Authority:** Adrien D. Thomas

## Repository responsibilities

- **Hypercube Heartbeat:** deterministic state, canonical scoring, scientific registry, guards,
  preregistrations, evidence packets, and research modules.
- **GARVIS:** direct conversational answers, memory/session orchestration, and explicitly approved
  tool execution.
- **AGI / Termux:** operator CLI, sensors, local state, and tmux interface.

## Conversation

- The operator-facing default is `brain answer "<question>"` or the GARVIS CLI.
- The exact question is preserved.
- Internal classifiers may add metadata but may not replace the question with a planbook entry.
- Hypercube does not silently call an LLM. It supplies deterministic evidence to GARVIS.

## Audit and planning

- `tools/audit_planbook.py` is optional and audit-only.
- The legacy runner path may remain temporarily as a deprecation shim.
- Audit output is not a conversational answer and is not evidence of consciousness.

## Scientific claims

- Every hypothesis is evidence-labeled and, when tested, preregistered.
- Result packets are immutable and hash-addressed.
- Negative and null results are retained permanently.
- Software PASS/FAIL is not an empirical SUPPORTED/NOT_SUPPORTED verdict.

## External actions

- No send, publish, delete, trade, payment, account edit, or other side effect occurs without:
  1. `GARVIS_EXTERNAL_ENABLED=true`;
  2. an exact, unexpired approval object;
  3. signature verification;
  4. content and destination equality;
  5. an audit record.
- Default runtime has no outside-world tools attached.

## Formula

The sole canonical formula is:

```text
C = O^1 * A^(1/phi) * B^(1/phi^2)
```

The scalar-multiplier form is retracted and must remain guard-blocked.

## Governance

- Minimal Law: **Think freely. Speak honestly. Prepare powerfully. Act with Adrien.**
- Branch → commit → tests → guard → PR → review → human merge.
- One PR is merged at a time.
- Machines do not merge.

## Directive authority

The Original Unified Build Directive is the governing implementation path.

The seven PRs may be developed concurrently, but they are merged in this order:

1. Repository Reconciliation
2. Package GARVIS Properly
3. Real GARVIS–Hypercube Bridge
4. Enforce Approval Gates
5. Strengthen CI
6. LL-10 Measurement Preparation
7. Lattice Thinker Research Integration

Any deviation must be documented as a proposed amendment and explicitly approved by Adrien D. Thomas.
