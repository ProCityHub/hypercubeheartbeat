# Hypercube Self-Design Proposal Report

## Status

Local Stage 2 draft-only report.

No network call.

No LLM call.

No repository action.

No autonomous action.

No raw runtime sensor data read.

## Repository Context

- Repository: `/data/data/com.termux/files/home/hypercubeheartbeat`
- Committed safe files inspected: 76
- Inspection mode: committed files only via `git ls-files`

## Evidence Families

### ai_infrastructure

- `ai_infrastructure/README.md`
- `ai_infrastructure/decisions/APP_INFRASTRUCTURE_SKELETON_DECISION.md`
- `ai_infrastructure/decisions/APP_LEDGER_VIEWER_CLI_DECISION.md`
- `ai_infrastructure/decisions/APP_LEDGER_VIEWER_CONTRACT_DECISION.md`
- `ai_infrastructure/decisions/GARVIS_BODY_ARCHITECTURE_DECISION.md`
- `ai_infrastructure/decisions/GARVIS_CONSTITUTION_SKELETON_DECISION.md`

### app_infrastructure

- `ai_infrastructure/decisions/APP_INFRASTRUCTURE_SKELETON_DECISION.md`
- `app_infrastructure/APP_INFRASTRUCTURE_CHARTER.md`
- `app_infrastructure/GARVIS_COCKPIT_VISION.md`
- `app_infrastructure/README.md`
- `app_infrastructure/interfaces/APP_LEDGER_VIEWER_CLI_RUNBOOK.md`
- `app_infrastructure/interfaces/APP_LEDGER_VIEWER_CONTRACT.md`

## Detected Infrastructure Organs

- GARVIS constitution / constitutional law
- Living constitution / growth doctrine
- Staged action ladder
- Stage 1 local senses ledger
- App infrastructure shell
- App ledger viewer CLI
- Hypercube self-design contract
- GARVIS message schema / communication protocol
- Guard checks and tests

## Proposal Cap

This report is capped at 3 proposals.

## Proposals

## Proposal 1: Scientific Cockpit Snapshot CLI

### Stage Classification

Stage 0 view-only now; Stage 3 only if future versions run approved commands.

### Purpose

Create a single local command that summarizes branch state, guard status, recent committed directives, viewer availability, and recent safe ledger metadata.

### What This Gives Adrien

A fast cockpit readout: what body exists, what memory exists, what branch is active, what checks are clean, and what is unsafe to commit.

### What This Gives GARVIS

No new hands. It only gives GARVIS a clearer status surface to report back to Adrien.

### Required Files Or Systems Touched

`tools/scientific_cockpit_snapshot.py`, tests, runbook, decision record.

### Network Access

No.

### Secrets

No.

### Sensor Data

No raw sensor data. Safe ledger metadata only.

### Outside-World Effects

No.

### Approval Requirement

Human-approved PR required.

### Ledger Requirement

No execution ledger needed for view-only mode; future command-running version would require approval records.

### Test Requirement

Tests should prove no network calls, no raw payload display, safe failure, and clear untracked-file warnings.

### Failure Modes

Could become a fake dashboard if it hides uncertainty; could be mistaken for approval; could accidentally expand into a command runner if not bounded.

### Case Against This Proposal

This may be premature because the current CLI viewer already gives visibility into memory. A cockpit snapshot could duplicate existing commands and create maintenance overhead. It gives GARVIS a broader self-reporting surface, which could steer Adrien if the report becomes too polished. It should not be built if the next priority is proposal comparison or experiment reproducibility.

### Alternatives Considered

Use raw git commands plus app_ledger_viewer.py; build a full graphical app first; or build the proposal runner deeper.

### Why Alternatives Were Rejected

Raw commands are fragmented. A full app is too early. A deeper proposal runner before a cockpit snapshot may lack operational visibility.

### Constitutional Basis

Consistent with Stage 0 view-only, App Infrastructure shell, App Ledger Viewer Contract, and Cockpit Vision.

### Ledger Or Repository Evidence Relied On

- `ai_infrastructure/decisions/APP_INFRASTRUCTURE_SKELETON_DECISION.md`
- `ai_infrastructure/decisions/APP_LEDGER_VIEWER_CLI_DECISION.md`
- `ai_infrastructure/decisions/APP_LEDGER_VIEWER_CONTRACT_DECISION.md`
- `ai_infrastructure/decisions/GARVIS_CONSTITUTION_SKELETON_DECISION.md`
- `ai_infrastructure/decisions/GARVIS_LIVING_CONSTITUTION_DOCTRINE_DECISION.md`
- `ai_infrastructure/decisions/GARVIS_MESSAGE_SCHEMA_DECISION.md`
- `ai_infrastructure/decisions/GARVIS_STAGED_ACTION_LADDER_AMENDMENT_DECISION.md`
- `ai_infrastructure/decisions/HYPERCUBE_BRAIN_SELF_DESIGN_PROPOSAL_CONTRACT_DECISION.md`
- `ai_infrastructure/decisions/SELF_DESIGN_PROPOSAL_RUNNER_DECISION.md`
- `ai_infrastructure/decisions/STAGE1_SENSES_DECISION.md`

### Advisor Inputs

None. Generated locally from committed repository context only.

### Next Smallest Safe Step

Build a read-only snapshot CLI that displays status and never executes tests by default.

## Proposal 2: Proposal Comparison Matrix

### Stage Classification

Stage 2 draft-only preparation.

### Purpose

Turn self-design proposals into a comparison table showing value, risk, stage, files touched, case against, evidence basis, and next smallest safe step.

### What This Gives Adrien

A better decision instrument. Adrien can compare proposals instead of being persuaded by the first strong narrative.

### What This Gives GARVIS

A stricter proposal grammar. It makes GARVIS argue against its own proposals and expose tradeoffs.

### Required Files Or Systems Touched

`tools/proposal_comparison_matrix.py`, tests, runbook, decision record.

### Network Access

No.

### Secrets

No.

### Sensor Data

No.

### Outside-World Effects

No.

### Approval Requirement

Human-approved PR required.

### Ledger Requirement

No execution ledger needed because it is draft-only.

### Test Requirement

Tests should prove max proposal count, required Case Against section, advisory labeling, and no network imports.

### Failure Modes

Could become bureaucracy if too heavy; could make weak proposals look formal; could hide qualitative judgment behind a table.

### Case Against This Proposal

This may slow real building by turning every idea into a form. It could make the system feel safer than it is. A matrix is not truth; it is only a decision aid. It gives GARVIS more framing power unless Adrien remains the final judge.

### Alternatives Considered

Continue manual review; build the cockpit snapshot first; build an experiment dashboard first.

### Why Alternatives Were Rejected

Manual review does not scale. A cockpit snapshot shows state but not proposal tradeoffs. An experiment dashboard is powerful but needs clearer proposal selection first.

### Constitutional Basis

Directly implements the Case Against requirement, proposal volume boundary, traceability requirement, and advisory ring law.

### Ledger Or Repository Evidence Relied On

- `ai_infrastructure/decisions/APP_INFRASTRUCTURE_SKELETON_DECISION.md`
- `ai_infrastructure/decisions/APP_LEDGER_VIEWER_CLI_DECISION.md`
- `ai_infrastructure/decisions/APP_LEDGER_VIEWER_CONTRACT_DECISION.md`
- `ai_infrastructure/decisions/GARVIS_CONSTITUTION_SKELETON_DECISION.md`
- `ai_infrastructure/decisions/GARVIS_LIVING_CONSTITUTION_DOCTRINE_DECISION.md`
- `ai_infrastructure/decisions/GARVIS_MESSAGE_SCHEMA_DECISION.md`
- `ai_infrastructure/decisions/GARVIS_STAGED_ACTION_LADDER_AMENDMENT_DECISION.md`
- `ai_infrastructure/decisions/HYPERCUBE_BRAIN_SELF_DESIGN_PROPOSAL_CONTRACT_DECISION.md`
- `ai_infrastructure/decisions/SELF_DESIGN_PROPOSAL_RUNNER_DECISION.md`
- `ai_infrastructure/schemas/garvis_message_schema_v1.json`

### Advisor Inputs

None. Generated locally from committed repository context only.

### Next Smallest Safe Step

Add a local draft-only matrix generator for proposal reports.

## Proposal 3: Experiment Reproducibility Harness

### Stage Classification

Stage 3 local approved execution for running commands; Stage 0 for viewing manifests.

### Purpose

Create manifest-based experiment runs so tests, diagnostics, LL-07 demos, and future studies can be executed with clear inputs, outputs, boundaries, and audit records.

### What This Gives Adrien

More scientific power: reproducible runs, visible parameters, repeatable reports, and clearer separation between demo, sandbox, and evidence.

### What This Gives GARVIS

No autonomous science. It gives GARVIS a stricter way to prepare and document approved experiments.

### Required Files Or Systems Touched

`tools/experiment_repro_harness.py`, manifest schema, tests, runbook, decision record.

### Network Access

No by default.

### Secrets

No.

### Sensor Data

No raw sensor data by default.

### Outside-World Effects

No.

### Approval Requirement

Explicit approval required for each run command or run manifest.

### Ledger Requirement

Future approved runs should reference an approval ID if they execute beyond view-only inspection.

### Test Requirement

Tests should prove dry-run mode, no unapproved execution, output separation, and no claim upgrade.

### Failure Modes

This is powerful and can drift into action automation. It could create impressive reports that look like evidence when they are only demos. It must preserve the boundary between sandbox outputs and empirical results.

### Case Against This Proposal

This may be too powerful as the immediate next step because it introduces command execution structure. If built too early, it could blur Stage 2 proposals with Stage 3 approved execution. It should wait until the cockpit snapshot and proposal comparison tools are mature.

### Alternatives Considered

Keep running tests manually; build only dry-run manifests; build dashboard viewing before execution.

### Why Alternatives Were Rejected

Manual runs are hard to audit. Dry-run-only is safer but less useful. Dashboard-only does not improve reproducibility.

### Constitutional Basis

Aligned with Staged Action Ladder, approval ledger law, LL-07 demo boundaries, and scientific-power direction.

### Ledger Or Repository Evidence Relied On

- `.github/scripts/guard_check.py`
- `ai_infrastructure/decisions/APP_LEDGER_VIEWER_CLI_DECISION.md`
- `ai_infrastructure/decisions/APP_LEDGER_VIEWER_CONTRACT_DECISION.md`
- `ai_infrastructure/decisions/GARVIS_CONSTITUTION_SKELETON_DECISION.md`
- `ai_infrastructure/decisions/GARVIS_LIVING_CONSTITUTION_DOCTRINE_DECISION.md`
- `ai_infrastructure/decisions/GARVIS_STAGED_ACTION_LADDER_AMENDMENT_DECISION.md`
- `ai_infrastructure/decisions/HYPERCUBE_BRAIN_SELF_DESIGN_PROPOSAL_CONTRACT_DECISION.md`
- `ai_infrastructure/decisions/LL07_DATA_COLLECTION_DECISION.md`
- `ai_infrastructure/decisions/LL07_DEMO_PIPELINE_RUNBOOK_DECISION.md`
- `ai_infrastructure/decisions/LL07_SOLO_SANDBOX_DECISION.md`

### Advisor Inputs

None. Generated locally from committed repository context only.

### Next Smallest Safe Step

Do not build full execution yet; first draft a dry-run manifest viewer.

## Final Boundary

This report is advice, not authority.

Adrien decides.

GitHub records only after approved repository procedure.
