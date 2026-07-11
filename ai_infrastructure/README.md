# AI Infrastructure Skeleton

This folder defines a safe workspace for future AI-assisted file, prompt, manifest, report, and decision workflows.

The system separates two organs:

1. Dreaming organ

The dreaming organ generates artifacts, metaphors, hypotheses, prompts, symbolic structures, and possible future tests.

2. Selecting organ

The selecting organ protects the record. It uses guards, declarations, hashes, conversion notes, one-run rules, and outcome logs to decide what survives contact with reality.

The lab exists so the dream never has to shrink; it only has to become testable.

## Safety model

AI agents may:

- read files placed in `inbox/`
- create manifests
- write reports
- propose prompts
- draft decision documents
- suggest next steps

AI agents may not:

- run preregistered tests without an explicit human directive
- edit frozen theory files
- edit preregistration files
- alter outcome logs
- replace a recorded negative result
- change data after seeing a result
- change formulas after seeing a result
- auto-merge empirical pull requests
- upgrade claims without a new declaration and later evidence

## Folder map

- `inbox/` receives user-provided files.
- `manifests/` records hashes, source URLs, provenance, and inventories.
- `prompts/` stores approved prompts and agent directives.
- `reports/` stores read-only inspection and analysis outputs.
- `decisions/` stores human-approved next-step declarations.

## Claim discipline

This infrastructure does not score claims.

This infrastructure does not run `preregistration_test.py`.

This infrastructure does not change data, formulas, criteria, mappings, or outcomes.

It only creates a safer nervous system for future work.
