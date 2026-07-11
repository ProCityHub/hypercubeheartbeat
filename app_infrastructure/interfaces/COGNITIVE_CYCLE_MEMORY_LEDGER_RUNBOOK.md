# Cognitive Cycle Memory Ledger Runbook

## Status

Implementation runbook for DIRECTIVE-008F.

This is a Stage 2 local memory-vessel initialization instrument.

It is not an append command.

It is not an autonomous runtime.

## Tool

`tools/cognitive_cycle_memory_ledger.py`

## Purpose

The Cognitive Cycle Memory Ledger CLI initializes a local SQLite memory vessel for future GARVIS cognitive-cycle records.

It creates the empty structure required for future thought continuity.

## Example

    python tools/cognitive_cycle_memory_ledger.py \
      --db data/cognitive_cycles/cognitive_cycle_memory.sqlite3 \
      --init-db \
      --verify

## Tables

The initialization creates:

- `ledger_metadata`
- `cognitive_cycle_records`
- `open_problem_placeholders`

## Cognitive cycle records

The `cognitive_cycle_records` table is prepared for future records containing:

- cycle identity
- cycle hash
- artifact links
- previous and next cycle links
- selected candidate
- decision
- confidence
- power request summary
- operator review
- claim and approval boundaries

## Open-problem placeholder

The `open_problem_placeholders` table is a seed for future no-claim/open-problem work.

It does not solve or claim anything.

It reserves space for unresolved research objects such as:

- consciousness
- thinking
- memory
- autonomy
- AGI
- proof
- evidence

The rule is:

Forbidden to claim.

Allowed to define, question, design tests, preserve uncertainty, and revisit.

## What this tool does

This tool may:

- create a local SQLite database
- create empty memory tables
- verify required tables exist
- print database status

## What this tool does not do

This tool does not:

- append cognitive cycles
- read cognitive cycle JSON
- write memory records
- execute candidate proposals
- call a network
- call an LLM
- contact the outside world
- commit
- push
- upgrade claims
- prove consciousness
- prove AGI

## Local runtime boundary

The default database path is:

`data/cognitive_cycles/cognitive_cycle_memory.sqlite3`

That database is local runtime state.

It should not be committed.

## Stage classification

This is Stage 2 memory-vessel initialization.

Future append behavior requires a separate directive.

## Standing boundary

A vessel is not memory yet.

A table is not proof.

A placeholder is not a claim.

Adrien decides what enters memory.
