# Triadic Deep Question Record Runbook

## Status

Implementation runbook for DIRECTIVE-008J.

This is a Stage 2 draft-only bridge-record instrument.

## Tool

tools/triadic_deep_question_record.py

## Purpose

This tool drafts one Dream Chamber -> Hypothesis Forge -> Lab Record bridge record for a deep question.

It gives GARVIS a notebook page for unresolved questions.

The page is not evidence.
The page is not a lab result.
The page is not a public claim.

## Default question

What is thinking, operationally, inside GARVIS?

## Example

python tools/triadic_deep_question_record.py --output tmp/deep_questions/triadic_deep_question_record.json --markdown --stdout

## From a cognitive cycle

python tools/triadic_deep_question_record.py --cycle tmp/cognitive_cycles/latest_cognitive_cycle.json --output tmp/deep_questions/triadic_deep_question_record.json --markdown

## What this tool may do

- read one local cognitive cycle JSON when provided
- draft one local bridge JSON
- optionally draft one Markdown view
- print a Markdown view to stdout
- validate required no-claim boundaries

## What this tool may not do

- call a network
- call an LLM
- execute an experiment
- append memory
- write a lab record
- write a result
- upgrade claims
- contact the outside world
- access secrets
- run in the background
- perform autonomous action

## Boundary

A bridge record is translation only.

It may request a future experiment manifest.

It cannot emit a scientific claim.

It cannot become evidence until a separate manifest and result exist.

## Standing sentence

008I taught GARVIS how to think the deep question; 008J gives that thought a lawful page without turning it into truth.
