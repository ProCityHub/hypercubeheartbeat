# Stage 1 Senses Runbook

## Status

This is Stage 1.

It is senses only.

It creates a local append-only SQLite decision ledger.

It does not create autonomous action.

It does not call an LLM.

It does not change claims.

It does not score LL-06.

It does not run LL-07.

It does not touch Kubota holdout data.

## Purpose

Stage 1 gives the GARVIS body local senses on Android/Termux while preserving human approval.

The loop may record:

- local sensor command availability
- optional camera capture result
- optional microphone recording result
- optional local notification result
- append-only decision ledger rows

## Files

Main script:

`tools/stage1_senses_loop.py`

Default local database:

`data/stage1_senses/decision_ledger.sqlite3`

Default local output folder:

`data/stage1_senses/`

The `data/stage1_senses/` folder is runtime output and should not be committed unless a future directive explicitly says so.

## Install requirements

Termux packages:

```bash
pkg install -y python python-pip python-numpy git openssh termux-api termux-services
python -m pip install --upgrade requests
termux-setup-storage
Grant Android permissions when prompted for:

- camera
- microphone
- notifications

Permission screens may vary by Android version and vendor ROM.

## Initialize database

```bash
python tools/stage1_senses_loop.py --init-db
python tools/stage1_senses_loop.py --self-test

SELF_TEST_OK
python tools/stage1_senses_loop.py --once
python tools/stage1_senses_loop.py --once --notify

python tools/stage1_senses_loop.py --once --camera
python tools/stage1_senses_loop.py --once --microphone-seconds 10
python tools/stage1_senses_loop.py --once --camera --microphone-seconds 10 --notify
