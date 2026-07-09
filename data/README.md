# Probe Data Intake

This folder receives external behavioral data for the pre-registered test defined in PREREGISTRATION.md.

Required CSV format: `offer,stake,rt_ms,accept`
- `offer`: amount offered to responder
- `stake`: total pot
- `rt_ms`: responder reaction time in milliseconds
- `accept`: `1` accepted / `0` rejected

Rule: data files must be committed BEFORE `preregistration_test.py` is run on them, and the commit message must include the source URL.
