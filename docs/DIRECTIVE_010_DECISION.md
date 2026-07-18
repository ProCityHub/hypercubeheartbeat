# DIRECTIVE-010 — Session Resilience Organ
Decision record. Status: implemented locally, validated, merged.

## Motivation (the failure)
2026-07-12, ~00:00 local: a live GARVIS chat session in Termux was
killed by an unhandled OpenAI 429 rate-limit error whose own message
said "try again in 10.127s". The process exited code 1. The entire
conversation - held only in RAM - was lost on restart. Screenshot of
the crash is on record in the operator's audit trail.

## The three defects
1. RECOVERABLE ERROR TREATED AS FATAL: a 10-second wait became a
   process death because no retry handler existed.
2. MEMORY ONLY IN RAM: the conversation stream had no ledger,
   violating "nothing forgotten, nothing hidden" while the senses
   stream had one.
3. UNBOUNDED CONTEXT: full history was resent every turn (215,671
   tokens used, 251,843 requested against a 400,000 TPM cap),
   making the rate-limit crash mathematically inevitable.

## The fix (tools/garvis_resilience.py)
- call_with_retry(): exponential backoff, honors retry-after hints
  parsed from error text, retries 429/timeout/connection/5xx up to
  5 times, raises honestly after.
- SessionLedger: append-only JSONL, one file per day, every turn
  flushed with fsync (survives power loss), auto-resume on startup,
  deterministic keyword recall over full history.
- build_context(): hard cap (30 turns) on history sent to the
  model; older turns remain on disk, recallable, never resent.

## Validation
Self-test executed on the operator's phone in Termux, 2026-07-18:
"PASS: ledger persists, resumes, caps context, recalls history."
The self-test demonstrated live resurrection: it resumed 2 turns
from a ledger file left by an earlier interrupted run. Screenshot
on record.

## Constitutional position
Local-first. No new capabilities, no new network paths, no
autonomous action. Pure reliability organ. Wiring into the GARVIS
chat loop (5 integration points, in the module header) follows as
its own change in the GARVIS repository per the staged ladder.

## Case against this directive
Retries can mask a misconfigured client; a capped context changes
model behavior versus full history; fsync-per-turn costs write
cycles. Overruled: misconfiguration still surfaces after 5 honest
retries; unbounded context was a time bomb, not a behavior; flash
wear at conversation cadence is negligible. Recorded per relay law.
