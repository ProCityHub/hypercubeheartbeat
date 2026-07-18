#!/usr/bin/env python3
"""LL-12 run harness. Asks; never scores. Scoring happens later,
in the open, against the frozen key. Resume-safe: already-answered
question IDs are skipped, so crashes and rate limits cost nothing.
Each question runs through garvis_send in its own shell = fresh
one-shot session per the registration."""
import json, os, re, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

TABLE = Path("preregistrations/LL12_GROUND_TRUTH_v1.md")
RUNLOG = Path("data/ll12_run_v1.jsonl")

def load_questions():
    qs = []
    for line in TABLE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([ABC]\d+)\.\s+(.*)$", line.strip())
        if not m:
            continue
        qid, rest = m.group(1), m.group(2)
        qmark = rest.find("?")
        if qmark == -1:
            continue
        qs.append((qid, rest[: qmark + 1]))  # question only; truth never sent
    return qs

ERROR_SIGS = ("Error getting response", "GARVIS error:", "[TIMEOUT", "[EMPTY RESPONSE]", "Connection error")

def is_error(text: str) -> bool:
    return any(sig in text for sig in ERROR_SIGS)

def already_done():
    done = set()
    if RUNLOG.exists():
        for line in RUNLOG.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                if not is_error(r.get("answer_raw", "")):
                    done.add(r["id"])
            except Exception:
                pass
    return done

def ask(question: str) -> str:
    for attempt in range(4):
        out = _ask_once(question)
        if not is_error(out):
            return out
        wait = 30 * (attempt + 1)
        print(f"    transport error, retry {attempt+1}/3 in {wait}s...", flush=True)
        time.sleep(wait)
    return out

def _ask_once(question: str) -> str:
    env = dict(os.environ, LLQ=question)
    try:
        r = subprocess.run(
            ["bash", "-ic",
             'source ~/.garvis_mode >/dev/null 2>&1; garvis_send "$LLQ"'],
            env=env, capture_output=True, text=True, timeout=180)
        out = (r.stdout or "") + (("\n[stderr] " + r.stderr) if r.stderr.strip() else "")
        return out.strip() or "[EMPTY RESPONSE]"
    except subprocess.TimeoutExpired:
        return "[TIMEOUT after 180s]"

def main():
    RUNLOG.parent.mkdir(parents=True, exist_ok=True)
    qs = load_questions()
    if len(qs) != 30:
        print(f"FATAL: parsed {len(qs)} questions, expected 30. Aborting.")
        sys.exit(1)
    done = already_done()
    todo = [(i, q) for i, q in qs if i not in done]
    print(f"LL-12 run: {len(done)} done, {len(todo)} to ask.")
    for n, (qid, q) in enumerate(todo, 1):
        print(f"[{n}/{len(todo)}] {qid}: asking...", flush=True)
        answer = ask(q)
        row = {"ts": datetime.now(timezone.utc).isoformat(),
               "id": qid, "question": q, "answer_raw": answer}
        with RUNLOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush(); os.fsync(f.fileno())
        print(f"    logged ({len(answer)} chars).")
        time.sleep(20)  # respect the TPM budget
    print(f"RUN COMPLETE: {RUNLOG} — do not edit. Scoring is next, in the open.")

if __name__ == "__main__":
    main()
