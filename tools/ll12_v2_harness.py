#!/usr/bin/env python3
"""LL-12 v2 harness. Same 30 frozen questions, run under two launch
conditions. The condition (control/treatment) is passed as arg 1 and
recorded on every row so the two runs never mix. Resume-safe per
condition. Asks only; never scores.

Usage:
  CONTROL   (from ~/GARVIS):          python3 <path>/ll12_v2_harness.py control
  TREATMENT (from ~/hypercubeheartbeat): python3 tools/ll12_v2_harness.py treatment
"""
import json, os, re, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

HB = Path.home() / "hypercubeheartbeat"
TABLE = HB / "preregistrations" / "LL12_GROUND_TRUTH_v1.md"
ERRS = ("Error getting response", "GARVIS error:", "[TIMEOUT", "[EMPTY RESPONSE]", "Connection error", "rate_limit", "RateLimit")

def load_questions():
    qs = []
    for line in TABLE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([ABC]\d+)\.\s+(.*)$", line.strip())
        if not m: continue
        q = m.group(2)
        qm = q.find("?")
        if qm != -1: qs.append((m.group(1), q[:qm+1]))
    return qs

def is_error(t): return any(e in t for e in ERRS)

def ask(question):
    for attempt in range(4):
        env = dict(os.environ, LLQ=question)
        try:
            r = subprocess.run(["bash","-ic",'source ~/.garvis_mode >/dev/null 2>&1; garvis_send "$LLQ"'],
                               env=env, capture_output=True, text=True, timeout=240)
            out = (r.stdout or "").strip()
            if out and not is_error(out): return out
            last = out or "[EMPTY RESPONSE]"
        except subprocess.TimeoutExpired:
            last = "[TIMEOUT]"
        if attempt < 3:
            print(f"    transport/rate issue, wait {45*(attempt+1)}s...", flush=True)
            time.sleep(45*(attempt+1))
    return last

def main():
    cond = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    if cond not in ("control","treatment"):
        print("FATAL: first arg must be 'control' or 'treatment'"); sys.exit(1)
    runlog = HB / "data" / f"ll12_v2_{cond}.jsonl"
    runlog.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if runlog.exists():
        for l in runlog.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(l)
                if not is_error(r.get("answer_raw","")): done.add(r["id"])
            except Exception: pass
    qs = load_questions()
    todo = [(i,q) for i,q in qs if i not in done]
    print(f"LL-12 v2 [{cond}] launched from {Path.cwd()}")
    print(f"{len(done)} done, {len(todo)} to ask.")
    for n,(qid,q) in enumerate(todo,1):
        print(f"[{n}/{len(todo)}] {qid}: asking...", flush=True)
        ans = ask(q)
        with runlog.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts":datetime.now(timezone.utc).isoformat(),
                    "condition":cond,"launch_dir":str(Path.cwd()),
                    "id":qid,"question":q,"answer_raw":ans}, ensure_ascii=False)+"\n")
            f.flush(); os.fsync(f.fileno())
        print(f"    logged ({len(ans)} chars).")
        time.sleep(20)
    print(f"RUN COMPLETE [{cond}]: {runlog}")

if __name__ == "__main__":
    main()
