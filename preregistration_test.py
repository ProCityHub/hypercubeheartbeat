#!/usr/bin/env python3
"""
PRE-REGISTRATION TEST — Lattice Law Behavioral Test v1.0
Implements PREREGISTRATION.md exactly. Do not modify after data contact.

Usage:
    python3 preregistration_test.py data/ug_probe.csv

CSV columns (header required): offer, stake, rt_ms, accept
    offer   : amount offered to responder
    stake   : total stake for that round
    rt_ms   : responder reaction time in milliseconds
    accept  : 1 = accepted, 0 = rejected

Stdlib only. Deterministic. Prints PASS / FAIL per pre-registered criteria.
"""

import csv
import math
import sys

PHI = (1 + math.sqrt(5)) / 2
EPS = 1e-9
CRITERION_MARGIN = 0.05   # AUC(phi) - AUC(flat) must be >= this


def load_rows(path):
    rows, excluded = [], 0
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                offer = float(r["offer"]); stake = float(r["stake"])
                rt = float(r["rt_ms"]);   accept = int(r["accept"])
            except (KeyError, ValueError):
                excluded += 1
                continue
            if rt < 200:            # pre-registered exclusion: anticipatory
                excluded += 1
                continue
            rows.append((offer, stake, rt, accept))
    return rows, excluded


def normalize(rows):
    rts = [r[2] for r in rows]
    stakes = [r[1] for r in rows]
    rt_min, rt_max = min(rts), max(rts)
    stake_max = max(stakes)
    out = []
    for offer, stake, rt, accept in rows:
        O = offer / stake if stake else 0.0                       # fairness
        A = 1 - (rt - rt_min) / (rt_max - rt_min) if rt_max > rt_min else 0.5
        B = stake / stake_max if stake_max else 0.0               # stake size
        out.append((O, A, B, accept))
    return out


def score(O, A, B, e_o, e_a, e_b):
    return (max(O, EPS) ** e_o) * (max(A, EPS) ** e_a) * (max(B, EPS) ** e_b)


def auc(scores, labels):
    """Rank-based AUC (Mann-Whitney), ties counted as 0.5. Deterministic."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main(path):
    rows, excluded = load_rows(path)
    print(f"Loaded {len(rows)} rows ({excluded} excluded per pre-registration)")
    if len(rows) < 10:
        print("ABORT: fewer than 10 usable rows — probe spec requires 10–20.")
        return

    data = normalize(rows)
    labels = [d[3] for d in data]

    s_phi = [score(O, A, B, 1.0, 1 / PHI, 1 / PHI**2) for O, A, B, _ in data]
    s_flat = [score(O, A, B, 1.0, 1.0, 1.0) for O, A, B, _ in data]

    auc_phi, auc_flat = auc(s_phi, labels), auc(s_flat, labels)
    if auc_phi is None:
        print("ABORT: labels are all one class — AUC undefined.")
        return

    diff = auc_phi - auc_flat
    print(f"AUC  phi-exponents  (O^1 A^0.618 B^0.382): {auc_phi:.4f}")
    print(f"AUC  flat-exponents (O^1 A^1     B^1    ): {auc_flat:.4f}")
    print(f"Difference: {diff:+.4f}   (criterion: >= +{CRITERION_MARGIN})")

    if auc_phi <= 0.5:
        print("RESULT: FAIL — phi model at or below chance.")
    elif diff >= CRITERION_MARGIN:
        print("RESULT: PASS (pilot) — signal detected. Next step: pre-register "
              "full test at N >= 200. This pilot alone confirms nothing.")
    else:
        print("RESULT: FAIL — phi exponents do not beat flat exponents by the "
              "pre-registered margin. Report this negative result publicly.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
