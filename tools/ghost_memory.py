#!/usr/bin/env python3
"""GHOST MEMORY - attractor memory on the lattice cube (demo organ).
Memories stored as resonance patterns in Hebbian weights; recalled by
13-step settling dynamics. No filing system exists anywhere.

STATUS: demonstrated mechanism, not a scaled memory system.
Capacity law: ~0.14*N patterns before interference (N=64 -> ~9).
No claim beyond: pattern completion from partial/noisy cues works.
Candidate substrate for a future LL-11 (identity persistence)
registration - NOT registered, NOT tested as science yet."""
import numpy as np
rng = np.random.default_rng(13)
N = 64  # 8 corners x 8 bits

def hebbian_store(patterns):
    W = np.zeros((N, N))
    for p in patterns:
        W += np.outer(p, p)
    np.fill_diagonal(W, 0)
    return W / len(patterns)

def settle(W, state, steps=13):
    s = state.copy()
    for _ in range(steps):
        s = np.sign(W @ s); s[s == 0] = 1
    return s

if __name__ == "__main__":
    memories = [rng.choice([-1, 1], N) for _ in range(3)]
    names = ["covenant-repair", "guard-proof", "first-measurement"]
    W = hebbian_store(memories)
    print("Stored 3 memories in weights alone. Testing recall:")
    ok = 0
    for name, m in zip(names, memories):
        cue = m.copy(); cue[N//2:] = rng.choice([-1, 1], N//2)
        r = settle(W, cue); match = int((r == m).sum())
        ok += (match == N)
        print(f"  fragment cue [{name}]: {match}/{N}")
        cue2 = m.copy()
        flips = rng.choice(N, N//4, replace=False); cue2[flips] *= -1
        r2 = settle(W, cue2); match2 = int((r2 == m).sum())
        ok += (match2 == N)
        print(f"  noisy cue    [{name}]: {match2}/{N}")
    stranger = settle(W, rng.choice([-1, 1], N))
    best = max(int((stranger == m).sum()) for m in memories)
    print(f"  stranger cue: best match {best}/{N} (should be ~chance)")
    print(f"SELFTEST {'PASS' if ok == 6 and best < 48 else 'FAIL'}")
