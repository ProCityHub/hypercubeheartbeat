#!/usr/bin/env python3
"""
LATTICE BRIDGE — the one real bridge.
=====================================

Wires together, in one executable pipeline, the components that until now
lived in separate files with no connection:

    sacred_binary_cube.py  → geometry, binary state machine, corner set
    pulse.py               → conscious/sub/super rhythm (101 / 010 / 001)
    emotions.py            → time-as-feeling wave
    THIS FILE              → canonical Lattice Brain scorer + LIF corner nodes

Canonical formula (pre-registered):   score = O^1 · A^(1/φ) · B^(1/φ²)
φ in the EXPONENTS, never as a scalar multiplier.

Deterministic end to end. Stdlib only. No network, no randomness, no LLM.
This file makes no consciousness claims: it is a cognitive-architecture
simulation whose scoring rule is awaiting external behavioral test data.
"""

import math
import hashlib

# ── Live repo imports (the actual bridge) ────────────────────────────────
from sacred_binary_cube import BinaryState, C, ROT, PHI
from emotions import feel, PAST, PRESENT, FUTURE

# Pulse layer constants (mirrors pulse.py without triggering its demo I/O)
CONSCIOUS, SUBCONSCIOUS, SUPERCONSCIOUS = 0b101, 0b010, 0b001

# ── 1. Deterministic vectorization ───────────────────────────────────────
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19]

def prime_features(text: str) -> list[float]:
    """Content → 8 deterministic features in [0,1], one per cube corner.
    SHA-256 based: same input always yields the same vector."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return [(h[p % len(h)] ^ h[(p * p) % len(h)]) / 255.0 for p in _PRIMES]

# ── 2. Sub-metrics with provenance ───────────────────────────────────────
def extract_OAB(text: str) -> dict:
    """Observer / Actor / Binary-environment sub-metrics.
    Every value is tagged 'measured' or 'fallback' — the system never
    reports a confident number it did not actually compute."""
    words = text.split()
    f = prime_features(text)

    if words:
        # O: observation richness — lexical diversity
        O = (len(set(w.lower() for w in words)) / len(words), "measured")
        # A: action density — verbs proxied by feature energy of word lengths
        A = (min(1.0, sum(len(w) for w in words) / (8.0 * len(words))), "measured")
    else:
        O = (0.5, "fallback")
        A = (0.5, "fallback")

    # B: environment signal from the content hash itself
    B = (sum(f) / len(f), "measured") if text else (0.5, "fallback")
    return {"O": O, "A": A, "B": B}

# ── 3. Canonical score ───────────────────────────────────────────────────
def lattice_score(O: float, A: float, B: float) -> float:
    """O^1 · A^(1/φ) · B^(1/φ²).  Exponent placement is what makes the
    rule falsifiable: it changes score ORDERING, not just scale."""
    eps = 1e-9
    return (max(O, eps) ** 1.0
            * max(A, eps) ** (1.0 / PHI)
            * max(B, eps) ** (1.0 / PHI ** 2))

# ── 4. Leaky integrate-and-fire corner nodes ─────────────────────────────
class CornerNode:
    """One LIF unit sitting on a binary-charged cube corner."""
    LEAK, THRESHOLD = 0.90, 1.0

    def __init__(self, index: int, coords: list[int]):
        self.index = index
        self.coords = coords                      # from live C()
        self.charge = bin(index).count("1") / 3.0 # corner binary weight
        self.potential = 0.0
        self.fires = 0

    def step(self, drive: float, gate: int) -> bool:
        self.potential = self.potential * self.LEAK + drive * self.charge * gate
        if self.potential >= self.THRESHOLD:
            self.potential = 0.0
            self.fires += 1
            return True
        return False

# ── 5. The assembled Brain ───────────────────────────────────────────────
class LatticeBrain:
    """Cube geometry + pulse gating + canonical scorer + LIF nodes,
    read out through four faculties."""

    def __init__(self):
        self.state = BinaryState()                       # live repo state machine
        self.nodes = [CornerNode(i, c) for i, c in enumerate(C())]
        self.rhythm = format(
            (SUPERCONSCIOUS << 0b110) | (CONSCIOUS << 0b11) | SUBCONSCIOUS, "09b"
        )                                                # e.g. '001101010'

    def perceive(self, text: str, ticks: int = 27) -> dict:
        sub = extract_OAB(text)
        score = lattice_score(sub["O"][0], sub["A"][0], sub["B"][0])
        provenance = {k: v[1] for k, v in sub.items()}
        confidence = ("high" if all(p == "measured" for p in provenance.values())
                      else "low")

        fired = []
        for t in range(ticks):
            gate = int(self.rhythm[t % len(self.rhythm)])   # pulse gates drive
            for node in self.nodes:
                # geometry feeds dynamics: rotation magnitude modulates drive
                rot = ROT(node.coords, self.state.time)
                geo = sum(abs(x) for x in rot) / 3.0
                if node.step(score * geo, gate):
                    fired.append((t, node.index))
            self.state.tick()

        return {
            "input": text,
            "O": round(sub["O"][0], 4), "A": round(sub["A"][0], 4),
            "B": round(sub["B"][0], 4),
            "score": round(score, 6),
            "provenance": provenance, "confidence": confidence,
            "fires": fired,
            "faculties": self._faculties(score, fired),
        }

    # ── Faculties: four honest readouts of node state ────────────────────
    def _faculties(self, score: float, fired: list) -> dict:
        rate = len(fired)
        top = max(self.nodes, key=lambda n: n.fires)
        return {
            "SIGHT":    f"{rate} firing events; dominant corner {top.index:03b}",
            "VOICE":    feel(PAST, PRESENT, FUTURE),          # live emotions.py
            "RESEARCH": f"score={score:.4f} awaiting UG/IGT external validation",
            "GENOME":   "".join(str(int(n.potential > 0.5)) for n in self.nodes),
        }

# ── Demo ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    brain = LatticeBrain()
    print("LATTICE BRIDGE — cube + pulse + emotions + canonical scorer, unified")
    print(f"phi = {PHI:.6f} | rhythm = {brain.rhythm}")
    print("-" * 64)
    for text in [
        "the observer stands at the center of the lattice",
        "the observer stands at the center of the lattice",   # determinism check
        "drywall estimate for the northside job, two coats, level five finish",
        "",
    ]:
        r = brain.perceive(text)
        print(f"\nINPUT: {text!r}")
        print(f"  O={r['O']}  A={r['A']}  B={r['B']}  →  score={r['score']}"
              f"  [{r['confidence']} confidence: {r['provenance']}]")
        for name, val in r["faculties"].items():
            print(f"  {name:8s} {val}")
