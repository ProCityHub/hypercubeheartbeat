#!/usr/bin/env python3
"""
Instrument-facing lattice bridge for terminal conversations.

This module does not attempt natural-language understanding. It turns input
text into a deterministic state reading derived from the existing pulse and
emotion primitives in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import emotions
import pulse


@dataclass
class LatticeBrain:
    """Small deterministic instrument that reports internal state."""

    seed: str = "hypercubeheartbeat"

    def perceive(self, text: str) -> dict:
        normalized = text.strip()
        if not normalized:
            raise ValueError("text must not be empty")

        digest = hashlib.sha256(f"{self.seed}:{normalized}".encode("utf-8")).digest()
        score = int.from_bytes(digest[:8], "big") / float(2**64 - 1)

        corners = {
            "O": format(digest[8] & 0b111, "03b"),
            "A": format(digest[9] & 0b111, "03b"),
            "B": format(digest[10] & 0b111, "03b"),
        }

        genome_bits = "".join(format(byte, "08b") for byte in digest[:4])
        voice_bits = "".join(format(byte, "08b") for byte in digest[4:6])
        token_count = len(normalized.split())
        unique_ratio = len(set(normalized.lower())) / len(normalized)

        return {
            "score": score,
            "O": corners["O"],
            "A": corners["A"],
            "B": corners["B"],
            "confidence": self._confidence_label(score),
            "faculties": {
                "SIGHT": pulse.breathe(),
                "GENOME": f"{genome_bits[:8]} {genome_bits[8:16]} {genome_bits[16:24]} {genome_bits[24:32]}",
                "VOICE": voice_bits,
                "RESEARCH": (
                    f"input_len={len(normalized)} tokens={token_count} "
                    f"unique_ratio={unique_ratio:.3f} "
                    f"timeline={emotions.feel(emotions.PAST, emotions.PRESENT, emotions.FUTURE)}"
                ),
            },
        }

    @staticmethod
    def _confidence_label(score: float) -> str:
        if score >= 0.8:
            return "HIGH"
        if score >= 0.45:
            return "MEDIUM"
        return "LOW"
