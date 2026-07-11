#!/usr/bin/env python3
"""Generate synthetic LL-07 demo ratings.

This tool creates artificial ratings for pipeline testing only.

Synthetic demo ratings are not human data.
Synthetic demo ratings are not evidence.
Synthetic demo ratings must not be used as an LL-07 empirical result.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DIMENSIONS = ("O", "A", "B")


def read_stimuli(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100:
        raise ValueError(f"Expected 100 stimuli, found {len(rows)}")
    return rows


def clamp(value: int) -> int:
    return max(0, min(100, value))


def score_for(item_index: int, rater_index: int, dimension: str) -> int:
    offsets = {"O": 11, "A": 47, "B": 73}
    base = (item_index * 7 + offsets[dimension]) % 101
    rater_noise = (rater_index % 5) - 2
    return clamp(base + rater_noise)


def generate_rows(stimuli: list[dict[str, str]], n_raters: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for rater_index in range(n_raters):
        rater_id = f"DEMO_RATER_{rater_index + 1:03d}"

        rows.append({
            "rater_id": rater_id,
            "item_id": "ATTN_O",
            "word": "attention check: enter O=100, A=0, B=0",
            "O": 100,
            "A": 0,
            "B": 0,
        })
        rows.append({
            "rater_id": rater_id,
            "item_id": "ATTN_A",
            "word": "attention check: enter O=0, A=100, B=0",
            "O": 0,
            "A": 100,
            "B": 0,
        })
        rows.append({
            "rater_id": rater_id,
            "item_id": "ATTN_B",
            "word": "attention check: enter O=0, A=0, B=100",
            "O": 0,
            "A": 0,
            "B": 100,
        })

        for item_index, stimulus in enumerate(stimuli):
            rows.append({
                "rater_id": rater_id,
                "item_id": stimulus["item_id"],
                "word": stimulus["word"],
                "O": score_for(item_index, rater_index, "O"),
                "A": score_for(item_index, rater_index, "A"),
                "B": score_for(item_index, rater_index, "B"),
            })

    return rows


def write_rows(path: str | Path, rows: list[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rater_id", "item_id", "word", "O", "A", "B"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli", default="data/ll07_word_stimuli.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-raters", type=int, default=12)
    args = parser.parse_args()

    if args.n_raters < 1:
        raise SystemExit("--n-raters must be at least 1")

    stimuli = read_stimuli(args.stimuli)
    rows = generate_rows(stimuli, args.n_raters)
    write_rows(args.output, rows)

    print("Synthetic LL-07 demo ratings generated.")
    print("This file is for pipeline testing only.")
    print("It is not human data and not evidence.")
    print(f"Output: {args.output}")
    print(f"Demo raters: {args.n_raters}")
    print(f"Rows excluding header: {len(rows)}")


if __name__ == "__main__":
    main()
