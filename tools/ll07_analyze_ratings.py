#!/usr/bin/env python3
"""LL-07 word-coordinate analysis.

This script is preregistered for the future LL-07 single-run analysis.
It should not be used on real rating data until the data collection PR
has been approved and manifested.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path


DIMENSIONS = ("O", "A", "B")
ATTENTION_TARGETS = {
    "ATTN_O": {"O": 100.0, "A": 0.0, "B": 0.0},
    "ATTN_A": {"O": 0.0, "A": 100.0, "B": 0.0},
    "ATTN_B": {"O": 0.0, "A": 0.0, "B": 100.0},
}


def read_csv_dicts(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_stimuli(path: str | Path) -> list[dict[str, str]]:
    rows = read_csv_dicts(path)
    if len(rows) != 100:
        raise ValueError(f"Expected 100 stimulus rows, found {len(rows)}")
    required = {"item_id", "stratum", "word"}
    for row in rows:
        if not required.issubset(row):
            raise ValueError("Stimulus file missing required columns")
    return rows


def parse_rating(value: str) -> float:
    try:
        score = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid rating value: {value!r}") from exc
    if score < 0 or score > 100:
        raise ValueError(f"Rating outside 0..100: {score}")
    return score


def valid_raters(rows: list[dict[str, str]], tolerance: float = 5.0) -> list[str]:
    by_rater: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_rater[row["rater_id"]].append(row)

    valid: list[str] = []
    for rater_id, rater_rows in sorted(by_rater.items()):
        failed = False
        by_item = {row["item_id"]: row for row in rater_rows}
        for item_id, targets in ATTENTION_TARGETS.items():
            if item_id not in by_item:
                failed = True
                break
            row = by_item[item_id]
            for dim, target in targets.items():
                if abs(parse_rating(row[dim]) - target) > tolerance:
                    failed = True
                    break
            if failed:
                break
        if not failed:
            valid.append(rater_id)
    return valid


def build_matrix(
    rating_rows: list[dict[str, str]],
    stimuli: list[dict[str, str]],
    dimension: str,
    raters: list[str],
) -> list[list[float]]:
    stimulus_ids = [row["item_id"] for row in stimuli]
    ratings: dict[tuple[str, str], float] = {}

    for row in rating_rows:
        item_id = row["item_id"]
        if item_id.startswith("ATTN_"):
            continue
        if row["rater_id"] in raters:
            ratings[(row["item_id"], row["rater_id"])] = parse_rating(row[dimension])

    matrix: list[list[float]] = []
    for item_id in stimulus_ids:
        item_scores: list[float] = []
        for rater_id in raters:
            key = (item_id, rater_id)
            if key not in ratings:
                raise ValueError(f"Missing rating for {item_id} by {rater_id}")
            item_scores.append(ratings[key])
        matrix.append(item_scores)

    return matrix


def rank_values(values: list[float]) -> list[float]:
    indexed = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][0] == indexed[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][1]] = avg_rank
        i = j + 1
    return ranks


def kendalls_w(matrix: list[list[float]]) -> float:
    n_items = len(matrix)
    if n_items < 2:
        raise ValueError("Need at least two items")
    n_raters = len(matrix[0])
    if n_raters < 2:
        raise ValueError("Need at least two raters")

    rank_sums = [0.0] * n_items
    for rater_index in range(n_raters):
        column = [matrix[item_index][rater_index] for item_index in range(n_items)]
        ranks = rank_values(column)
        for item_index, rank in enumerate(ranks):
            rank_sums[item_index] += rank

    mean_rank_sum = sum(rank_sums) / n_items
    s_value = sum((rank_sum - mean_rank_sum) ** 2 for rank_sum in rank_sums)
    denominator = (n_raters**2) * (n_items**3 - n_items)
    return 12.0 * s_value / denominator


def permutation_p_value(
    matrix: list[list[float]],
    observed_w: float,
    permutations: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    n_items = len(matrix)
    n_raters = len(matrix[0])
    columns = [
        [matrix[item_index][rater_index] for item_index in range(n_items)]
        for rater_index in range(n_raters)
    ]

    exceed = 0
    for _ in range(permutations):
        shuffled_columns = []
        for column in columns:
            shuffled = list(column)
            rng.shuffle(shuffled)
            shuffled_columns.append(shuffled)
        shuffled_matrix = [
            [shuffled_columns[rater_index][item_index] for rater_index in range(n_raters)]
            for item_index in range(n_items)
        ]
        if kendalls_w(shuffled_matrix) >= observed_w:
            exceed += 1

    return (exceed + 1) / (permutations + 1)


def mean_coordinates(
    rating_rows: list[dict[str, str]],
    stimuli: list[dict[str, str]],
    raters: list[str],
) -> list[list[float]]:
    coords: list[list[float]] = []
    for row in stimuli:
        item_id = row["item_id"]
        dim_means = []
        for dim in DIMENSIONS:
            values = [
                parse_rating(rating_row[dim])
                for rating_row in rating_rows
                if rating_row["item_id"] == item_id and rating_row["rater_id"] in raters
            ]
            if len(values) != len(raters):
                raise ValueError(f"Missing coordinate values for {item_id}")
            dim_means.append(sum(values) / len(values))
        coords.append(dim_means)
    return coords


def pc1_variance_ratio(coords: list[list[float]]) -> float:
    n = len(coords)
    means = [sum(row[i] for row in coords) / n for i in range(3)]
    centered = [[row[i] - means[i] for i in range(3)] for row in coords]

    cov = [[0.0 for _ in range(3)] for _ in range(3)]
    for row in centered:
        for i in range(3):
            for j in range(3):
                cov[i][j] += row[i] * row[j]
    for i in range(3):
        for j in range(3):
            cov[i][j] /= max(1, n - 1)

    trace = cov[0][0] + cov[1][1] + cov[2][2]
    if trace <= 0:
        return 1.0

    vector = [1.0, 1.0, 1.0]
    for _ in range(100):
        new_vector = [
            sum(cov[i][j] * vector[j] for j in range(3))
            for i in range(3)
        ]
        norm = math.sqrt(sum(value * value for value in new_vector))
        if norm == 0:
            return 0.0
        vector = [value / norm for value in new_vector]

    numerator = sum(
        vector[i] * sum(cov[i][j] * vector[j] for j in range(3))
        for i in range(3)
    )
    return numerator / trace


def analyze(
    stimuli_path: str | Path,
    ratings_path: str | Path,
    permutations: int = 10000,
    seed: int = 7707,
) -> dict[str, object]:
    stimuli = read_stimuli(stimuli_path)
    ratings = read_csv_dicts(ratings_path)
    raters = valid_raters(ratings)

    if len(raters) < 12:
        raise ValueError(f"Need at least 12 valid raters, found {len(raters)}")

    dimension_results = {}
    for dim_index, dim in enumerate(DIMENSIONS):
        matrix = build_matrix(ratings, stimuli, dim, raters)
        w_value = kendalls_w(matrix)
        p_value = permutation_p_value(matrix, w_value, permutations, seed + dim_index)
        dimension_results[dim] = {
            "kendalls_w": w_value,
            "permutation_p": p_value,
            "meets_dimension_criterion": w_value >= 0.50 and p_value < 0.01,
        }

    coords = mean_coordinates(ratings, stimuli, raters)
    pc1_ratio = pc1_variance_ratio(coords)

    criterion_met = (
        all(result["meets_dimension_criterion"] for result in dimension_results.values())
        and pc1_ratio < 0.80
    )

    return {
        "status": "ll07_single_run_output",
        "n_valid_raters": len(raters),
        "n_words": len(stimuli),
        "dimensions": dimension_results,
        "pc1_variance_ratio": pc1_ratio,
        "criterion_met": criterion_met,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli", default="data/ll07_word_stimuli.csv")
    parser.add_argument("--ratings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7707)
    args = parser.parse_args()

    result = analyze(args.stimuli, args.ratings, args.permutations, args.seed)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
