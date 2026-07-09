#!/usr/bin/env python3
"""Convert externally-downloaded probe CSV data into the pre-registered format."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path

FIELDNAMES = ["offer", "stake", "rt_ms", "accept"]
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("ug_probe.csv")
TRUE_VALUES = {"1", "yes", "accept", "true"}
FALSE_VALUES = {"0", "no", "reject", "false"}


def parse_number(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite number")
    return number


def normalize_accept(value: object) -> int:
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return 1
    if text in FALSE_VALUES:
        return 0
    raise ValueError(f"unsupported accept value: {value!r}")


def format_number(value: float) -> str:
    return format(value, "g")


def collect_rows(
    raw_file: Path,
    *,
    offer_col: str,
    stake_col: str | None,
    stake_const: float | None,
    rt_col: str,
    accept_col: str,
    n: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    with raw_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            try:
                offer = parse_number(raw_row[offer_col])
                stake = stake_const if stake_const is not None else parse_number(raw_row[stake_col])  # type: ignore[index]
                rt_ms = parse_number(raw_row[rt_col])
                accept = normalize_accept(raw_row[accept_col])
            except (KeyError, TypeError, ValueError):
                continue

            if rt_ms < 200:
                continue

            rows.append(
                {
                    "offer": offer,
                    "stake": stake,
                    "rt_ms": rt_ms,
                    "accept": accept,
                }
            )
            if len(rows) >= n:
                break

    return rows


def write_output(rows: list[dict[str, object]], output_file: Path) -> None:
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "offer": format_number(float(row["offer"])),
                    "stake": format_number(float(row["stake"])),
                    "rt_ms": format_number(float(row["rt_ms"])),
                    "accept": str(int(row["accept"])),
                }
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_file(
    raw_file: str | Path,
    *,
    offer_col: str,
    stake_col: str | None,
    stake_const: float | None,
    rt_col: str,
    accept_col: str,
    n: int = 20,
    output_file: str | Path = DEFAULT_OUTPUT,
) -> tuple[list[dict[str, object]], Path, str]:
    raw_path = Path(raw_file)
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw file not found: {raw_path}")
    if n <= 0:
        raise ValueError("n must be positive")

    rows = collect_rows(
        raw_path,
        offer_col=offer_col,
        stake_col=stake_col,
        stake_const=stake_const,
        rt_col=rt_col,
        accept_col=accept_col,
        n=n,
    )
    if not rows:
        raise ValueError("no valid rows found; never generate or invent data")

    output_path = Path(output_file)
    write_output(rows, output_path)
    return rows, output_path, sha256_file(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert externally-downloaded probe CSV data without generating any rows."
    )
    parser.add_argument("raw_file")
    parser.add_argument("--offer-col", required=True)
    parser.add_argument("--rt-col", required=True)
    parser.add_argument("--accept-col", required=True)
    stake_group = parser.add_mutually_exclusive_group(required=True)
    stake_group.add_argument("--stake-col")
    stake_group.add_argument("--stake-const", type=float)
    parser.add_argument("--n", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        rows, output_path, digest = convert_file(
            args.raw_file,
            offer_col=args.offer_col,
            stake_col=args.stake_col,
            stake_const=args.stake_const,
            rt_col=args.rt_col,
            accept_col=args.accept_col,
            n=args.n,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Rows written: {len(rows)}")
    for index, row in enumerate(rows, start=1):
        print(
            f"{index}: offer={format_number(float(row['offer']))}, "
            f"stake={format_number(float(row['stake']))}, "
            f"rt_ms={format_number(float(row['rt_ms']))}, "
            f"accept={int(row['accept'])}"
        )
    print(f"Output: {output_path}")
    print(f"SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
