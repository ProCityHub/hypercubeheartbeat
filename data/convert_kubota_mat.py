#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np
import scipy.io

SOURCE_URL = "https://osf.io/ugdcz"
STAKE = 10.0
RT_FLOOR_MS = 200.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_float(value):
    try:
        v = float(value)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def get_field(obj, name: str):
    if isinstance(obj, dict):
        if name in obj:
            return obj[name]
        for key in obj:
            if str(key).lower() == name.lower():
                return obj[key]

    if hasattr(obj, name):
        return getattr(obj, name)

    fields = list(getattr(obj, "_fieldnames", []) or [])
    for field in fields:
        if str(field).lower() == name.lower():
            return getattr(obj, field)

    raise KeyError(f"{name}; available fields: {fields}")


def as_two_col(value, name: str):
    arr = np.asarray(value)
    arr = np.squeeze(arr)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{name} expected 2 columns, got {arr.shape}")
    return arr


def load_mat(path: Path):
    try:
        return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False), "scipy.io.loadmat"
    except NotImplementedError:
        import mat73
        return mat73.loadmat(path), "mat73.loadmat"


def normalize_decision(value):
    v = finite_float(value)
    if v is None:
        return None, "no_response"
    rounded = round(v)
    if abs(v - rounded) > 1e-9:
        return None, f"unexpected:{v}"
    code = int(rounded)
    if code == 1:
        return 1, None
    if code == 2:
        return 0, None
    return None, f"unexpected:{code}"


def mat_members(raw_zip: Path):
    with zipfile.ZipFile(raw_zip) as z:
        return sorted(
            n for n in z.namelist()
            if n.lower().endswith(".mat") and "__macosx" not in n.lower()
        )


def process_mat_file(mat_path: Path, member_name: str):
    data, loader = load_mat(mat_path)
    if "behavior" not in data:
        raise SystemExit(f"STOP: {member_name} has no behavior structure")

    behavior = data["behavior"]
    ptoffer = as_two_col(get_field(behavior, "pTOffer"), "pTOffer")
    utdecision = as_two_col(get_field(behavior, "utDecision"), "utDecision")

    if ptoffer.shape[0] != utdecision.shape[0]:
        raise SystemExit(f"STOP: row mismatch in {member_name}")

    counts = Counter()
    counts["total_trials"] = int(ptoffer.shape[0])
    unexpected_codes = Counter()
    raw_latencies = []
    candidate_rows = []

    for i in range(ptoffer.shape[0]):
        offer_time = finite_float(ptoffer[i, 0])
        offer = finite_float(ptoffer[i, 1])
        decision_time = finite_float(utdecision[i, 0])
        decision_raw = utdecision[i, 1]

        if offer_time is None or decision_time is None:
            counts["excluded_bad_parse"] += 1
            continue

        raw_latency = decision_time - offer_time
        if not math.isfinite(raw_latency) or raw_latency <= 0:
            counts["excluded_nonpositive_latency"] += 1
            continue

        raw_latencies.append(raw_latency)

        if offer is None:
            counts["excluded_bad_parse"] += 1
            continue

        accept, reason = normalize_decision(decision_raw)

        if reason == "no_response":
            counts["excluded_no_response"] += 1
            continue

        if reason and reason.startswith("unexpected:"):
            code = reason.split(":", 1)[1]
            unexpected_codes[code] += 1
            counts["excluded_unexpected_decision"] += 1
            continue

        candidate_rows.append(
            {
                "source_file": member_name,
                "trial_index": i,
                "offer": offer,
                "raw_latency": raw_latency,
                "accept": accept,
            }
        )

    return {
        "loader": loader,
        "counts": counts,
        "unexpected_codes": unexpected_codes,
        "raw_latencies": raw_latencies,
        "candidate_rows": candidate_rows,
    }


def write_probe_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["offer", "stake", "rt_ms", "accept"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "offer": row["offer"],
                    "stake": STAKE,
                    "rt_ms": row["rt_ms"],
                    "accept": row["accept"],
                }
            )


def write_notes(
    path: Path,
    raw_zip: Path,
    raw_sha: str,
    output_csv: Path,
    output_sha: str,
    loader_counts: Counter,
    per_file_counts,
    totals: Counter,
    unexpected_codes: Counter,
    median_raw_latency: float,
    unit_decision: str,
    multiplier: float,
    command: str,
):
    lines = [
        "# Phase 4 Step 2b-1C — Kubota MAT Conversion Notes",
        "",
        f"Source URL: {SOURCE_URL}",
        "",
        "This PR performs conversion only.",
        "",
        "No scoring was performed.",
        "",
        "No preregistration_test.py execution was performed.",
        "",
        "No empirical outcome label was produced.",
        "",
        "## Raw input",
        "",
        f"- Raw archive: `{raw_zip.name}`",
        f"- Raw archive sha256: `{raw_sha}`",
        "",
        "## Loader used",
        "",
    ]

    for loader, count in sorted(loader_counts.items()):
        lines.append(f"- `{loader}`: {count} file(s)")

    lines += [
        "",
        "## Converter invocation",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Frozen mapping applied",
        "",
        "- `offer = pTOffer[:, 1]`",
        "- `stake = 10`",
        "- `rt_ms = utDecision[:, 0] - pTOffer[:, 0]`",
        "- `accept = utDecision[:, 1]`, normalized as `1 -> 1`, `2 -> 0`",
        "",
        "## Timing unit decision",
        "",
        f"- Median raw latency before unit conversion: `{median_raw_latency}`",
        f"- Unit decision: `{unit_decision}`",
        f"- Multiplier applied to raw latency: `{multiplier}`",
        "",
        "The unit decision used timing magnitude only.",
        "",
        "No offer-vs-RT, accept-vs-RT, or offer-vs-accept statistic was computed.",
        "",
        "## Row counts",
        "",
        f"- Total raw trials: `{totals['total_trials']}`",
        f"- Excluded no-response trials: `{totals['excluded_no_response']}`",
        f"- Excluded bad-parse rows: `{totals['excluded_bad_parse']}`",
        f"- Excluded nonpositive-latency rows: `{totals['excluded_nonpositive_latency']}`",
        f"- Excluded unexpected-decision rows: `{totals['excluded_unexpected_decision']}`",
        f"- Excluded rt_ms below {RT_FLOOR_MS}: `{totals['excluded_rt_floor']}`",
        f"- Final output rows: `{totals['final_rows']}`",
        "",
        "## Unexpected decision codes",
        "",
    ]

    if unexpected_codes:
        for code, count in sorted(unexpected_codes.items()):
            lines.append(f"- `{code}`: {count}")
    else:
        lines.append("- None observed.")

    lines += [
        "",
        "## Per-file trial counts",
        "",
        "| File | Total | Included | No response | Bad parse | Nonpositive latency | Unexpected decision | RT floor |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for item in per_file_counts:
        c = item["counts"]
        lines.append(
            f"| {item['file']} | {c['total_trials']} | {c['included']} | "
            f"{c['excluded_no_response']} | {c['excluded_bad_parse']} | "
            f"{c['excluded_nonpositive_latency']} | "
            f"{c['excluded_unexpected_decision']} | {c['excluded_rt_floor']} |"
        )

    lines += [
        "",
        "## Output",
        "",
        f"- Output file: `{output_csv}`",
        f"- Output sha256: `{output_sha}`",
        "",
        "## Claim discipline",
        "",
        "This conversion is not an empirical result.",
        "",
        "This conversion does not score the Lattice Law.",
        "",
        "This conversion does not prove consciousness.",
        "",
        "This conversion does not prove AGI.",
        "",
        "The only purpose of this file is to document the reproducible data conversion before the single preregistered run.",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-zip",
        default=str(Path.home() / "kubota_conversion" / "raw" / "UltimatumRawData.zip"),
    )
    parser.add_argument("--output", default="data/ug_probe.csv")
    parser.add_argument("--notes", default="data/KUBOTA_CONVERSION_NOTES.md")
    args = parser.parse_args()

    raw_zip = Path(args.raw_zip).expanduser()
    output_csv = Path(args.output)
    notes_path = Path(args.notes)

    if not raw_zip.exists():
        raise SystemExit(f"STOP: raw archive not found: {raw_zip}")

    raw_sha = sha256_file(raw_zip)
    members = mat_members(raw_zip)

    if not members:
        raise SystemExit("STOP: no .mat files found in raw archive")

    all_candidates = []
    all_raw_latencies = []
    loader_counts = Counter()
    totals = Counter()
    unexpected_codes = Counter()
    per_file_counts = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        with zipfile.ZipFile(raw_zip) as z:
            for member in members:
                mat_path = tmp / Path(member).name
                mat_path.write_bytes(z.read(member))

                result = process_mat_file(mat_path, member)

                loader_counts[result["loader"]] += 1
                all_candidates.extend(result["candidate_rows"])
                all_raw_latencies.extend(result["raw_latencies"])
                unexpected_codes.update(result["unexpected_codes"])

                counts = result["counts"]
                for key, value in counts.items():
                    totals[key] += value

                per_file_counts.append({"file": member, "counts": counts})

    if not all_raw_latencies:
        raise SystemExit("STOP: no valid positive raw latencies found")

    median_raw_latency = float(median(all_raw_latencies))

    if median_raw_latency < 200:
        unit_decision = "seconds_to_milliseconds"
        multiplier = 1000.0
    else:
        unit_decision = "already_milliseconds"
        multiplier = 1.0

    final_rows = []
    included_by_file = Counter()
    rt_floor_by_file = Counter()

    for row in all_candidates:
        rt_ms = row["raw_latency"] * multiplier

        if rt_ms < RT_FLOOR_MS:
            totals["excluded_rt_floor"] += 1
            rt_floor_by_file[row["source_file"]] += 1
            continue

        final_rows.append(
            {
                "offer": row["offer"],
                "rt_ms": rt_ms,
                "accept": row["accept"],
            }
        )
        included_by_file[row["source_file"]] += 1

    if not final_rows:
        raise SystemExit(
            "STOP: no rows survived declared exclusions and rt_ms floor"
        )

    totals["final_rows"] = len(final_rows)

    for item in per_file_counts:
        file_name = item["file"]
        item["counts"]["included"] = included_by_file[file_name]
        item["counts"]["excluded_rt_floor"] = rt_floor_by_file[file_name]

    write_probe_csv(output_csv, final_rows)
    output_sha = sha256_file(output_csv)

    command = " ".join(
        shlex.quote(part)
        for part in [sys.executable, *sys.argv]
    )

    write_notes(
        notes_path,
        raw_zip,
        raw_sha,
        output_csv,
        output_sha,
        loader_counts,
        per_file_counts,
        totals,
        unexpected_codes,
        median_raw_latency,
        unit_decision,
        multiplier,
        command,
    )

    result = {
        "source_url": SOURCE_URL,
        "raw_archive": str(raw_zip),
        "raw_sha256": raw_sha,
        "mat_files": len(members),
        "total_trials": totals["total_trials"],
        "final_rows": totals["final_rows"],
        "median_raw_latency": median_raw_latency,
        "unit_decision": unit_decision,
        "output": str(output_csv),
        "output_sha256": output_sha,
        "notes": str(notes_path),
    }

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
