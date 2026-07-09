import csv
import importlib.util
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
MODULE_PATH = REPO_ROOT / "data" / "convert_probe.py"


def load_convert_probe():
    spec = importlib.util.spec_from_file_location("convert_probe", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_raw_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_convert_probe_maps_columns_and_normalizes_accepts(tmp_path):
    convert_probe = load_convert_probe()
    raw_file = tmp_path / "raw.csv"
    output_file = tmp_path / "ug_probe.csv"
    write_raw_csv(
        raw_file,
        ["offer_amt", "total_stake", "latency_ms", "decision"],
        [
            {"offer_amt": "4", "total_stake": "10", "latency_ms": "250", "decision": "yes"},
            {"offer_amt": "5", "total_stake": "10", "latency_ms": "260", "decision": "reject"},
            {"offer_amt": "6", "total_stake": "12", "latency_ms": "270", "decision": "TRUE"},
            {"offer_amt": "7", "total_stake": "12", "latency_ms": "280", "decision": "0"},
        ],
    )

    rows, _, digest = convert_probe.convert_file(
        raw_file,
        offer_col="offer_amt",
        stake_col="total_stake",
        stake_const=None,
        rt_col="latency_ms",
        accept_col="decision",
        output_file=output_file,
    )

    assert [row["accept"] for row in rows] == [1, 0, 1, 0]
    assert digest
    assert output_file.read_text(encoding="utf-8").splitlines() == [
        "offer,stake,rt_ms,accept",
        "4,10,250,1",
        "5,10,260,0",
        "6,12,270,1",
        "7,12,280,0",
    ]


def test_convert_probe_filters_rt_uses_stake_const_and_keeps_first_n_order(tmp_path):
    convert_probe = load_convert_probe()
    raw_file = tmp_path / "raw.csv"
    output_file = tmp_path / "ug_probe.csv"
    write_raw_csv(
        raw_file,
        ["offer_amt", "latency_ms", "decision"],
        [
            {"offer_amt": "skip-low-rt", "latency_ms": "150", "decision": "yes"},
            {"offer_amt": "3", "latency_ms": "190", "decision": "yes"},
            {"offer_amt": "4", "latency_ms": "220", "decision": "accept"},
            {"offer_amt": "bad", "latency_ms": "230", "decision": "no"},
            {"offer_amt": "5", "latency_ms": "240", "decision": "false"},
            {"offer_amt": "6", "latency_ms": "250", "decision": "1"},
            {"offer_amt": "7", "latency_ms": "260", "decision": "0"},
        ],
    )

    rows, _, _ = convert_probe.convert_file(
        raw_file,
        offer_col="offer_amt",
        stake_col=None,
        stake_const=20.0,
        rt_col="latency_ms",
        accept_col="decision",
        n=3,
        output_file=output_file,
    )

    assert rows == [
        {"offer": 4.0, "stake": 20.0, "rt_ms": 220.0, "accept": 1},
        {"offer": 5.0, "stake": 20.0, "rt_ms": 240.0, "accept": 0},
        {"offer": 6.0, "stake": 20.0, "rt_ms": 250.0, "accept": 1},
    ]


def test_convert_probe_errors_on_missing_file(tmp_path):
    missing = tmp_path / "missing.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            str(missing),
            "--offer-col",
            "offer",
            "--rt-col",
            "rt_ms",
            "--accept-col",
            "accept",
            "--stake-const",
            "10",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "ERROR: raw file not found" in result.stderr
