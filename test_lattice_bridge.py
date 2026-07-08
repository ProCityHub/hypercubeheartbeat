import pytest

from lattice_bridge import LatticeBrain


def test_lattice_brain_perceive_returns_reading():
    reading = LatticeBrain().perceive("hello lattice")

    assert 0.0 <= reading["score"] <= 1.0
    assert reading["confidence"] in {"LOW", "MEDIUM", "HIGH"}
    assert reading["O"] and len(reading["O"]) == 3
    assert reading["A"] and len(reading["A"]) == 3
    assert reading["B"] and len(reading["B"]) == 3
    assert set(reading["faculties"]) == {"SIGHT", "GENOME", "VOICE", "RESEARCH"}


def test_lattice_brain_rejects_empty_input():
    with pytest.raises(ValueError):
        LatticeBrain().perceive("   ")
