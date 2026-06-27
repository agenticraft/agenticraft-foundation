"""CLI tests for ``python -m agenticraft_foundation.resilience``.

Verifies the CI-gating exit codes (0 met / 1 unmet / 2 input error) and the
text + JSON output surfaces of the argparse entry point.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agenticraft_foundation.resilience.__main__ import main

_FIXTURES = Path(__file__).parent / "fixtures"
_STAR = _FIXTURES / "resilience_star_app.yaml"  # hub => f_crash=0, f_byz=0
_MESH = _FIXTURES / "resilience_mesh_app.yaml"  # complete(4) => f_byz=1


def test_no_target_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(_STAR)]) == 0
    out = capsys.readouterr().out
    assert "f_crash" in out and "f_byz" in out


def test_unmet_byzantine_target_exits_one(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(_STAR), "--target-byzantine", "1"]) == 1
    assert "NOT MET" in capsys.readouterr().out


def test_met_target_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(_MESH), "--target-byzantine", "1"]) == 0


def test_json_output_is_capacity_tagged(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(_STAR), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["f_byz"] == 0
    assert data["byz_semantics"] == "capacity"


def test_missing_file_exits_two(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert main([str(tmp_path / "nope.yaml")]) == 2


def test_malformed_topology_exits_two_not_one(tmp_path: Path) -> None:
    # Structurally malformed manifest must be an input error (2), not be
    # conflated with the "target not met" code (1).
    bad = tmp_path / "bad.yaml"
    bad.write_text("agents:\n  - id: a\ntopology:\n  connections:\n    - oops\n")
    assert main([str(bad)]) == 2
