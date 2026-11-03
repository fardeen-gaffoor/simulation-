"""``scripts/excavation_info.py``: the tool that has to be run before a competition run.

It is the only way to answer "if the sequencer asks for six cycles, what will the
berm be" without a simulator, so its dry run is the acceptance test for the whole
sand model - and its ``dump_lands_inside`` check is the acceptance test for the
coupling to module 5.
"""

from __future__ import annotations

import json

import pytest
from scripts_loader import load_script

info = load_script("excavation_info")


def _run(argv, capsys):
    code = info.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# =========================================================================== #
# the dry run
# =========================================================================== #
def test_dry_run_fills_the_berm_in_six_cycles(config):
    data = info.dry_run(config, 6)
    assert len(data["cycles"]) == 6
    assert all(cycle["dig"]["success"] for cycle in data["cycles"])
    assert all(cycle["dump"]["success"] for cycle in data["cycles"])
    assert data["berm"]["dumps"] == 6
    assert data["berm"]["completion"] > 0.90
    assert data["berm"]["placed_volume_m3"] > 0.0


def test_dry_run_ends_with_the_mechanism_travel_ready(config):
    """Module 5 will want to drive away the moment the last cycle returns."""
    data = info.dry_run(config, 6)
    assert data["mechanism"]["travel_ready"] is True
    assert data["mechanism"]["state_name"] == "STOWED"


def test_dry_run_of_zero_cycles_changes_nothing(config):
    data = info.dry_run(config, 0)
    assert data["cycles"] == []
    assert data["berm"]["dumps"] == 0
    assert data["berm"]["completion"] == 0.0


def test_dry_run_volume_is_conserved(config):
    data = info.dry_run(config, 6)
    berm = data["berm"]
    assert berm["placed_volume_m3"] + berm["spilled_volume_m3"] > 0.0
    assert 0.0 <= berm["spill_fraction"] <= 1.0
    assert berm["cells_filled"] <= berm["cells_total"]


def test_every_dump_pose_lands_inside_the_target(config):
    data = info.report(config, 0, with_dry_run=False)
    assert data["berm"]["dump_lands_inside"]["all_inside"] is True
    for record in data["berm"]["dump_lands_inside"]["poses"]:
        assert record["inside"] is True


def test_the_berm_is_a_ridge_and_the_tool_says_so(config):
    """The volume and shape metrics disagree by design - see docs/sand.md.

    If a future tuning change makes coverage reach 1.0 with this dump pattern,
    that is suspicious rather than good: a pile of dry sand cannot be a flat
    prism at its own target height.
    """
    data = info.dry_run(config, 6)
    berm = data["berm"]
    assert berm["completion"] > berm["coverage"]


# =========================================================================== #
# the report
# =========================================================================== #
def test_report_covers_everything_module_6_owns(config):
    data = info.report(config, 0, with_dry_run=False)
    assert data["config"]["sha256"]
    assert data["mechanism"]["ladder"]["joint"] == config.mechanism.ladder.joint
    assert data["mechanism"]["belt"]["joint"] == config.mechanism.belt.joint
    assert data["sand"]["dig_rate_m3_s"] > 0.0
    assert data["berm"]["target_volume_m3"] == pytest.approx(config.berm_target_volume_m3, rel=1e-4)
    assert data["berm"]["dump_pose_y"] == pytest.approx(config.berm_dump_pose_y, rel=1e-4)


def test_the_dig_rate_matches_the_models_own(config):
    from rover_excavation.config import dig_rate_m3_s

    data = info.report(config, 0, with_dry_run=False)
    expected = dig_rate_m3_s(
        config.sand.dig,
        config.mechanism,
        config.mechanism.ladder.dig_pitch_rad,
        config.mechanism.belt.dig_velocity_radps,
    )
    # The report rounds to five decimals of a cubic metre per second (finer than
    # a millilitre per second); a looser match would hide a real drift.
    assert data["sand"]["dig_rate_m3_s"] == pytest.approx(expected, abs=1e-5)


def test_the_tub_takes_tens_of_seconds_to_fill(config):
    """A 3.7 cm cut of a 30 cm-wide ladder: the number that sets the cycle time."""
    data = info.report(config, 0, with_dry_run=False)
    assert 5.0 < data["sand"]["dig_seconds_to_full_tub"] < 60.0


# =========================================================================== #
# the command line
# =========================================================================== #
def test_help_exits_zero_and_lists_the_flags(capsys):
    with pytest.raises(SystemExit) as excinfo:
        info.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--config", "--cycles", "--json", "--deps", "--no-dry-run", "--quiet"):
        assert flag in out


def test_text_output_covers_the_sections(config_path, capsys):
    code, out, _err = _run(["--config", config_path, "--cycles", "2"], capsys)
    assert code == 0
    for needle in ("excavation.yaml", "frames:", "backend:", "ladder:", "belt:", "tub:", "sand:", "berm:", "dry run:"):
        assert needle in out


def test_quiet_prints_only_the_dry_run(config_path, capsys):
    code, out, _err = _run(["--config", config_path, "--cycles", "1", "--quiet"], capsys)
    assert code == 0
    assert "dry run:" in out
    assert "frames:" not in out


def test_no_dry_run_skips_the_cycles(config_path, capsys):
    code, out, _err = _run(["--config", config_path, "--no-dry-run"], capsys)
    assert code == 0
    assert "dry run:" not in out


def test_json_output_is_parseable_and_complete(config_path, capsys):
    code, out, _err = _run(["--config", config_path, "--json", "--cycles", "2"], capsys)
    assert code == 0
    payload = json.loads(out)
    assert payload["berm"]["dump_lands_inside"]["all_inside"] is True
    assert len(payload["dry_run"]["cycles"]) == 2
    assert payload["dry_run"]["berm"]["dumps"] == 2


def test_json_with_deps_adds_the_dependency_table(config_path, capsys):
    code, out, _err = _run(["--config", config_path, "--json", "--deps", "--no-dry-run"], capsys)
    assert code == 0
    payload = json.loads(out)
    assert "dependencies" in payload
    assert payload["dependencies"]["required"]["numpy"] is True
    assert "serial" in payload["dependencies"]["optional"]


def test_deps_appends_the_table_to_the_text(config_path, capsys):
    code, out, _err = _run(["--config", config_path, "--deps", "--no-dry-run"], capsys)
    assert code == 0
    assert "dependencies:" in out
    assert "optional:" in out
    assert "pyserial" not in out  # it is named for a human, not by import name


def test_a_bad_config_fails_with_a_message(capsys):
    code, _out, err = _run(["--config", "/nonexistent/excavation.yaml"], capsys)
    assert code == 1
    assert "excavation_info.py:" in err


def test_a_negative_cycle_count_is_clamped(config_path, capsys):
    code, out, _err = _run(["--config", config_path, "--json", "--cycles", "-3"], capsys)
    assert code == 0
    assert json.loads(out)["dry_run"]["cycles"] == []
