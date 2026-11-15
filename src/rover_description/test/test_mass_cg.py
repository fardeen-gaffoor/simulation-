"""Tests for the Stage 1 mass budget with CG and inertia (mass_cg.py)."""

from __future__ import annotations

import pytest

from rover_description.mass_cg import (
    InertiaTensor,
    MassBudgetResult,
    compute_mass_budget,
    load_all_configs,
)


@pytest.fixture
def configs(tmp_path):
    import os
    import shutil

    here = os.path.dirname(__file__)
    src_root = os.path.join(here, "..", "config")
    dst = tmp_path / "cfg"
    dst.mkdir()
    for name in ("robot_params.yaml", "motors.yaml", "battery.yaml", "cameras.yaml"):
        shutil.copy(os.path.join(src_root, name), dst / name)
    return load_all_configs(
        robot_path=str(dst / "robot_params.yaml"),
        motors_path=str(dst / "motors.yaml"),
        battery_path=str(dst / "battery.yaml"),
        cameras_path=str(dst / "cameras.yaml"),
    )


def test_mass_budget_totals(configs):
    robot, motors, battery, cameras = configs
    budget = compute_mass_budget(robot, motors, battery, cameras)
    assert budget.structural_mass_kg > 0.0
    assert budget.payload_mass_kg > 0.0
    assert budget.total_mass_kg == pytest.approx(budget.structural_mass_kg + budget.payload_mass_kg)


def test_mass_within_50kg_target(configs):
    robot, motors, battery, cameras = configs
    budget = compute_mass_budget(robot, motors, battery, cameras)
    assert budget.ok_target, f"total {budget.total_mass_kg:.2f} kg exceeds 50 kg target"


def test_mass_within_80kg_ceiling(configs):
    robot, motors, battery, cameras = configs
    budget = compute_mass_budget(robot, motors, battery, cameras)
    assert budget.ok_ceiling, f"total {budget.total_mass_kg:.2f} kg exceeds 80 kg ceiling"


def test_cg_is_finite(configs):
    robot, motors, battery, cameras = configs
    budget = compute_mass_budget(robot, motors, battery, cameras)
    cg = budget.cg
    assert abs(cg.x) < 1.0
    assert abs(cg.y) < 1.0
    assert 0.0 < cg.z < 1.0


def test_inertia_positive(configs):
    robot, motors, battery, cameras = configs
    budget = compute_mass_budget(robot, motors, battery, cameras)
    it = budget.inertia_about_cg
    assert it.ixx > 0.0
    assert it.iyy > 0.0
    assert it.izz > 0.0
    # triangle inequality
    assert it.ixx + it.iyy >= it.izz - 1e-9


def test_weight_splits(configs):
    robot, motors, battery, cameras = configs
    budget = compute_mass_budget(robot, motors, battery, cameras)
    assert len(budget.weight_splits) == 3
    for s in budget.weight_splits:
        assert abs(s["front_fraction"] + s["rear_fraction"] - 1.0) < 1e-9
        assert abs(s["front_mass_kg"] + s["rear_mass_kg"] - budget.total_mass_kg) < 1e-9


def test_table_prints(configs):
    robot, motors, battery, cameras = configs
    budget = compute_mass_budget(robot, motors, battery, cameras)
    text = budget.table()
    assert "TOTAL" in text
    assert "50 kg" in text or "design target" in text
    assert "80 kg" in text or "competition ceiling" in text


def test_cg_table_prints(configs):
    robot, motors, battery, cameras = configs
    budget = compute_mass_budget(robot, motors, battery, cameras)
    text = budget.cg_table()
    assert "Centre of gravity" in text
    assert "x =" in text


def test_inertia_table_prints(configs):
    robot, motors, battery, cameras = configs
    budget = compute_mass_budget(robot, motors, battery, cameras)
    text = budget.inertia_table()
    assert "Inertia tensor" in text
    assert "principal moments" in text
