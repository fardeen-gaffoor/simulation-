"""Tests for the mass budget table."""

from __future__ import annotations

import pytest

from rover_description.mass import mass_budget


def test_totals_match_config(cfg):
    budget = mass_budget(cfg)
    assert budget.structural_mass == pytest.approx(cfg.structural_mass)
    assert budget.payload_mass == pytest.approx(cfg.payload_mass)
    assert budget.total_mass == pytest.approx(cfg.total_mass)


def test_within_limit(cfg):
    budget = mass_budget(cfg)
    assert budget.ok
    assert budget.margin > 0.0


def test_table_shape(cfg):
    text = mass_budget(cfg).table()
    assert "TOTAL" in text
    assert "limit" in text
    assert "margin" in text


def test_rows_have_categories(cfg):
    budget = mass_budget(cfg)
    assert any(r.category == "structural" for r in budget.rows)
    assert any(r.category == "payload" for r in budget.rows)
    assert all(r.mass > 0.0 for r in budget.rows)


def test_battery_in_payload(cfg):
    names = [r.name for r in mass_budget(cfg).rows]
    assert any("battery" in n for n in names)


def test_script_exit_zero(capsys):
    from loaders import load_script

    module = load_script("mass_budget")
    assert module.main([]) == 0
    assert "TOTAL" in capsys.readouterr().out
