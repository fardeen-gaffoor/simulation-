"""Tests for the stowed-envelope check and its CLI script."""

from __future__ import annotations

import pytest

from rover_description.envelope import check_envelope


def test_report_ok(cfg):
    report = check_envelope(cfg)
    assert report.ok
    assert report.measured_envelope.as_tuple() == pytest.approx(report.measured.as_tuple())


def test_margins_positive(cfg):
    report = check_envelope(cfg)
    for value in report.margins().values():
        assert value >= 0.0


def test_table_reports_pass(cfg):
    text = check_envelope(cfg).table()
    assert "PASS" in text
    assert "length_x" in text and "height_z" in text


def test_oversized_fails(make_config):
    from rover_description.config import load_config

    def mutate(doc):
        doc["chassis"]["length"] = 1.7  # longer than the 1.5 m limit

    report = check_envelope(load_config(make_config(mutate)))
    assert not report.ok


def test_tall_fails(make_config):
    from rover_description.config import load_config

    def mutate(doc):
        doc["excavation"]["funnel"]["center"] = [-0.15, 0.0, 0.80]

    report = check_envelope(load_config(make_config(mutate)))
    assert not report.ok


def test_script_exit_zero(cfg, capsys):
    from loaders import load_script

    module = load_script("check_envelope")
    assert module.main([]) == 0
    out = capsys.readouterr().out
    assert "PASS" in out


def test_script_json(capsys):
    from loaders import load_script

    module = load_script("check_envelope")
    assert module.main(["--json"]) == 0
    out = capsys.readouterr().out
    assert '"envelope"' in out and '"ok": true' in out
