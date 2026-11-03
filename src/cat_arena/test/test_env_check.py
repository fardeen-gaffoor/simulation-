"""Tests for the environment checker.

``env_check.py`` is the first thing anyone runs on a new machine, so it has to
be correct in two directions: it must not cry wolf on a working laptop, and it
must actually fail when something that module 1 needs is broken.

These tests use ``--no-smoke`` (no simulator is started) so they stay fast and
side-effect free; the live smoke tests are run manually via the CLI and are
covered by the README checklist.
"""

from __future__ import annotations

import json
import shutil

import pytest

from scripts_loader import load_script

env_check = load_script("env_check")


# --------------------------------------------------------------------------- #
# Report bookkeeping (pure unit tests)
# --------------------------------------------------------------------------- #
def test_report_counts_and_ok():
    report = env_check.Report()
    report.add("s", "a", env_check.PASS)
    report.add("s", "b", env_check.WARN, "detail")
    assert report.counts() == {
        env_check.PASS: 1,
        env_check.WARN: 1,
        env_check.FAIL: 0,
        env_check.SKIP: 0,
    }
    assert report.ok() is True
    report.add("s", "c", env_check.FAIL, "boom")
    assert report.ok() is False


def test_report_renders_sections_and_summary():
    report = env_check.Report()
    report.add("platform", "python", env_check.PASS, "3.10.12")
    report.add("gazebo (Harmonic)", "gz sim version", env_check.PASS, "8.15.0")
    text = report.render(use_color=False)
    assert "== platform ==" in text
    assert "== gazebo (Harmonic) ==" in text
    assert "[PASS] python" in text
    assert "summary:" in text


def test_report_json_shape():
    report = env_check.Report()
    report.add("s", "a", env_check.PASS, "d")
    payload = json.loads(
        json.dumps({"ok": report.ok(), "counts": report.counts(),
                    "results": [r.as_dict() for r in report.results]})
    )
    assert payload["ok"] is True
    assert payload["results"][0] == {"section": "s", "name": "a", "status": "PASS", "detail": "d"}


def test_arg_parser_defaults():
    args = env_check.build_arg_parser().parse_args([])
    assert args.no_smoke is False
    assert args.render is False
    assert args.arena is False
    assert args.json is False


# --------------------------------------------------------------------------- #
# static checks against this machine
# --------------------------------------------------------------------------- #
def test_static_checks_find_no_failures(capsys):
    """Nothing module 1 needs may be reported as FAIL on this machine."""
    rc = env_check.main(["--no-smoke", "--json"])
    payload = json.loads(capsys.readouterr().out)
    failures = [r for r in payload["results"] if r["status"] == env_check.FAIL]
    assert failures == [], "environment check reported failures:\n" + "\n".join(
        f"  {r['name']}: {r['detail']}" for r in failures
    )
    assert rc == 0


def test_static_report_mentions_the_essentials(capsys):
    env_check.main(["--no-smoke", "--json"])
    payload = json.loads(capsys.readouterr().out)
    names = " | ".join(r["name"] for r in payload["results"])
    for required in ("ROS_DISTRO", "gz sim version", "DART bullet collision"):
        assert required in names


def test_platform_section_matches_this_box(capsys):
    import platform
    import sys

    env_check.main(["--no-smoke", "--json"])
    payload = json.loads(capsys.readouterr().out)
    results = {r["name"]: r for r in payload["results"]}
    assert results["python"]["detail"] == sys.version.split()[0]
    assert results["python executable"]["detail"] == sys.executable
    assert platform.machine() in results["os"]["detail"]


@pytest.mark.skipif(shutil.which("gz") is None, reason="gz CLI not available")
def test_gazebo_details_are_reported(capsys):
    env_check.main(["--no-smoke", "--json"])
    payload = json.loads(capsys.readouterr().out)
    results = {r["name"]: r for r in payload["results"]}
    assert results["gz sim version"]["status"] == env_check.PASS
    assert "8.15" in results["gz sim version"]["detail"] or "version 8." in results["gz sim version"]["detail"]


def test_bridge_compatibility_section_is_present(capsys):
    env_check.main(["--no-smoke", "--json"])
    payload = json.loads(capsys.readouterr().out)
    sections = {r["section"] for r in payload["results"]}
    assert "ros_gz bridge compatibility" in sections
    assert "ros packages" in sections


def test_install_hints_lists_apt_packages(capsys):
    env_check.main(["--no-smoke", "--json", "--install-hints"])
    payload = json.loads(capsys.readouterr().out)
    hints = [r for r in payload["results"] if r["name"] == "install hint"]
    assert hints, "expected at least one missing package to produce a hint"
    assert hints[0]["detail"].startswith("sudo apt install -y")


def test_skips_are_reported_when_smoke_is_disabled(capsys):
    env_check.main(["--no-smoke", "--json"])
    payload = json.loads(capsys.readouterr().out)
    skipped = {r["name"] for r in payload["results"] if r["status"] == env_check.SKIP}
    assert "gz sim boot + world load" in skipped
    assert "/clock bridge" in skipped
    assert "arena generation" in skipped


# --------------------------------------------------------------------------- #
# the arena self-check
# --------------------------------------------------------------------------- #
def test_arena_check_generates_and_validates(capsys):
    env_check.main(["--no-smoke", "--arena", "--json"])
    payload = json.loads(capsys.readouterr().out)
    results = {r["name"]: r for r in payload["results"]}
    assert results["arena.yaml"]["status"] == env_check.PASS
    assert results["generate seed 1"]["status"] == env_check.PASS
    assert results["zones partition the arena"]["status"] == env_check.PASS
    assert results["assumptions flagged"]["status"] == env_check.WARN
    if shutil.which("gz"):
        assert results["gz sdf -k"]["status"] == env_check.PASS
        assert "OK" in results["gz sdf -k"]["detail"]


def test_find_library_locates_a_real_library():
    found = env_check._find_library("libgz-transport")
    if shutil.which("gz") is None:
        assert found is None
    else:
        assert found and found.endswith(".so") or ".so." in found
