"""The ``ros2 run`` entry points: the tools must work and the shims must not.

Two kinds of file live in ``scripts/``:

* the three **tools** (``generate_tags``, ``preview_tags``, ``perception_info``)
  whose logic is genuinely in the script, so they are driven end to end here, and
* the five **node shims**, which are three lines that forward to the node module.
  A shim cannot be run to completion in a test (it would spin), so the test only
  pins what a shim can get wrong: that it exposes ``main`` and that ``--help``
  reaches argparse before rclpy is ever initialised.
"""

from __future__ import annotations

import json
import os

import pytest

from rover_perception import tags as taglib
from scripts_loader import load_script

generate_tags = load_script("generate_tags")
preview_tags = load_script("preview_tags")
perception_info = load_script("perception_info")

_SHIMS = (
    "camera_manager",
    "tag_detector",
    "tag_localizer",
    "obstacle_detector",
    "health_monitor",
)


# --------------------------------------------------------------------------- #
# every script is loadable
# --------------------------------------------------------------------------- #
def test_every_script_exposes_main():
    for name in ("generate_tags", "preview_tags", "perception_info", *_SHIMS):
        module = load_script(name)
        assert callable(module.main), name


# --------------------------------------------------------------------------- #
# generate_tags
# --------------------------------------------------------------------------- #
def test_generate_tags_writes_the_whole_authoring_set(cfg, tmp_path):
    out = str(tmp_path / "tags")
    assert generate_tags.main(["--config", cfg.path, "--out-dir", out]) == 0
    assert os.path.isfile(os.path.join(out, "tags.yaml"))
    assert os.path.isfile(os.path.join(out, "tag_boards.yaml"))
    catalogue = taglib.load_tags(os.path.join(out, "tags.yaml"))
    assert sorted(catalogue.ids) == [0, 1, 2]
    for tag_id in catalogue.ids:
        assert os.path.isfile(os.path.join(out, f"tag_{tag_id}_tag36h11.png"))
        assert os.path.isfile(os.path.join(out, f"tag_board_{tag_id}.sdf"))


def test_generate_tags_json_is_parseable(cfg, tmp_path, capsys):
    out = str(tmp_path / "tags")
    assert generate_tags.main(["--config", cfg.path, "--out-dir", out, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 3
    assert payload["family"] == cfg.tags.family
    assert payload["markers_seed"] == 0 or isinstance(payload["markers_seed"], int)
    assert sorted(tag["id"] for tag in payload["tags"]) == [0, 1, 2]
    assert len(payload["boards"]) == 3


def test_generate_tags_print_paths_is_one_path_per_line(cfg, tmp_path, capsys):
    out = str(tmp_path / "tags")
    assert generate_tags.main(["--config", cfg.path, "--out-dir", out, "--print-paths"]) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert len(lines) == 8, "tags.yaml + plan + 3 bitmaps + 3 SDFs"
    for line in lines:
        assert os.path.isfile(line), line


def test_generate_tags_no_textures_writes_only_the_catalogue(cfg, tmp_path):
    out = str(tmp_path / "tags")
    assert generate_tags.main(["--config", cfg.path, "--out-dir", out, "--no-textures", "--quiet"]) == 0
    assert os.path.isfile(os.path.join(out, "tags.yaml"))
    assert not os.path.isfile(os.path.join(out, "tag_boards.yaml"))
    assert not [name for name in os.listdir(out) if name.endswith(".png")]


def test_generate_tags_reports_a_missing_marker_file(cfg, tmp_path, capsys):
    code = generate_tags.main(
        [
            "--config",
            cfg.path,
            "--markers",
            "/nonexistent/markers.yaml",
            "--out-dir",
            str(tmp_path / "tags"),
        ]
    )
    assert code == 1
    assert "generate_tags:" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# preview_tags
# --------------------------------------------------------------------------- #
def test_preview_tags_draws_the_requested_ids(cfg, tmp_path):
    out = str(tmp_path / "preview.png")
    assert preview_tags.main(["--config", cfg.path, "--ids", "0,1,2", "--out", out]) == 0
    assert os.path.isfile(out)


def test_preview_tags_json_lists_the_tags(cfg, tmp_path, capsys):
    out = str(tmp_path / "preview.png")
    assert preview_tags.main(["--config", cfg.path, "--ids", "0,1", "--out", out, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ids"] == [0, 1]
    assert payload["image"] == os.path.abspath(out)
    assert len(payload["tags"]) == 2


def test_preview_tags_defaults_to_the_marker_ids(cfg, capsys):
    # markers.yaml is on disk (the conftest fixture), so no --ids is needed
    assert preview_tags.main(["--config", cfg.path, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ids"] == [0, 1, 2]


def test_preview_tags_print_paths(cfg, tmp_path, capsys):
    out = str(tmp_path / "preview.png")
    assert preview_tags.main(["--config", cfg.path, "--ids", "0", "--out", out, "--print-paths"]) == 0
    assert capsys.readouterr().out.strip() == os.path.abspath(out)


def test_preview_tags_id_parsing_tolerates_spaces():
    assert preview_tags._ids_from(" 0, 1 ,2 ") == [0, 1, 2]
    assert preview_tags._ids_from("") == []


# --------------------------------------------------------------------------- #
# perception_info
# --------------------------------------------------------------------------- #
def test_perception_info_json_reports_the_effective_config(cfg, catalogue, capsys):
    # catalogue guarantees the tags.yaml that perception_info inspects exists
    assert perception_info.main(["--config", cfg.path, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["path"] == cfg.path
    assert payload["config"]["sha256"] == cfg.sha256
    assert payload["detector"]["backend"] == cfg.detector.backend
    assert payload["cameras"][0]["topic"] == cfg.cameras[0].topic
    assert payload["artefacts"]["calibration"]["state"] == "present"
    assert payload["artefacts"]["tags"]["count"] == 3


def test_perception_info_json_deps_includes_the_dependency_table(cfg, capsys):
    assert perception_info.main(["--config", cfg.path, "--json-deps"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "dependencies" in payload
    assert any(row["name"] == "numpy" for row in payload["dependencies"]["dependencies"])


def test_perception_info_text_lists_every_section(cfg, capsys):
    assert perception_info.main(["--config", cfg.path]) == 0
    out = capsys.readouterr().out
    for heading in ("config", "arena", "cameras", "camera mgr", "detector", "tags", "localization", "artefacts"):
        assert heading in out
    assert f"zone start" in out or "zone construction" in out


def test_perception_info_deps_appends_the_table(cfg, capsys):
    assert perception_info.main(["--config", cfg.path, "--deps"]) == 0
    assert "dependencies" in capsys.readouterr().out


def test_perception_info_no_artefacts_still_loads_the_config(cfg, capsys):
    assert perception_info.main(["--config", cfg.path, "--no-artefacts", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    # the path is still resolved and reported, only the contents are skipped
    assert payload["artefacts"]["calibration"]["state"] == "present"
    assert "cameras" not in payload["artefacts"]["calibration"]


def test_perception_info_quiet_prints_nothing_when_ready(cfg, capsys):
    assert perception_info.main(["--config", cfg.path, "--quiet"]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""


def test_perception_info_rejects_a_bad_config(capsys):
    assert perception_info.main(["--config", "/nonexistent/perception.yaml"]) == 1
    assert "perception_info:" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# the node shims
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", _SHIMS)
def test_a_node_shim_forwards_to_the_node_main(name):
    from rover_perception.nodes import (
        camera_manager_node,
        health_monitor_node,
        obstacle_detector_node,
        tag_detector_node,
        tag_localizer_node,
    )

    expected = {
        "camera_manager": camera_manager_node,
        "health_monitor": health_monitor_node,
        "obstacle_detector": obstacle_detector_node,
        "tag_detector": tag_detector_node,
        "tag_localizer": tag_localizer_node,
    }[name]
    assert load_script(name).main is expected.main


@pytest.mark.parametrize("name", _SHIMS)
def test_help_exits_before_rclpy_starts(name, capsys):
    module = load_script(name)
    with pytest.raises(SystemExit) as excinfo:
        module.main(["--help"])
    assert excinfo.value.code == 0
    assert "--config" in capsys.readouterr().out
