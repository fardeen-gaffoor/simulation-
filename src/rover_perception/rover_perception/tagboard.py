"""Turn the tag catalogue into textured boards that can be spawned into a sim.

Module 1's marker boards are painted flat: a light 25 cm board with a dark 18 cm
square on it, which is exactly right for the *compliance* geometry (markers are
visual-only, never collide, and stay inside the competition limits) but says
nothing to an AprilTag detector.  This module supplies the missing picture.

It deliberately does **not** touch ``cat_arena`` or its world file:

* the boards are rebuilt as separate, static **thin box** visuals, one per tag,
  offset in front of module 1's board by ``tags.board.standoff_m`` (which is
  recorded in ``tags.yaml``, so the tag's pose is never guessed),
* they are spawned into the running world with ``ros_gz_sim create`` - the same
  mechanism module 1 uses for the rover - so a world without module 4 keeps
  working unchanged and module 4 can be added to any already-generated arena,
* a **box** is used, not a ``<plane>``: ``gz-sim``'s plane primitive carries no
  texture coordinates, so a plane renders a textured tag as one flat colour and
  no tag ever decodes (measured on Harmonic 8.15; see ``docs/tags.md``).  A
  2 mm slab does generate UVs, and its face is placed exactly on the catalogue
  plane so the half-thickness costs no pose offset.

``scripts/generate_tags.py`` writes the PNGs and these SDFs;
``perception.launch.py`` spawns them.  See docs/tags.md.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import TagBoardCfg
from .tags import TagCatalogue, TagSpec, png_name
from .tagimages import render, write_png

__all__ = [
    "TagBoardError",
    "TagBoard",
    "model_name",
    "board_sdf",
    "build_plan",
    "plan_document",
    "spawn_arguments",
    "file_uri",
]


class TagBoardError(ValueError):
    """Raised when a tag board cannot be built."""


def file_uri(path: str) -> str:
    """``file://`` URI for an absolute path (what gz's resource loader wants)."""
    return "file://" + os.path.abspath(path)


def model_name(tag_id: int) -> str:
    """Stable gz model name, so a duplicate spawn is detectable by name."""
    return f"tag_board_{int(tag_id)}"


#: Thickness of the rendered tag slab, in metres.  2 mm is invisible at any
#: competition viewing distance and keeps the visible face clear of the board.
_BOARD_THICKNESS_M = 0.002

#: Visual pose (rpy, radians) of the textured face inside the tag model.
#:
#: The model frame is AprilTag's frame: ``+z`` into the board, ``+x`` the
#: viewer's right, ``+y`` the viewer's down.  A ``<box>`` maps its ``+z`` face's
#: texture ``u`` along local ``-y`` and ``v`` along local ``-x`` (measured on
#: Harmonic 8.15 with a four-quadrant texture), so composing the required
#: "``u`` to the viewer's right, ``v`` down" with that mapping gives
#: ``(roll pi, pitch 0, yaw -pi/2)``.  This exact value is what makes the tags in
#: ``~/cat`` decode in Gazebo Harmonic (OGRE2); it is asserted by
#: ``test_tagboard.py`` and re-verified against the running simulator by
#: ``docs/tags.md``'s in-sim check.
_TAG_FACE_RPY: Tuple[float, float, float] = (math.pi, 0.0, -math.pi / 2.0)

#: Where the slab centre sits in the model frame so the *visible* ``+z`` face -
#: half a thickness toward the viewer - lands exactly on the catalogue plane.
_TAG_FACE_POSITION: Tuple[float, float, float] = (0.0, 0.0, _BOARD_THICKNESS_M / 2.0)


@dataclass(frozen=True)
class TagBoard:
    """One rendered tag board: its texture, its SDF and its world pose."""

    tag: TagSpec
    name: str
    texture_path: str
    sdf_path: str
    size_m: float

    @property
    def position(self) -> Tuple[float, float, float]:
        return self.tag.position

    @property
    def rpy(self) -> Tuple[float, float, float]:
        return self.tag.rpy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.tag.id,
            "name": self.name,
            "size_m": round(self.size_m, 6),
            "texture": self.texture_path,
            "sdf": self.sdf_path,
            "position": [round(v, 6) for v in self.position],
            "rpy": [round(v, 9) for v in self.rpy],
            "normal": [round(v, 9) for v in self.tag.normal],
        }


def board_sdf(tag: TagSpec, texture_uri: str, size_m: float) -> str:
    """A one-visual static model: a thin textured slab showing the tag's face.

    The **model** carries no pose of its own: the world pose comes from
    ``ros_gz_sim create -x -y -z -R -P -Y``, which keeps exactly one authority
    over where the tag is and means the SDF text is identical for every tag id.
    The model's frame is the *tag* frame from ``tags.yaml`` (AprilTag's frame,
    ``+z`` into the board), so the visual carries ``_TAG_FACE_RPY`` and
    ``_TAG_FACE_POSITION``: the flip points the textured face out of the wall
    (visible from the arena) without mirroring the bitmap, and the position puts
    that face exactly on the catalogue plane.
    """
    if size_m <= 0.0:
        raise TagBoardError(f"tag {tag.id}: size_m must be > 0, got {size_m}")
    px, py, pz = _TAG_FACE_POSITION
    roll, pitch, yaw = _TAG_FACE_RPY
    return (
        '<?xml version="1.0" ?>\n'
        '<!-- rover_perception: rendered AprilTag board, spawned at runtime.\n'
        f'     tag id {tag.id} ({tag.family}), black square {tag.size_m:.4f} m,\n'
        f'     bitmap {size_m:.6f} m square including its quiet zone,\n'
        f'     mount: {tag.mount}.  This model has no <pose>: the spawn command owns it.\n'
        '     A thin <box> is used because gz-sim\'s <plane> has no texture\n'
        '     coordinates; the visual is posed so the textured face points out of\n'
        '     the wall, un-mirrored, exactly on the tag plane (see tagboard.py). -->\n'
        '<sdf version="1.8">\n'
        f'  <model name="{model_name(tag.id)}">\n'
        "    <static>true</static>\n"
        '    <link name="link">\n'
        '      <visual name="tag">\n'
        f"        <pose>{px:.9f} {py:.9f} {pz:.9f} {roll:.9f} {pitch:.9f} {yaw:.9f}</pose>\n"
        "        <geometry>\n"
        "          <box>\n"
        f"            <size>{size_m:.6f} {size_m:.6f} {_BOARD_THICKNESS_M:.6f}</size>\n"
        "          </box>\n"
        "        </geometry>\n"
        "        <material>\n"
        "          <ambient>0.9 0.9 0.9 1</ambient>\n"
        "          <diffuse>1.0 1.0 1.0 1</diffuse>\n"
        "          <specular>0.04 0.04 0.04 1</specular>\n"
        "          <pbr>\n"
        "            <metal>\n"
        f"              <albedo_map>{texture_uri}</albedo_map>\n"
        "              <roughness>0.65</roughness>\n"
        "              <metalness>0.0</metalness>\n"
        "            </metal>\n"
        "          </pbr>\n"
        "        </material>\n"
        "      </visual>\n"
        "    </link>\n"
        "  </model>\n"
        "</sdf>\n"
    )


def build_plan(
    catalogue: TagCatalogue,
    cfg: TagBoardCfg,
    out_dir: str,
    write: bool = True,
) -> List[TagBoard]:
    """Render every tag's texture and SDF into ``out_dir``.

    Returns the boards in catalogue order.  With ``write=False`` nothing touches
    the disk (the tests use that to check the plan without side effects).
    """
    if not cfg.enabled:
        return []
    directory = os.path.abspath(os.path.expanduser(out_dir))
    boards: List[TagBoard] = []
    for tag_id in catalogue.ids:
        tag = catalogue.tag(tag_id)
        rendered = render(
            tag_id,
            tag.family,
            tag.size_m,
            cfg.texture_size_px,
            cfg.quiet_zone_modules,
        )
        texture_path = os.path.join(directory, png_name(tag_id, tag.family))
        sdf_path = os.path.join(directory, f"{model_name(tag_id)}.sdf")
        if write:
            os.makedirs(directory, exist_ok=True)
            write_png(texture_path, rendered.array)
            with open(sdf_path, "w", encoding="utf-8") as handle:
                handle.write(board_sdf(tag, file_uri(texture_path), rendered.drawing_size_m))
        boards.append(
            TagBoard(
                tag=tag,
                name=model_name(tag_id),
                texture_path=texture_path,
                sdf_path=sdf_path,
                size_m=rendered.drawing_size_m,
            )
        )
    return boards


def spawn_arguments(
    board: TagBoard,
    world: str,
    executable: str = "ros2",
    ros_gz_sim: str = "ros_gz_sim",
) -> List[str]:
    """The ``ros_gz_sim create`` command line for one board.

    The pose is passed explicitly (rather than baked into the SDF) so the SDF is
    identical for all tags and the spawn log shows where each tag went.
    """
    x, y, z = board.position
    roll, pitch, yaw = board.rpy
    return [
        executable,
        "run",
        ros_gz_sim,
        "create",
        "-world",
        world,
        "-file",
        board.sdf_path,
        "-name",
        board.name,
        "-x",
        f"{x:.6f}",
        "-y",
        f"{y:.6f}",
        "-z",
        f"{z:.6f}",
        "-R",
        f"{roll:.9f}",
        "-P",
        f"{pitch:.9f}",
        "-Y",
        f"{yaw:.9f}",
    ]


def plan_document(boards: Sequence[TagBoard], out_dir: str, world: str = "cat_arena") -> Dict[str, Any]:
    """The report ``generate_tags.py`` prints and writes next to the textures."""
    return {
        "schema_version": 1,
        "output_dir": os.path.abspath(os.path.expanduser(out_dir)),
        "world": world,
        "count": len(boards),
        "boards": [board.to_dict() for board in boards],
        "spawn_commands": [" ".join(spawn_arguments(board, world)) for board in boards],
    }


def write_plan(path: str, document: Dict[str, Any]) -> str:
    """Write the spawn plan as YAML (a human-auditable record of a run)."""
    import yaml

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# rover_perception tag board spawn plan - generated\n")
        yaml.safe_dump(document, handle, sort_keys=False, default_flow_style=False)
    return path
