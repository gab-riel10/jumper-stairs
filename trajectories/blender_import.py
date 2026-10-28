"""Import one recorded jump into Blender: the robot's meshes, keyframed.

Run inside Blender (4.x), from this repository's root or with `REPO` set:

    blender --python trajectories/blender_import.py -- \\
        trajectories/curb15_alpha0.3/jump_01.npz

or paste into Blender's Text Editor, set `NPZ` below and press Run Script.

What it builds:
- one Empty per robot body (`body:<name>`), keyframed every scene frame from
  the recording's `body_pos` / `body_quat` (resampled from 200 Hz to the scene's
  fps by picking the nearest sample -- no interpolation is invented);
- each visible part as the STL of this repository, parented to its body's
  Empty with the pose in `model.json` (`file_pos`, `file_quat_wxyz`), coloured
  with its `rgba`;
- optionally (`BUILD_KERB`) two boxes for the street and the kerb, the sizes of
  the simulated ones, so the scene can be checked against the physics.

Coordinates: metres; +x up the kerb, +z up. Everything is shifted so the
kerb's riser is the plane x = 0 and the street is z = 0 (`model.json` holds the
simulator's own numbers if they are wanted).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import bpy
import numpy as np

NPZ = ""  # e.g. "trajectories/curb15_alpha0.3/jump_01.npz"; the command line wins
REPO = Path(os.environ.get("REPO", Path.cwd()))
BUILD_KERB = True


def _arg_npz() -> Path:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    path = Path(argv[0] if argv else NPZ)
    if not path.is_absolute():
        path = REPO / path
    return path


def _import_stl(path: Path):
    before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, "stl_import"):  # Blender 4.1+
        bpy.ops.wm.stl_import(filepath=str(path))
    else:  # older: the legacy add-on
        bpy.ops.import_mesh.stl(filepath=str(path))
    new = [o for o in bpy.data.objects if o not in before]
    return new[0]


def _material(rgba):
    key = "jumper_" + "_".join(f"{c:.2f}" for c in rgba)
    mat = bpy.data.materials.get(key)
    if mat is None:
        mat = bpy.data.materials.new(key)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = tuple(rgba)
            bsdf.inputs["Roughness"].default_value = 0.5
        mat.diffuse_color = tuple(rgba)
    return mat


def main() -> None:
    npz = _arg_npz()
    folder = npz.parent
    scene_json = json.loads((folder / "model.json").read_text())
    data = np.load(npz)
    body_pos, body_quat, t = data["body_pos"], data["body_quat"], data["time"]
    kerb = scene_json["kerb"]
    shift = np.array([kerb["riser_x"], kerb["center_y"], kerb["ground_z"]])

    root = bpy.data.objects.new(f"jumper:{npz.stem}", None)
    bpy.context.scene.collection.objects.link(root)

    empties = {}
    for name in scene_json["body_names"]:
        e = bpy.data.objects.new(f"body:{name}", None)
        e.empty_display_size = 0.01
        e.rotation_mode = "QUATERNION"
        e.parent = root
        bpy.context.scene.collection.objects.link(e)
        empties[name] = e

    for g in scene_json["robot"]["geoms"]:
        obj = _import_stl(REPO / g["file"])
        obj.name = g["name"]
        obj.parent = empties[g["body"]]
        obj.rotation_mode = "QUATERNION"
        obj.location = g["file_pos"]
        obj.rotation_quaternion = g["file_quat_wxyz"]
        obj.scale = g["scale"]
        obj.data.materials.clear()
        obj.data.materials.append(_material(g["rgba"]))

    scene = bpy.context.scene
    fps = scene.render.fps / scene.render.fps_base
    frames = int(np.floor(t[-1] * fps)) + 1
    for f in range(frames):
        i = int(np.argmin(np.abs(t - f / fps)))
        for b, name in enumerate(scene_json["body_names"]):
            e = empties[name]
            e.location = (body_pos[i, b] - shift).tolist()
            e.rotation_quaternion = body_quat[i, b].tolist()  # w x y z, Blender's order too
            e.keyframe_insert("location", frame=f + 1)
            e.keyframe_insert("rotation_quaternion", frame=f + 1)
    scene.frame_start, scene.frame_end = 1, frames

    if BUILD_KERB:
        h, top, land, width = kerb["height_m"], kerb["top_depth_m"], kerb["landing_depth_m"], kerb["width_m"]
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-land / 2, 0.0, -0.05))
        street = bpy.context.active_object
        street.name, street.scale = "sim:street", (land, width, 0.10)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(top / 2, 0.0, h / 2))
        curb = bpy.context.active_object
        curb.name, curb.scale = "sim:kerb", (top, width, h)

    print(f"[jumper] {npz.name}: {len(empties)} bodies, {len(scene_json['robot']['geoms'])} meshes, "
          f"{frames} frames at {fps:g} fps")


main()
