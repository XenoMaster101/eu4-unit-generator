"""Test 11 pre-check (Blender only): render a posed unit from a staged test mod: the derived body
(textured with the base game's diffuse, read in place) playing a vanilla .anim, with its attached
parts (part .mesh files with their DDS beside them) following their bones. Views: side, rear,
rear-quarter close-ups and map-scale (48 px) front- and rear-quarter shots from above.
Writes only to out_dir.

args: {"body", "diffuse", "parts": [{"mesh", "bone", "bind": [12]}], "anim", "frames": [..],
       "out_dir", "tag", "size", "map_px", "result_path"}
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src", "eu4unitgen", "blender", "jobs"))

import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import _pdx  # noqa: E402
import check_deform as cd  # noqa: E402

S = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))  # PDX <-> Blender axis swap
VIEWS = {"side": (-1, 0, 0), "rear": (0, -1, 0), "rear_quarter": (0.6, -0.7, -0.35)}
MAP = {"map_front_quarter": (-0.45, 0.55, -0.7), "map_rear_quarter": (0.45, -0.55, -0.7)}  # map camera UNVERIFIED


def main():
    a = _pdx.args()
    ie = _pdx.io_module()
    _pdx.clear_scene()
    rig, body, new = cd.import_body(ie, a["body"])
    sc, cam = cd.setup_render({"renders": {"diffuse": a["diffuse"]}}, body, new)
    parts = []
    for p in a["parts"]:
        before = set(bpy.data.objects)
        ie.import_meshfile(p["mesh"])
        objs = [o for o in bpy.data.objects if o not in before]
        bind = Matrix([p["bind"][0:4], p["bind"][4:8], p["bind"][8:12], [0, 0, 0, 1]])
        rest_world = S @ bind @ S
        bone_rest = rig.matrix_world @ rig.data.bones[p["bone"]].matrix_local
        for o in objs:
            if o.parent is None:
                parts.append((o, p["bone"], bone_rest.inverted() @ rest_world))
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    cd.import_anim(ie, rig, a["anim"])
    out = {}
    for f in a["frames"]:
        sc.frame_set(f)
        for o, bone, offset in parts:
            o.matrix_world = rig.matrix_world @ rig.pose.bones[bone].matrix @ offset
        bpy.context.view_layer.update()
        co = cd.evaluated(body)
        for o, _, _ in parts:
            co += [o.matrix_world @ Vector(c) for c in o.bound_box]
        for view, d in VIEWS.items():
            out[f"f{f}_{view}"] = cd.shoot(sc, cam, co, d, os.path.join(a["out_dir"], f"{a['tag']}_f{f}_{view}.png"),
                                           a.get("size", 320))
        for view, d in MAP.items():
            out[f"f{f}_{view}"] = cd.shoot(sc, cam, co, d, os.path.join(a["out_dir"], f"{a['tag']}_f{f}_{view}.png"),
                                           a.get("map_px", 48))
    _pdx.write_result(a, out)


main()
