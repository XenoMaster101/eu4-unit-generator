"""Blender job: skinned .mesh files posed at the same sampled frames of each animation (the animation review: ours
next to vanilla's body at identical frames). Writes one .npz per mesh: ``tri`` (T, 3), ``rest`` (V, 3), ``bone``
(V,) the index of each vertex's dominant bone into ``bones``, and per animation ``<anim>`` (F, V, 3) posed positions,
``<anim>_frames`` (F,) and ``<anim>_joints`` (F, B, 3) every bone's head; positions in Blender's world (Z up).

args: {"meshes": [{"name", "path"}], "anims": [{"id", "path"}], "samples": 8, "out_dir", "result_path"}
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402 - Blender module
import numpy as np  # noqa: E402 - Blender's numpy

import _pdx  # noqa: E402
from check_mesh_deform import coords, frame_range, import_anim, import_mesh  # noqa: E402


def main():
    a = _pdx.args()
    ie = _pdx.io_module()
    scene = bpy.context.scene
    n = int(a.get("samples", 8))
    res = {}
    for m in a["meshes"]:
        out = {}
        bones = None
        for anim in a["anims"]:
            _pdx.clear_scene()
            rig, mesh = import_mesh(ie, m["path"])
            if bones is None:
                bones = [b.name for b in rig.data.bones]
                out["tri"] = np.array([tuple(p.vertices) for p in mesh.data.polygons], dtype=np.int32)
                out["rest"] = np.array([tuple(c) for c in coords(mesh)], dtype=np.float32)
                gname = {g.index: g.name for g in mesh.vertex_groups}
                dom = []
                for v in mesh.data.vertices:
                    best = max(v.groups, key=lambda g: g.weight, default=None)
                    dom.append(bones.index(gname[best.group]) if best and gname[best.group] in bones else -1)
                out["bone"] = np.array(dom, dtype=np.int32)
            import_anim(ie, rig, anim["path"])
            f0, f1 = frame_range(rig)
            frames = sorted({int(round(f)) for f in np.linspace(f0, f1, n)})
            pos, joints = [], []
            for f in frames:
                scene.frame_set(f)
                pos.append([tuple(c) for c in coords(mesh)])
                joints.append([tuple(rig.matrix_world @ rig.pose.bones[b].head) if b in rig.pose.bones else (0, 0, 0)
                               for b in bones])
            out[anim["id"]] = np.array(pos, dtype=np.float32)
            out[anim["id"] + "_frames"] = np.array(frames, dtype=np.int32)
            out[anim["id"] + "_joints"] = np.array(joints, dtype=np.float32)
        out["bones"] = np.array(bones)
        path = os.path.join(a["out_dir"], f"{m['name']}.npz")
        np.savez_compressed(path, **out)
        res[m["name"]] = path
    _pdx.write_result(a, res)


main()
