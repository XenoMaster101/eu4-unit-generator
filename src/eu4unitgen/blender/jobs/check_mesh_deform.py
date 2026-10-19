"""Blender job: deformation check of a whole skinned .mesh under its animations (Track C, test 15:
fresh meshes on a vanilla skeleton; run on the vanilla mesh too, as the baseline).

For each animation: import the .mesh (io_pdx_mesh), import the .anim onto its rig (bones the
.anim names but the mesh lacks get unweighted dummies under the root bone: the game ignores
those tracks), and measure every frame over all triangle edges and faces:

  * stretch: edge length / rest length (min, max, p05, p95);
  * flips: triangles whose normal turns > 90 degrees from its rest normal carried by the
    triangle's dominant bone (the bone with the most weight over its three vertices);
  * the worst frame: the one whose edge furthest from its rest length (in log terms) is worst,
    and the dominant bones of that edge's ends.

Optional renders of each animation's worst frame (side and 3/4, vertices coloured by stretch:
grey = rest length, red = stretched or squashed by 50% or more).

args: {"mesh", "anims": [{"id", "path"}], "renders": {"out_dir", "name", "size"} | null, "result_path"}
"""
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402 - Blender module
from mathutils import Vector  # noqa: E402 - Blender module

import _pdx  # noqa: E402


def import_mesh(ie, path):
    before = set(bpy.data.objects)
    ie.import_meshfile(path)
    new = [o for o in bpy.data.objects if o not in before]
    return next(o for o in new if o.type == "ARMATURE"), next(o for o in new if o.type == "MESH")


def import_anim(ie, rig, path):
    try:
        ie.import_animfile(path, frame_start=1)
    except RuntimeError as exc:
        missing = re.findall(r"'([^']+)'", str(exc)) if "Missing bones" in str(exc) else None
        if not missing:
            raise
        root = next(b.name for b in rig.data.bones if b.parent is None)
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode="EDIT")
        for n in missing:
            b = rig.data.edit_bones.new(n)
            b.head, b.tail = (0, 0, 0), (0, 0, 0.5)
            b.parent = rig.data.edit_bones[root]
        bpy.ops.object.mode_set(mode="OBJECT")
        for n in missing:
            rig.data.bones[n].use_deform = False
        ie.import_animfile(path, frame_start=1)


def coords(obj):
    ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    me = ev.to_mesh()
    co = [obj.matrix_world @ v.co for v in me.vertices]
    ev.to_mesh_clear()
    return co


def tri_normal(co, t):
    return (co[t[1]] - co[t[0]]).cross(co[t[2]] - co[t[0]]).normalized()


def frame_range(rig):
    act = rig.animation_data.action if rig.animation_data else None
    if act is None:
        return 1, 1
    a, b = act.frame_range
    return int(a), int(b)


def render(scene, mesh, values, rig, out, size):
    """Vertex colours from ``values`` (|log stretch| per vertex), side and 3/4 views."""
    attr = mesh.data.color_attributes.get("stretch") or mesh.data.color_attributes.new("stretch", "FLOAT_COLOR", "POINT")
    for i, v in enumerate(values):
        t = min(1.0, v / math.log(1.5))
        attr.data[i].color = (0.62 + 0.38 * t, 0.62 * (1 - t), 0.64 * (1 - t), 1.0)
    mesh.data.color_attributes.active_color = attr
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "VERTEX"
    scene.render.resolution_x = scene.render.resolution_y = size
    rig.hide_render = True
    co = coords(mesh)
    lo = Vector((min(c.x for c in co), min(c.y for c in co), min(c.z for c in co)))
    hi = Vector((max(c.x for c in co), max(c.y for c in co), max(c.z for c in co)))
    centre, extent = (lo + hi) / 2, (hi - lo).length
    cam_data = bpy.data.cameras.get("cam") or bpy.data.cameras.new("cam")
    cam_data.type, cam_data.ortho_scale = "ORTHO", extent * 1.05
    cam = bpy.data.objects.get("cam") or bpy.data.objects.new("cam", cam_data)
    if cam.name not in scene.collection.objects:
        scene.collection.objects.link(cam)
    scene.camera = cam
    paths = {}
    for view, (az, el) in (("side", (90, 5)), ("three_quarter", (45, 20))):
        a, e = math.radians(az), math.radians(el)
        d = Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))
        cam.location = centre + d * extent * 2
        cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = f"{out}_{view}.png"
        bpy.ops.render.render(write_still=True)
        paths[view] = scene.render.filepath
    return paths


def main():
    a = _pdx.args()
    ie = _pdx.io_module()
    scene = bpy.context.scene
    out = {"anims": {}}
    for anim in a["anims"]:
        _pdx.clear_scene()
        rig, mesh = import_mesh(ie, a["mesh"])
        rest = [mesh.matrix_world @ v.co for v in mesh.data.vertices]
        tris = [tuple(p.vertices) for p in mesh.data.polygons]
        edges = sorted({tuple(sorted((t[i], t[(i + 1) % 3]))) for t in tris for i in range(3)})
        rest_len = [(rest[p] - rest[q]).length for p, q in edges]
        groups = {g.index: g.name for g in mesh.vertex_groups}
        dominant_v = []
        for v in mesh.data.vertices:
            best = max(v.groups, key=lambda g: g.weight, default=None)
            dominant_v.append(groups[best.group] if best else None)
        tri_bone = []
        for t in tris:
            score = {}
            for vi in t:
                for g in mesh.data.vertices[vi].groups:
                    score[groups[g.group]] = score.get(groups[g.group], 0) + g.weight
            tri_bone.append(max(score, key=score.get) if score else None)
        rest_n = [tri_normal(rest, t) for t in tris]
        import_anim(ie, rig, anim["path"])
        rest_bone = {b.name: rig.matrix_world @ b.matrix_local for b in rig.data.bones}
        f0, f1 = frame_range(rig)
        res = {"frames": [f0, f1], "str_min": 9.0, "str_max": 0.0, "str_p05": 9.0, "str_p95": 0.0, "flips": 0,
               "worst_frame": f0, "worst_log": 0.0, "worst_bones": None}
        for f in range(f0, f1 + 1):
            scene.frame_set(f)
            co = coords(mesh)
            ratios = [(co[p] - co[q]).length / l0 for (p, q), l0 in zip(edges, rest_len) if l0 > 1e-6]
            srt = sorted(ratios)
            res["str_min"] = min(res["str_min"], srt[0])
            res["str_max"] = max(res["str_max"], srt[-1])
            res["str_p05"] = min(res["str_p05"], srt[int(0.05 * (len(srt) - 1))])
            res["str_p95"] = max(res["str_p95"], srt[int(0.95 * (len(srt) - 1))])
            logs = [abs(math.log(max(r, 1e-6))) for r in ratios]
            k = max(range(len(logs)), key=logs.__getitem__)
            if logs[k] > res["worst_log"]:
                p, q = [e for e, l0 in zip(edges, rest_len) if l0 > 1e-6][k]
                res.update(worst_log=logs[k], worst_frame=f, worst_bones=sorted({dominant_v[p], dominant_v[q]} - {None}))
            flips = 0
            for t, n0, b in zip(tris, rest_n, tri_bone):
                if b is None or b not in rig.pose.bones:
                    continue
                m = (rig.matrix_world @ rig.pose.bones[b].matrix) @ rest_bone[b].inverted()
                if (m.to_3x3() @ n0).dot(tri_normal(co, t)) < 0:
                    flips += 1
            res["flips"] = max(res["flips"], flips)
        if a.get("renders"):
            scene.frame_set(res["worst_frame"])
            co = coords(mesh)
            per_v = [0.0] * len(co)
            for (p, q), l0 in zip(edges, rest_len):
                if l0 > 1e-6:
                    v = abs(math.log(max((co[p] - co[q]).length / l0, 1e-6)))
                    per_v[p], per_v[q] = max(per_v[p], v), max(per_v[q], v)
            r = a["renders"]
            res["renders"] = render(scene, mesh, per_v, rig, os.path.join(r["out_dir"], f"{r['name']}_{anim['id']}_worst"),
                                    int(r.get("size", 512)))
        out["anims"][anim["id"]] = res
    out["vertices"], out["triangles"] = len(rest), len(tris)
    _pdx.write_result(a, out)


main()
