"""Blender job: deformation check of a derived body's skinned additions (runs in Blender only;
SPEC Track R, the spike R0 metrics made a pipeline step).

For each animation: import the derived body .mesh (written by core.derived_body; nothing
exported), import the vanilla .anim (bones the .anim names but the mesh lacks get unweighted
dummies under ``root``: the game ignores those tracks), and measure every frame, for every
addition (a vertex range appended after the base vertices):

  * stretch: edge length / bind length over the addition's triangle edges (min, max, p05, p95);
  * penetration: addition vertices inside the base body's triangles (nearest face, with the
    smooth normal interpolated there: the flat face normal misreads points nearest an edge),
    except the addition's ``exempt`` rings (sunk or flared into the body by design);
  * flips: triangles whose normal turns > 90 degrees from the bind pose in the root bone's frame;
  * root drift: ring-0 vertices' distance from their bind position in the root bone's frame;
  * behind: the smallest distance behind the ``behind`` bone's joint along the unit's back
    direction (from ``root``, which never pitches), rings >= 2; skipped when ``behind`` is null.

Optional worst-frame renders (Workbench, the body textured with the given diffuse).

args: {"body", "base_vertices", "additions": [{"name", "start", "count", "rings", "root", "behind",
       "exempt"}], "anims": [{"id", "path"}], "renders": {"out_dir", "size", "diffuse"} | null, "result_path"}
"""
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402 - Blender module
from mathutils import Vector  # noqa: E402 - Blender module
from mathutils.bvhtree import BVHTree  # noqa: E402 - Blender module

import _pdx  # noqa: E402


def import_body(ie, path):
    before = set(bpy.data.objects)
    ie.import_meshfile(path)
    new = [o for o in bpy.data.objects if o not in before]
    return next(o for o in new if o.type == "ARMATURE"), next(o for o in new if o.type == "MESH"), new


def import_anim(ie, rig, path):
    try:
        ie.import_animfile(path, frame_start=1)
    except RuntimeError as exc:
        missing = re.findall(r"'([^']+)'", str(exc)) if "Missing bones" in str(exc) else None
        if not missing:
            raise
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode="EDIT")
        for n in missing:
            b = rig.data.edit_bones.new(n)
            b.head, b.tail = (0, 0, 0), (0, 0, 0.5)
            b.parent = rig.data.edit_bones["root"]
        bpy.ops.object.mode_set(mode="OBJECT")
        for n in missing:
            rig.data.bones[n].use_deform = False
        rig.select_set(True)
        ie.import_animfile(path, frame_start=1)


def evaluated(obj, normals=False):
    ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    me = ev.to_mesh()
    co = [obj.matrix_world @ v.co for v in me.vertices]
    rot = obj.matrix_world.to_3x3()
    nv = [(rot @ v.normal).normalized() for v in me.vertices] if normals else None
    ev.to_mesh_clear()
    return (co, nv) if normals else co


def smooth_normal(co, nv, tri, loc):
    """The vertex normals of ``tri`` interpolated at ``loc`` (barycentric). Unlike the flat face
    normal it gives the right side for points nearest an edge or corner of a convex surface
    (test 11 prep: the flat normal flagged a tuft tip in the air as inside the head)."""
    a, b, c = (co[i] for i in tri)
    v0, v1, v2 = b - a, c - a, loc - a
    d00, d01, d11, d20, d21 = v0.dot(v0), v0.dot(v1), v1.dot(v1), v2.dot(v0), v2.dot(v1)
    den = d00 * d11 - d01 * d01
    if abs(den) < 1e-12:
        return (nv[tri[0]] + nv[tri[1]] + nv[tri[2]]).normalized()
    w1 = (d11 * d20 - d01 * d21) / den
    w2 = (d00 * d21 - d01 * d20) / den
    w0 = 1 - w1 - w2
    return (nv[tri[0]] * w0 + nv[tri[1]] * w1 + nv[tri[2]] * w2).normalized()


def bone(rig, name):
    return rig.matrix_world @ rig.pose.bones[name].matrix


def tri_normal(co, t):
    return (co[t[1]] - co[t[0]]).cross(co[t[2]] - co[t[0]]).normalized()


class Addition:
    def __init__(self, spec, body, rig, co):
        self.spec = spec
        lo, n = spec["start"], spec["count"]
        self.idx = list(range(lo, lo + n))
        self.ring = {lo + i: r for i, r in enumerate(spec["rings"])}
        polys = [tuple(p.vertices) for p in body.data.polygons]
        self.tris = [t for t in polys if all(lo <= v < lo + n for v in t)]
        self.edges = sorted({tuple(sorted((t[i], t[(i + 1) % len(t)]))) for t in self.tris for i in range(len(t))})
        self.rest_len = [(co[a] - co[b]).length for a, b in self.edges]
        inv = bone(rig, spec["root"]).inverted()
        self.rest_local = {v: inv @ co[v] for v in self.idx}
        rot = inv.to_3x3()
        self.rest_normals = [(rot @ tri_normal(co, t)).normalized() for t in self.tris]
        self.exempt = set(spec.get("exempt") or [0])
        root_inv = (rig.matrix_world @ rig.data.bones["root"].matrix_local).to_3x3().inverted()
        self.back_in_root = root_inv @ Vector((0, 1, 0))  # world +Y = the unit's back, in root's frame

    def measure(self, rig, co, tree, nv=None, base_tris=None):
        ratios = sorted((co[a] - co[b]).length / l0 for (a, b), l0 in zip(self.edges, self.rest_len) if l0 > 1e-6)
        q = lambda f: ratios[min(len(ratios) - 1, int(f * (len(ratios) - 1)))]  # noqa: E731
        inv = bone(rig, self.spec["root"]).inverted()
        rot = inv.to_3x3()
        flips = sum(1 for t, n0 in zip(self.tris, self.rest_normals) if (rot @ tri_normal(co, t)).dot(n0) < 0)
        drift = max((inv @ co[v] - self.rest_local[v]).length for v in self.idx if self.ring[v] == 0)
        pen, depth, checked = 0, 0.0, 0
        for v in self.idx:
            if self.ring[v] in self.exempt:
                continue
            checked += 1
            loc, nrm, fi, dist = tree.find_nearest(co[v])
            if loc is not None and nv is not None and fi is not None:
                nrm = smooth_normal(co, nv, base_tris[fi], loc)
            if loc is not None and (co[v] - loc).dot(nrm) < 0:
                pen += 1
                depth = max(depth, dist)
        behind = None
        if self.spec.get("behind"):
            ref = bone(rig, self.spec["behind"])
            back = bone(rig, "root").to_3x3() @ self.back_in_root
            back.z = 0
            back.normalize()
            behind = min((co[v] - ref.translation).dot(back) for v in self.idx if self.ring[v] >= 2)
        return {"str_min": ratios[0], "str_max": ratios[-1], "str_p05": q(0.05), "str_p95": q(0.95),
                "pen_pct": 100.0 * pen / max(1, checked), "pen_depth": depth, "flips": flips, "drift": drift,
                "behind": behind}


def setup_render(a, body, new):
    r = a["renders"]
    img = bpy.data.images.load(r["diffuse"], check_existing=True)
    mat = bpy.data.materials.new("check")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = img
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nodes.active = tex
    body.data.materials.clear()
    body.data.materials.append(mat)
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading
    sh.light, sh.color_type, sh.show_backface_culling = "STUDIO", "TEXTURE", False
    world = sc.world or bpy.data.worlds.new("w")
    sc.world = world
    world.color = (0.82, 0.84, 0.86)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    cam.data.type = "ORTHO"
    sc.camera = cam
    for o in new:
        if o.type != "MESH":
            o.hide_render = True
    return sc, cam


def shoot(sc, cam, co, direction, path, size):
    lo = Vector((min(c.x for c in co), min(c.y for c in co), min(c.z for c in co)))
    hi = Vector((max(c.x for c in co), max(c.y for c in co), max(c.z for c in co)))
    centre, d = (lo + hi) / 2, Vector(direction).normalized()
    extent = (hi - lo).length
    cam.location = centre - d * (extent * 2 + 10)
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    cam.data.clip_end = extent * 4 + 40
    right = d.cross(Vector((0, 0, 1)))
    right = right.normalized() if right.length > 1e-6 else Vector((1, 0, 0))
    up = right.cross(d).normalized()
    corners = [Vector((x, y, z)) for x in (lo.x, hi.x) for y in (lo.y, hi.y) for z in (lo.z, hi.z)]
    w = max(abs((c - centre).dot(right)) for c in corners) * 2
    h = max(abs((c - centre).dot(up)) for c in corners) * 2
    cam.data.ortho_scale = max(w, h) * 1.06
    sc.render.resolution_x = sc.render.resolution_y = size
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path


VIEWS = {"front": (0, 1, 0), "side": (-1, 0, 0), "rear": (0, -1, 0)}


def main():
    a = _pdx.args()
    ie = _pdx.io_module()
    out = {"anims": {}}
    for anim in a["anims"]:
        _pdx.clear_scene()
        for act in list(bpy.data.actions):
            bpy.data.actions.remove(act)
        rig, body, new = import_body(ie, a["body"])
        if len(body.data.vertices) < a["base_vertices"] + sum(s["count"] for s in a["additions"]):
            raise RuntimeError(f"imported {len(body.data.vertices)} vertices, fewer than the file's")
        bpy.context.view_layer.update()
        rest = evaluated(body)
        adds = [Addition(s, body, rig, rest) for s in a["additions"]]
        base_n = a["base_vertices"]
        base_tris = [tuple(p.vertices) for p in body.data.polygons if all(v < base_n for v in p.vertices)]
        bpy.context.view_layer.objects.active = rig
        rig.select_set(True)
        import_anim(ie, rig, anim["path"])
        sc = bpy.context.scene
        rows = {s["name"]: [] for s in a["additions"]}
        for f in range(sc.frame_start, sc.frame_end + 1):
            sc.frame_set(f)
            co, nv = evaluated(body, normals=True)
            tree = BVHTree.FromPolygons(co, base_tris)
            for add in adds:
                m = add.measure(rig, co, tree, nv, base_tris)
                m["frame"] = f
                rows[add.spec["name"]].append(m)
        res = {"frames": [sc.frame_start, sc.frame_end], "additions": {}}
        for name, ms in rows.items():
            worst = max(ms, key=lambda m: (m["pen_depth"], max(m["str_max"] - 1, 1 - m["str_min"])))
            behinds = [m["behind"] for m in ms if m["behind"] is not None]
            res["additions"][name] = {
                "str_min": min(m["str_min"] for m in ms), "str_max": max(m["str_max"] for m in ms),
                "str_p05": min(m["str_p05"] for m in ms), "str_p95": max(m["str_p95"] for m in ms),
                "pen_depth": max(m["pen_depth"] for m in ms), "pen_pct": max(m["pen_pct"] for m in ms),
                "flips": max(m["flips"] for m in ms), "drift": max(m["drift"] for m in ms),
                "behind": min(behinds) if behinds else None, "worst_frame": worst["frame"]}
        if a.get("renders"):
            r = a["renders"]
            sc, cam = setup_render(a, body, new)
            shots = {}
            for name, info in res["additions"].items():
                sc.frame_set(info["worst_frame"])
                co = evaluated(body)
                shots[name] = [shoot(sc, cam, co, d, os.path.join(r["out_dir"], f"{anim['id']}_{name}_worst_{v}.png"),
                                     r.get("size", 360)) for v, d in VIEWS.items()]
            res["renders"] = shots
        out["anims"][anim["id"]] = res
    _pdx.write_result(a, out)


if __name__ == "__main__":
    main()
