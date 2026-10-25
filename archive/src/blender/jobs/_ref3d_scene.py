"""Shared scene helpers for the Track M Blender jobs (E-B plates, E-C cloth; runs inside Blender only).

Frames: PDX mesh space is left-handed (Y up, the horse faces -Z, its left at +X); _ref3d_body mirrors x
before projecting, so the jobs build their scenes in that mirrored frame (``to_scene``) and mirror their
outputs back (``to_pdx``). ``camera`` makes a Blender camera equal to a solved reference camera.
"""
import math
import sys
from pathlib import Path

import bpy  # noqa: E402 - Blender module
import numpy as np
from mathutils import Matrix, Vector  # noqa: E402 - Blender module

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "ai" / "jobs"))
import _ref3d_body as RB  # noqa: E402 - numpy only

MIRROR = np.array([-1.0, 1.0, 1.0])


def to_scene(X):
    """PDX space -> the scene's mirrored frame (the target at the origin)."""
    return (np.asarray(X, float) - RB.TARGET) * MIRROR


def to_pdx(X):
    return np.asarray(X, float) * MIRROR + RB.TARGET


def load_rgb(path):
    """(H, W, 3) uint8, top row first."""
    img = bpy.data.images.load(str(path), check_existing=False)
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    return (np.clip(px.reshape(h, w, 4)[::-1, :, :3], 0, 1) * 255 + 0.5).astype(np.uint8)


def decode(rgb, classes):
    """Label image -> class index (-1 none), exact colours (tolerance 3)."""
    out = np.full(rgb.shape[:2], -1, np.int32)
    for i, c in enumerate(classes.values()):
        out[np.all(np.abs(rgb.astype(int) - np.array(c)) <= 3, axis=-1)] = i
    return out


def mesh_object(name, V, F, colour=None, collection=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in V], [], [tuple(int(i) for i in f) for f in F])
    me.update()
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    if colour is not None:
        ob.color = (*colour, 1.0)
    return ob


def material(name, base, metallic=0.0, roughness=0.5):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*base, 1.0)
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = roughness
    m.diffuse_color = (*base, 1.0)
    return m


def vertex_normals(V, F):
    n = np.zeros_like(V)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def orient_outward(V, F, part):
    """Each tube's triangles wound outward (positive signed volume)."""
    F = F.copy()
    for p in np.unique(part):
        s = part == p
        a, b, c = V[F[s, 0]], V[F[s, 1]], V[F[s, 2]]
        if np.einsum("ij,ij->i", a, np.cross(b, c)).sum() < 0:
            F[s] = F[s][:, ::-1]
    return F


def camera(name, R, t, focal, cx, cy, W, H):
    """A Blender camera equal to _ref3d_body's: Xc = R X' + t (X' the scene frame), pixels
    u = cx + f x/z, v = cy - f y/z."""
    cd = bpy.data.cameras.new(name)
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = 36.0
    cd.lens = focal * 36.0 / W
    # shift is a fraction of the fitted (horizontal) sensor side, the width, for both axes (tested on
    # 4.2: shift 0.1 moved a 660 x 790 image's axis point 66 px): the axis lands at u = W/2 - shift_x W
    cd.shift_x = (W / 2 - cx) / W
    cd.shift_y = (cy - H / 2) / W
    # depth range: a band round the target (E-B: fitted cameras sit up to ~2000 units away, nearly
    # orthographic; with 0.01-5000 Workbench's depth buffer couldn't separate millimetre layers)
    dist = float(t[2])
    cd.clip_start, cd.clip_end = max(0.01, dist - 60.0), dist + 60.0
    ob = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(ob)
    right, up, fwd = R
    M = Matrix(((right[0], up[0], -fwd[0]), (right[1], up[1], -fwd[1]), (right[2], up[2], -fwd[2])))
    ob.matrix_world = Matrix.Translation(Vector(-R.T @ t)) @ M.to_4x4()
    return ob


def look_camera(name, target, az, el, dist, lens=60.0):
    d = np.array([-math.cos(math.radians(el)) * math.cos(math.radians(az)), math.sin(math.radians(el)),
                  -math.cos(math.radians(el)) * math.sin(math.radians(az))])
    cd = bpy.data.cameras.new(name)
    cd.lens, cd.sensor_width = lens, 36.0
    cd.clip_start, cd.clip_end = 0.01, 5000
    ob = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(ob)
    loc = np.asarray(target, float) + d * dist
    # Y is up in these scenes (to_track_quat assumes Blender's Z up and rolled the views 90 deg)
    fwd = np.asarray(target, float) - loc
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    M = Matrix(((right[0], up[0], -fwd[0]), (right[1], up[1], -fwd[1]), (right[2], up[2], -fwd[2])))
    ob.matrix_world = Matrix.Translation(Vector(loc)) @ M.to_4x4()
    return ob


def check_camera(cam_ob, cam, X_pdx, W, H):
    """Max pixel difference between Blender's projection and _ref3d_body's on sample points."""
    from bpy_extras.object_utils import world_to_camera_view
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
    p2, _ = RB.project(X_pdx, cam)
    err = 0.0
    diffs = []
    for x, (u, v) in zip(to_scene(X_pdx), p2):
        q = world_to_camera_view(sc, cam_ob, Vector(x))
        err = max(err, math.hypot(q.x * W - u, (1 - q.y) * H - v))
        diffs.append((q.x * W - u, (1 - q.y) * H - v, u, v))
    d = np.array(diffs)
    print("camera check: mean du dv", d[:, 0].mean(), d[:, 1].mean(), "corr u", np.corrcoef(d[:, 0], d[:, 2])[0, 1],
          "corr v", np.corrcoef(d[:, 1], d[:, 3])[0, 1], flush=True)
    return err


def body_in_view(a, sk):
    b = a["body"]
    vn = a["view_name"]
    B, Hs = RB.skeleton(sk["bones"], sk["seat"], b["skeleton"])
    P = {**b["shape"], "head_pitch": b["poses"][vn]["head_pitch"], "head_yaw": b["poses"][vn].get("head_yaw", 0.0)}
    Bp = RB.pose(B, Hs, P, b["poses"][vn])
    V, F, part, P2 = RB.build(Bp, Hs, P)
    return Bp, Hs, P2, V, F, part


def vertex_part(F, part, n):
    """The part index per vertex (the tubes share no vertices)."""
    vp = np.full(n, -1)
    for k in range(3):
        vp[F[:, k]] = part
    return vp


def inside(bvh, p, dirs=((0.577, 0.577, 0.577), (-0.6, 0.48, -0.64))):
    """Ray parity in two directions (both odd = inside a closed tube). E-B v001 used the closest
    point's normal: beside a tube's edges and from a nearby leg it called points 0.5 m outside
    'inside'."""
    votes = 0
    for d in dirs:
        dv = Vector(d).normalized()
        o, n = Vector(p), 0
        for _ in range(16):
            loc, normal, fi, dist = bvh.ray_cast(o, dv)
            if loc is None:
                break
            n += 1
            o = loc + dv * 1e-5
        votes += n % 2
    return votes == len(dirs)


def in_other_tube(p, own, tubes):
    """Is ``p`` inside a tube other than part ``own``? (the tubes overlap: the neck runs into the
    barrel, the head over the poll; only the union's outer surface carries armour)."""
    for q, (bvh, lo, hi) in tubes.items():
        if q != own and np.all(p >= lo) and np.all(p <= hi) and inside(bvh, p):
            return True
    return False


def set_engine_workbench(sc, colour_type="OBJECT", flat=False):
    sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading
    sh.light = "FLAT" if flat else "STUDIO"
    sh.color_type = colour_type
    sh.show_cavity = not flat
    sh.show_shadows = False
    sh.show_specular_highlight = not flat
    sc.display.render_aa = "OFF" if flat else "8"
    sc.view_settings.view_transform = "Standard"
    sc.render.film_transparent = False


def set_engine_cycles(sc, samples):
    sc.render.engine = "CYCLES"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.view_settings.view_transform = "AgX"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = kind
                prefs.get_devices()
                if any(d.type == kind for d in prefs.devices):
                    for d in prefs.devices:
                        d.use = d.type == kind
                    sc.cycles.device = "GPU"
                    return kind
            except TypeError:
                continue
    except KeyError:
        pass
    sc.cycles.device = "CPU"
    return "CPU"


def render(sc, cam_ob, path, W, H, color_mode="RGB"):
    sc.camera = cam_ob
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
    sc.render.filepath = str(path)
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = color_mode
    bpy.ops.render.render(write_still=True)


def world_sky(sc):
    w = bpy.data.worlds.new("EB_sky")
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    bg = nt.nodes["Background"]
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_elevation = math.radians(35)
    sky.sun_rotation = math.radians(140)
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.35


def write_obj(path, V, F, header):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {header}\n")
        f.writelines(f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in V)
        f.writelines(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in F)


def smooth_normal(Vs, Fs, vn, p, fi):
    """The vertex normals interpolated (barycentric) at point ``p`` on triangle ``fi``."""
    a, b, c = Vs[Fs[fi]]
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11, d20, d21 = v0 @ v0, v0 @ v1, v1 @ v1, v2 @ v0, v2 @ v1
    den = d00 * d11 - d01 * d01
    if abs(den) < 1e-18:
        n = vn[Fs[fi]].mean(axis=0)
    else:
        v = (d11 * d20 - d01 * d21) / den
        w = (d00 * d21 - d01 * d20) / den
        n = (1 - v - w) * vn[Fs[fi][0]] + v * vn[Fs[fi][1]] + w * vn[Fs[fi][2]]
    return n / max(np.linalg.norm(n), 1e-12)
