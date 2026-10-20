"""Blender job (runs inside Blender only), Track M master build (SPEC M4 S2-S7): assemble the master from
the stage outputs and render it through the solved reference cameras.

Usage: blender --background --python master_assemble.py -- args.json

Parts (scene: the mirrored frame of _ref3d_scene; outputs mirrored back to PDX space):
  body      E-A's fitted body in L's pose; classes by part: coat, leg (the legs below the barrel),
            chanfron (head and ears: E-A's head is the head and chanfron together).
  armour    the plates and backing of the master's armour run (E-B's generator over every scale area).
  cloth     the caparison panels of the cloth runs: their grid, pattern UV and texture; their class
            map (cloth / trim) read from the texture's colours (gold = trim).
  tail      a tapered hair tube from the tail bone, hanging to ``tail_drop`` of the seat height.
  plume     a red tuft at the poll.
  tassels   along every panel's hem, every ``tassel_step_m``.
  decals    key details that lie on the body, not on a panel: each detail's reference pixel shot through
            its view's solved camera onto the master; a disc there carrying that patch of the reference
            (S7: unique ornament projected where observed).
Materials: every class's colour is the median of the reference's observed pixels of that class (L and
R); the panels and decals carry their projected textures.
Renders per view: lit (Cycles), albedo and class ID (flat emission, one sample, no filter: exact), and
the H3 views (a close-up of the neck and chest plates; the unseen right side). ``breaks`` re-render with
one deliberate break (E-D in 3D): trim_removed, caparison_short, plates_bigger, head_5deg, wrong_red.
"""
import colorsys
import json
import math
import sys
import time
from pathlib import Path

import bpy  # noqa: E402 - Blender module
import numpy as np
from mathutils import Vector  # noqa: E402 - Blender module
from mathutils.bvhtree import BVHTree  # noqa: E402 - Blender module

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "ai" / "jobs"))
import _ref3d_body as RB  # noqa: E402 - numpy only
from _ref3d_scene import (body_in_view, camera, decode, load_rgb, look_camera, mesh_object, orient_outward,  # noqa: E402
                          render, set_engine_cycles, to_pdx, to_scene, vertex_part, world_sky, write_obj)
from eb_plates import poly_mask  # noqa: E402


def srgb_to_linear(c):
    c = np.asarray(c, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def palettes(a):
    """Median sRGB colour per class over the references' observed pixels (L, then R where L has none)."""
    out = {}
    for vn, v in a["views"].items():
        img = load_rgb(v["image"])
        lab = decode(load_rgb(v["labels"]), a["classes"])
        inp = load_rgb(v["inpainted"])[..., 0] > 127
        never = poly_mask(lab.shape, v.get("never_use"))
        for i, c in enumerate(a["classes"]):
            if c in out:
                continue
            m = (lab == i) & ~inp & ~never
            if m.sum() >= 200:
                out[c] = np.median(img[m], axis=0)
    return out


def read_obj(path):
    V, F = [], []
    for line in open(path, encoding="utf-8"):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            F.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
    return np.array(V), np.array(F)


def emission_material(name, colour=None, image=None, closest=False):
    """A flat emission material (the albedo and ID passes): a colour, or an image on the UVs."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    if image is not None:
        tx = nt.nodes.new("ShaderNodeTexImage")
        tx.image = image
        tx.interpolation = "Closest" if closest else "Linear"
        nt.links.new(tx.outputs["Color"], em.inputs["Color"])
    else:
        em.inputs["Color"].default_value = (*colour, 1.0)
    return m


def principled(name, colour, metallic=0.0, roughness=0.5, image=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*colour, 1.0)
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = roughness
    if image is not None:
        tx = m.node_tree.nodes.new("ShaderNodeTexImage")
        tx.image = image
        m.node_tree.links.new(tx.outputs["Color"], b.inputs["Base Color"])
    return m


def image_from_array(name, arr):
    H, W = arr.shape[:2]
    img = bpy.data.images.new(name, W, H, alpha=True)
    rgba = np.ones((H, W, 4), np.float32)
    rgba[..., :3] = arr[..., :3] / 255.0
    if arr.shape[2] == 4:
        rgba[..., 3] = arr[..., 3] / 255.0
    img.pixels.foreach_set(rgba[::-1].ravel())
    img.pack()
    return img


def tube(path, radii, n=10):
    """A closed tube along ``path`` (k, 3) with ``radii`` (k,). -> (V, F)."""
    V, F = [], []
    k = len(path)
    for i in range(k):
        t = path[min(i + 1, k - 1)] - path[max(i - 1, 0)]
        t /= np.linalg.norm(t)
        u = np.cross(t, [1.0, 0.0, 0.0])
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(t, [0.0, 0.0, 1.0])
        u /= np.linalg.norm(u)
        w = np.cross(t, u)
        for j in range(n):
            ang = 2 * np.pi * j / n
            V.append(path[i] + radii[i] * (np.cos(ang) * u + np.sin(ang) * w))
    for i in range(k - 1):
        for j in range(n):
            a0, a1 = i * n + j, i * n + (j + 1) % n
            b0, b1 = (i + 1) * n + j, (i + 1) * n + (j + 1) % n
            F += [(a0, b0, b1), (a0, b1, a1)]
    c0, c1 = len(V), len(V) + 1
    V += [path[0], path[-1]]
    for j in range(n):
        F += [(c0, (j + 1) % n, j), (c1, (k - 1) * n + j, (k - 1) * n + (j + 1) % n)]
    return np.array(V), np.array(F)


def classify_texture(tex, cloth_i, trim_i):
    """Panel texels: gold (hue 25-65 deg, saturation > 0.35, value > 0.45) = trim, the rest cloth."""
    rgb = tex[..., :3].astype(float) / 255.0
    mx, mn = rgb.max(axis=-1), rgb.min(axis=-1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    hsv_h = np.zeros(mx.shape)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    d = np.maximum(mx - mn, 1e-6)
    hsv_h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    gold = (hsv_h > 25) & (hsv_h < 65) & (sat > 0.35) & (mx > 0.45)
    return np.where(gold, trim_i, cloth_i)


BARY7 = np.array([[1 / 3, 1 / 3, 1 / 3], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6],
                  [0.45, 0.45, 0.1], [0.1, 0.45, 0.45], [0.45, 0.1, 0.45]])


def label_faces(a, Vs, Fs, fpart, want, cand_parts, bvh):
    """S6 by projection: per body triangle of ``cand_parts``, 7 samples through each view's solved
    camera (R on the barrel only: the body stands in L's pose) with visibility; each sample's label
    (majority in a 5 px window). A triangle takes the class of ``want`` with the most votes when that
    class has 3+ samples and outvotes the others; triangles no view decides take their mirror's
    (symmetry). -> class name per face (None: none), evidence per face ("observed" / "symmetry")."""
    names = list(a["classes"])
    out = np.full(len(Fs), None, object)
    ev = np.full(len(Fs), None, object)
    labs = {vn: decode(load_rgb(v["labels"]), a["classes"]) for vn, v in a["views"].items()}
    for vn, v in a["views"].items():
        never = poly_mask(labs[vn].shape, v.get("never_use"))
        labs[vn][never] = -1
    fids = np.nonzero(np.isin(fpart, cand_parts))[0]
    votes = np.zeros((len(Fs), len(names) + 1), int)
    for vn, v in a["views"].items():
        lab = labs[vn]
        Hh, Ww = lab.shape
        R, t = RB.camera_matrix(v["camera"])
        cpos = Vector(-R.T @ t)
        for f in fids:
            if vn != "L" and fpart[f] != 0:
                continue
            pts = BARY7 @ Vs[Fs[f]]
            p2, _ = RB.project(to_pdx(pts), v["camera"])
            for x, (u, w) in zip(pts, p2):
                if not (0 <= u < Ww and 0 <= w < Hh):
                    continue
                d = Vector(x) - cpos
                hit = bvh.ray_cast(cpos, d.normalized(), d.length - 1e-3)
                if hit[0] is not None and (hit[0] - Vector(x)).length > 0.05:
                    continue
                win = lab[int(max(w - 2, 0)):int(w + 3), int(max(u - 2, 0)):int(u + 3)].ravel()
                win = win[win >= 0]
                if len(win):
                    votes[f, np.bincount(win).argmax()] += 1
    wi = [names.index(c) for c in want]
    for f in fids:
        if votes[f].sum() < 3:
            continue
        best = int(np.argmax(votes[f]))
        if best in wi and votes[f, best] >= 3:
            out[f], ev[f] = names[best], "observed"
        else:
            ev[f] = "observed"  # decided: not one of ``want``
    cen = Vs[Fs].mean(axis=1)
    known = fids[ev[fids] != None]  # noqa: E711
    if len(known):
        from mathutils.kdtree import KDTree
        kd = KDTree(len(known))
        for f in known:
            kd.insert(Vector(cen[f]), int(f))
        kd.balance()
        for f in fids[ev[fids] == None]:  # noqa: E711
            co, g, dist = kd.find(Vector(cen[f] * np.array([-1.0, 1.0, 1.0])))
            if g is not None and dist < 0.3:
                out[f], ev[f] = out[g], "symmetry"
    return out, ev


def shell(Vs, Fs, sel, nrm, offset):
    """The selected faces of the body, offset along the vertex normals (their own copy). -> (V, F)."""
    used = np.unique(Fs[sel])
    remap = -np.ones(len(Vs), int)
    remap[used] = np.arange(len(used))
    return Vs[used] + nrm[used] * offset, remap[Fs[sel]]


def main():
    t0 = time.time()
    a = json.loads(open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8").read())
    out = Path(a["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    brk = a.get("break")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sk = json.loads(Path(a["skeleton"]).read_text(encoding="utf-8"))
    classes = list(a["classes"])
    cid = {c: np.array(a["classes"][c], float) / 255.0 for c in classes}  # ID colours (sRGB, emitted as-is)
    pal = palettes(a)
    res = {"palettes": {k: [float(x) for x in v] for k, v in pal.items()}, "parts": {}}

    def lin(c):
        return tuple(srgb_to_linear(pal.get(c, [128, 128, 128])))

    a["view_name"] = "L"
    body = a["body"]
    if brk == "head_5deg":  # E-D's head break: 5 deg nose-down
        body = json.loads(json.dumps(body))
        body["poses"]["L"]["head_pitch"] += 5.0
        a["body"] = body
    Bp, Hs, P, V, F, part = body_in_view(a, sk)
    Vs = to_scene(V)
    Fs = orient_outward(Vs, F[:, ::-1], part)
    vpart = vertex_part(F, part, len(V))
    ground = float(V[:, 1].min())
    band = (vpart == 0) & (np.abs(V[:, 2] - (Bp["HorseFrontLeftUp"][2] + 0.4)) < 0.5)
    mpu = a["withers_m"] / (float(V[band, 1].max()) - ground)
    res["metres_per_unit"] = mpu

    objs = {}  # name -> (object, class, lit material, albedo material, id material)

    def add(name, Vsc, Fsc, cls, lit_mat, alb=None, idm=None):
        ob = mesh_object(name, Vsc, Fsc)
        ob.data.materials.append(lit_mat)
        objs[name] = (ob, cls, lit_mat, alb or emission_material(f"{name}_alb", lin(cls)),
                      idm or emission_material(f"{name}_id", tuple(srgb_to_linear(cid[cls] * 255))))
        for poly in ob.data.polygons:
            poly.use_smooth = True
        return ob

    # ---- body by class: legs below the barrel's bottom, head and ears chanfron, the rest coat
    barrel_bottom = float(V[vpart == 0, 1].min())
    fcls = np.full(len(Fs), "coat", object)
    cen = V[F].mean(axis=1)
    fcls[(part >= 5) & (cen[:, 1] < barrel_bottom)] = "leg"
    fcls[np.isin(part, [2, 3, 4])] = "chanfron"
    for c, mat in (("coat", principled("coat", lin("coat"), 0.0, 0.55)),
                   ("leg", principled("leg", lin("leg"), 0.0, 0.5)),
                   ("chanfron", principled("chanfron", lin("chanfron"), 0.9, 0.3))):
        sel = fcls == c
        if sel.any():
            add(f"body_{c}", Vs, Fs[sel], c, mat)
    res["parts"]["body"] = {"tris": int(len(Fs))}

    # ---- armour
    arm = Path(a["armour_dir"])
    pv, pf = read_obj(arm / "plates.obj")
    if brk == "plates_bigger":  # E-D: plates x1.3, each about its own centre
        k = a["verts_per_plate"]
        P3 = pv.reshape(-1, k, 3)
        c3 = P3.mean(axis=1, keepdims=True)
        pv = (c3 + 1.3 * (P3 - c3)).reshape(-1, 3)
    add("plates", to_scene(pv), pf[:, ::-1], "scale", principled("plates", lin("scale"), 0.85, 0.35))
    bv, bf = read_obj(arm / "backing.obj")
    add("backing", to_scene(bv), bf[:, ::-1], "scale", principled("backing", (0.03, 0.02, 0.015), 0.0, 0.8))
    res["parts"]["armour"] = {"plates_tris": int(len(pf)), "backing_tris": int(len(bf))}

    # ---- S6 by projection: straps, saddle (leather) and gold bands (trim) on the neck and barrel
    from _ref3d_scene import vertex_normals
    body_bvh = BVHTree.FromPolygons([tuple(v) for v in Vs], [tuple(int(i) for i in f) for f in Fs])
    fcls_lab, fev_lab = label_faces(a, Vs, Fs, part, ["leather", "trim"], [0, 1], body_bvh)
    nrm_b = vertex_normals(Vs, Fs)
    seam_y = a.get("cloth_seam_y")  # below the caparison's seam, trim belongs to the panels
    res["parts"]["label_shells"] = {}
    for c, off, thick, metal in (("leather", 0.02, 0.006, 0.0), ("trim", 0.02, 0.004, 0.8)):
        sel = fcls_lab == c
        if c == "trim" and seam_y is not None:
            sel &= cen[:, 1] > seam_y
        if brk == "trim_removed" and c == "trim":
            continue
        if sel.sum() == 0:
            continue
        sv, sf = shell(Vs, Fs, sel, nrm_b, off / mpu)
        ob = add(f"shell_{c}", sv, sf, c, principled(f"shell_{c}", lin(c), metal, 0.45))
        so = ob.modifiers.new("thickness", "SOLIDIFY")
        so.thickness = thick / mpu
        res["parts"]["label_shells"][c] = {"faces": int(sel.sum()), "symmetry": int((sel & (fev_lab == "symmetry")).sum())}

    # ---- cloth panels
    names = list(a["classes"])
    for pn in a["panels"]:
        Vf = np.array(json.loads(Path(pn["vertices"]).read_text(encoding="utf-8")))
        rows, cols = pn["rows"], pn["cols"]
        if brk == "caparison_short":  # E-D: the caparison 10% shorter (the panel's lowest tenth gone)
            keep = int(round(rows * 0.9))
            Vf = Vf.reshape(rows + 1, cols, 3)[:keep + 1].reshape(-1, 3)
            rows = keep
        tex = load_rgb(pn["texture"]).astype(np.float64)
        idx = classify_texture(tex, names.index("cloth"), names.index("trim"))
        if brk == "trim_removed":  # E-D: the trim painted out (cloth colour)
            tex[idx == names.index("trim")] = pal.get("cloth", [140, 30, 40])
            idx[:] = names.index("cloth")
        if brk == "wrong_red":  # E-D: the cloth's hue +15 deg
            cl = idx == names.index("cloth")
            hsv = np.array([colorsys.rgb_to_hsv(*(c / 255.0)) for c in tex[cl]])
            hsv[:, 0] = (hsv[:, 0] + 15 / 360.0) % 1.0
            tex[cl] = np.array([colorsys.hsv_to_rgb(*h) for h in hsv]) * 255
        Hh = tex.shape[0]
        tex = tex[: int(round(Hh * rows / pn["rows"]))] if rows != pn["rows"] else tex
        idx = idx[: tex.shape[0]]
        img = image_from_array(f"{pn['name']}_tex", tex)
        id_rgb = (np.array([a["classes"][c] for c in names])[idx]).astype(np.float64)
        id_img = image_from_array(f"{pn['name']}_id", id_rgb)
        # sRGB like every colour here: decoded to linear, emitted, encoded back by the Standard view: exact
        Fg = []
        for r in range(rows):
            for c in range(cols - 1):
                a0, a1, b0, b1 = r * cols + c, r * cols + c + 1, (r + 1) * cols + c, (r + 1) * cols + c + 1
                Fg += [(a0, b0, b1), (a0, b1, a1)]
        ob = add(pn["name"], Vf, np.array(Fg), "cloth", principled(f"{pn['name']}_lit", lin("cloth"), 0.0, 0.85, img),
                 emission_material(f"{pn['name']}_alb", image=img),
                 emission_material(f"{pn['name']}_idm", image=id_img, closest=True))
        uv = ob.data.uv_layers.new(name="pattern")
        uvs = np.c_[np.tile(np.linspace(0, 1, cols), rows + 1), 1 - np.repeat(np.linspace(0, 1, rows + 1), cols)]
        uv.data.foreach_set("uv", uvs[np.array([lp.vertex_index for lp in ob.data.loops])].astype(np.float32).ravel())
        sol = ob.modifiers.new("thickness", "SOLIDIFY")
        sol.thickness = 0.004 / mpu
        # tassels along the hem (none when the trim is removed: they're trim)
        if brk != "trim_removed":
            hem = Vf.reshape(rows + 1, cols, 3)[-1]
            seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(hem, axis=0), axis=1))]
            TV, TF = [], []
            for s_ in np.arange(0.5 * a["tassel_step_m"] / mpu, seg[-1], a["tassel_step_m"] / mpu):
                p = np.array([np.interp(s_, seg, hem[:, k]) for k in range(3)])
                path = np.array([p, p + np.array([0, -0.5, 0]) * a["tassel_len_m"] / mpu, p + np.array([0, -1.0, 0]) * a["tassel_len_m"] / mpu])
                tv, tf = tube(path, np.array([0.006, 0.012, 0.004]) / mpu, 6)
                TF.append(tf + sum(len(x) for x in TV))
                TV.append(tv)
            if TV:
                add(f"{pn['name']}_tassels", np.vstack(TV), np.vstack(TF), "trim", principled(f"{pn['name']}_tas", lin("trim"), 0.6, 0.4))
        res["parts"][pn["name"]] = {"rows": rows, "cols": cols, "evidence": pn.get("evidence")}

    # ---- tail: from the tail bone down and a little back
    tt = to_scene(np.array([Bp["HorseTailTop"]]))[0]
    drop = a["tail_drop"] * Hs
    path = np.array([tt + np.array([0, -t * drop, 0.12 * Hs * math.sin(t * 1.4)]) for t in np.linspace(0, 1, 12)])
    rad = np.interp(np.linspace(0, 1, 12), [0, 0.15, 1], [0.05, 0.075, 0.035]) * Hs
    tv, tf = tube(path, rad, 12)
    add("tail", tv, tf, "tail", principled("tail", lin("tail"), 0.0, 0.7))
    # ---- plume: a tuft of cones up and back from the poll
    poll = to_scene(np.array([Bp["HorseHead"]]))[0]
    PV, PF = [], []
    for k, (dx, dz) in enumerate([(0, 0), (0.03, 0.04), (-0.03, 0.04), (0.02, -0.03), (-0.02, -0.03)]):
        base = poll + np.array([dx, 0.05, dz]) * Hs
        tip = base + np.array([dx * 0.5, 0.16, 0.06 + dz]) * Hs
        tv, tf = tube(np.array([base, 0.5 * (base + tip), tip]), np.array([0.025, 0.02, 0.003]) * Hs, 8)
        PF.append(tf + sum(len(x) for x in PV))
        PV.append(tv)
    add("plume", np.vstack(PV), np.vstack(PF), "plume", principled("plume", lin("plume"), 0.0, 0.8))

    # ---- decals: key details on the body, projected from their view
    allV, allF = [], []
    for name, (ob, *_r) in objs.items():
        me = ob.data
        vv = np.array([v.co[:] for v in me.vertices])
        ff = np.array([p.vertices[:] for p in me.polygons if len(p.vertices) == 3])
        allF.append(ff + sum(len(x) for x in allV))
        allV.append(vv)
    bvh = BVHTree.FromPolygons([tuple(v) for v in np.vstack(allV)], [tuple(int(i) for i in f) for f in np.vstack(allF)])
    res["decals"] = {}
    for d in a["decals"]:
        if brk == "trim_removed" and d.get("class") == "trim":
            continue
        v = a["views"][d["view"]]
        R, t = RB.camera_matrix(v["camera"])
        cpos = -R.T @ t
        cam = v["camera"]
        x, y = d["xy"]
        ray = R.T @ np.array([(x - cam["cx"]) / cam["focal"], -(y - cam["cy"]) / cam["focal"], 1.0])
        ray /= np.linalg.norm(ray)
        loc, nrm, fi, dist = bvh.ray_cast(Vector(cpos), Vector(ray))
        if loc is None:
            res["decals"][d["name"]] = {"placed": False}
            continue
        loc, nrm = np.array(loc), np.array(nrm)
        if nrm @ ray > 0:
            nrm = -nrm
        size = d["size"] * dist / cam["focal"]  # the patch's width at that depth
        img_ref = load_rgb(v["image"])
        h2 = d["size"] // 2
        crop = img_ref[max(0, int(y) - h2):int(y) + h2, max(0, int(x) - h2):int(x) + h2]
        yy, xx = np.mgrid[:crop.shape[0], :crop.shape[1]]
        alpha = (np.hypot(yy - crop.shape[0] / 2, xx - crop.shape[1] / 2) < 0.48 * min(crop.shape[:2])) * 255
        dimg = image_from_array(f"decal_{d['name']}", np.dstack([crop, alpha]).astype(np.float64))
        # a quad facing the view through the hit point, 2 mm off the surface
        right, up = R[0], R[1]
        c0 = loc + nrm * 0.002 / mpu
        q = np.array([c0 + (-right - up) * size / 2, c0 + (right - up) * size / 2, c0 + (right + up) * size / 2, c0 + (-right + up) * size / 2])
        mat = bpy.data.materials.new(f"decal_{d['name']}")
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = nt.nodes["Principled BSDF"]
        tx = nt.nodes.new("ShaderNodeTexImage")
        tx.image = dimg
        nt.links.new(tx.outputs["Color"], bsdf.inputs["Base Color"])
        nt.links.new(tx.outputs["Alpha"], bsdf.inputs["Alpha"])
        mat.blend_method = "HASHED"
        alb = emission_material(f"decal_{d['name']}_alb", image=dimg)
        idm = emission_material(f"decal_{d['name']}_idm", tuple(srgb_to_linear(cid[d["class"]] * 255)))
        for m_ in (alb, idm):  # cut out by the disc's alpha
            nt2 = m_.node_tree
            em = [n for n in nt2.nodes if n.type == "EMISSION"][0]
            outn = [n for n in nt2.nodes if n.type == "OUTPUT_MATERIAL"][0]
            mix = nt2.nodes.new("ShaderNodeMixShader")
            tr = nt2.nodes.new("ShaderNodeBsdfTransparent")
            tx2 = nt2.nodes.new("ShaderNodeTexImage")
            tx2.image = dimg
            nt2.links.new(tx2.outputs["Alpha"], mix.inputs[0])
            nt2.links.new(tr.outputs[0], mix.inputs[1])
            nt2.links.new(em.outputs[0], mix.inputs[2])
            nt2.links.new(mix.outputs[0], outn.inputs["Surface"])
        ob = add(f"decal_{d['name']}", q, np.array([[0, 1, 2], [0, 2, 3]]), d["class"], mat, alb, idm)
        uvl = ob.data.uv_layers.new(name="uv")
        uvl.data.foreach_set("uv", np.array([[0, 0], [1, 0], [1, 1], [0, 0], [1, 1], [0, 1]], np.float32).ravel())
        res["decals"][d["name"]] = {"placed": True, "pdx": to_pdx(loc[None])[0].tolist(), "size_m": float(size * mpu)}

    # ---- renders
    world_sky(sc)
    sc.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 4.0
    sun.rotation_euler = (math.radians(50), math.radians(-20), math.radians(35))
    sc.collection.objects.link(sun)
    res["cycles_device"] = set_engine_cycles(sc, a.get("samples", 96))
    files = {}

    def set_pass(kind):
        for ob, cls, lit, alb, idm in objs.values():
            ob.data.materials.clear()
            ob.data.materials.append({"lit": lit, "albedo": alb, "id": idm}[kind])
        flat = kind != "lit"
        sc.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0 if flat else 1.0  # the reference reads bright
        sun.hide_render = flat
        sc.cycles.samples = 1 if kind == "id" else (16 if kind == "albedo" else a.get("samples", 96))
        sc.cycles.use_denoising = kind == "lit"
        sc.cycles.filter_width = 0.01 if kind == "id" else 1.5
        sc.view_settings.view_transform = "Standard" if flat else "AgX"
        sc.render.film_transparent = flat
        sc.render.dither_intensity = 0.0 if flat else 1.0  # dither would nudge the exact class colours

    for vn, v in a["views"].items():
        W, H = v["size"]
        R, t = RB.camera_matrix(v["camera"])
        cob = camera(f"cam_{vn}", R, t, v["camera"]["focal"], v["camera"]["cx"], v["camera"]["cy"], W, H)
        for kind in (("id", "albedo") if brk else ("lit", "albedo", "id")):
            set_pass(kind)
            render(sc, cob, out / f"{vn}_{kind}.png", W, H, "RGBA" if kind != "lit" else "RGB")
            files[f"{vn}_{kind}"] = str(out / f"{vn}_{kind}.png")
    if not brk:
        set_pass("lit")
        sc.render.image_settings.color_mode = "RGB"
        pts = np.vstack([objs["plates"][0].data.vertices[i].co[:] for i in range(0, len(objs["plates"][0].data.vertices), 37)])
        centre = pts.mean(axis=0)
        rad = float(np.percentile(np.linalg.norm(pts - centre, axis=1), 90))
        allp = np.vstack([np.array([vt.co[:] for vt in ob.data.vertices]) for ob, *_r in objs.values()])
        hc = 0.5 * (allp.min(axis=0) + allp.max(axis=0))  # the whole horse
        hr = 0.5 * float(np.linalg.norm(allp.max(axis=0) - allp.min(axis=0)))
        # az: 0 the horse's left, +90 its front, 180 its right (camera_matrix's convention)
        for name, az, el, target, dist in (("closeup_plates", 45, 12, centre, 0.55 * rad / math.tan(math.radians(14))),
                                           ("unseen_right", 140, 10, hc, 1.15 * hr / math.tan(math.radians(17))),
                                           ("front", 90, 5, hc, 1.15 * hr / math.tan(math.radians(17)))):
            cam_o = look_camera(f"cam_{name}", target, az, el, dist, 50.0)
            render(sc, cam_o, out / f"{name}.png", 1400, 1000)
            files[name] = str(out / f"{name}.png")
        # parts as OBJ in PDX space (the master's geometry), and the scene
        for name, (ob, *_r) in objs.items():
            me = ob.data
            vv = np.array([vt.co[:] for vt in me.vertices])
            ff = np.array([p.vertices[:] for p in me.polygons])
            write_obj(out / f"{name}.obj", to_pdx(vv), ff[:, ::-1], f"Track M master part {name} (generated; PDX mesh space)")
        bpy.ops.wm.save_as_mainfile(filepath=str(out / "master.blend"), compress=True)
        files["blend"] = str(out / "master.blend")
    # the master's landmarks per view (body landmarks; the hem landmarks: the panels' nearest hem point)
    lm3 = RB.landmarks(Bp, Hs, P)
    hems = [np.array(json.loads(Path(pn["vertices"]).read_text(encoding="utf-8"))).reshape(pn["rows"] + 1, pn["cols"], 3)[-1]
            for pn in a["panels"]]
    res["landmarks"] = {}
    for vn, v in a["views"].items():
        names_ = [k for k in lm3 if k in v["landmarks"]]
        p2, _ = RB.project(np.array([lm3[k] for k in names_]), v["camera"]) if names_ else (np.zeros((0, 2)), None)
        lm = {k: [float(x) for x in p] for k, p in zip(names_, p2)}
        for k, xy in v["landmarks"].items():
            if k.startswith("hem_") and hems:
                hp, _ = RB.project(to_pdx(np.vstack(hems)), v["camera"])
                j = int(np.argmin(np.linalg.norm(hp - np.array(xy), axis=1)))
                lm[k] = [float(x) for x in hp[j]]
        res["landmarks"][vn] = lm
    res["files"] = files
    res["seconds"] = round(time.time() - t0, 1)
    with open(a["result_path"], "w", encoding="utf-8") as f:
        json.dump(res, f, default=float)


main()
