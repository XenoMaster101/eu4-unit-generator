"""Blender job: build procedural rigid parts and export each as a static .mesh (runs in Blender only).

Geometry is written in PDX mesh space (Y-up, the attach bone's local frame) and converted to
Blender (Z-up) with ``v()``; io_pdx_mesh converts back on export. Conventions (core/parts.py,
SPEC §10 Phase 3): hand weapons along +Z from a grip at the origin ("point": "down" mirrors
them along Z, like the cavalry sabre); shields in the hand's YZ plane, domed towards +X; head
parts in the ``head`` bone frame, the unit facing -Z.

UVs: each zone's faces sit on that zone's swatch (``zone_uvs``, image coordinates, v down);
the ``face`` zone of a disc layout is planar-mapped onto the image's inscribed circle.
io_pdx_mesh flips V on export, so image coordinates become Blender UVs with v -> 1 - v.

args: {"parts": [{"kind", "builder", "shape", "out_mesh", "shader", "textures": {diff, n, spec},
                  "zones": n, "layout", "zone_uvs": [[u, v], ...], "params": {...}}], "result_path": ...}
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bmesh  # noqa: E402 - Blender module
import bpy  # noqa: E402 - Blender module

import _pdx  # noqa: E402


def v(x, y, z):
    """PDX (Y-up) -> Blender (Z-up), the inverse of io_pdx_mesh's SPACE_MATRIX swap."""
    return (x, z, y)


class Builder:
    def __init__(self, zone_uvs, flip_z=False):
        self.bm = bmesh.new()
        self.uv = self.bm.loops.layers.uv.new("map1")
        self.zone = self.bm.faces.layers.int.new("zone")  # kept through triangulation; the atlas painter needs it
        self.zone_uvs = zone_uvs
        self.flip_z = flip_z
        self.pdx = {}  # vert -> PDX coords (for planar UVs)

    def vert(self, x, y, z):
        if self.flip_z:
            z = -z
        vert = self.bm.verts.new(v(x, y, z))
        self.pdx[vert] = (x, y, z)
        return vert

    def face(self, verts, zone, planar=None):
        if self.flip_z:  # mirroring turns faces inside out; recalc_face_normals fixes closed shells, keep winding sane
            verts = list(reversed(verts))
        f = self.bm.faces.new(verts)
        f[self.zone] = zone
        for loop in f.loops:
            u, vv = planar(self.pdx[loop.vert]) if planar else self.zone_uvs[zone]
            loop[self.uv].uv = (u, 1.0 - vv)
        return f

    def ring(self, z, radius, sides, start=0.0, sx=1.0, sy=1.0):
        return [self.vert(radius * sx * math.cos(start + 2 * math.pi * i / sides),
                          radius * sy * math.sin(start + 2 * math.pi * i / sides), z) for i in range(sides)]

    def tube(self, a, b, zone):
        n = len(a)
        for i in range(n):
            self.face([a[i], a[(i + 1) % n], b[(i + 1) % n], b[i]], zone)

    def cap(self, ring, tip_or_none, zone, reverse=False):
        if tip_or_none is None:
            self.face(list(reversed(ring)) if reverse else ring, zone)
            return
        n = len(ring)
        for i in range(n):
            tri = [ring[i], ring[(i + 1) % n], tip_or_none]
            self.face(list(reversed(tri)) if reverse else tri, zone)

    def finish(self, name):
        bmesh.ops.recalc_face_normals(self.bm, faces=self.bm.faces)
        # io_pdx_mesh computes tangents, which Blender only does for tris/quads: no n-gons
        bmesh.ops.triangulate(self.bm, faces=self.bm.faces)
        mesh = bpy.data.meshes.new(name)
        self.bm.to_mesh(mesh)
        self.bm.free()
        return mesh


def build_spear(b, p):
    """Dory and sarissa. Shaft (zone 0, wood) from the butt to the head's base; leaf head and
    butt spike (zone 1, metal). Grip at z = 0, point at +Z."""
    r, sides = p["shaft_radius"], 8
    z0, tip = p["butt"], p["tip"]
    z1 = tip - p["head_length"]
    lo, hi = b.ring(z0, r, sides), b.ring(z1, r, sides)
    b.tube(lo, hi, 0)
    if p["spike"] > 0:
        b.cap(lo, b.vert(0, 0, z0 - p["spike"]), 1, reverse=True)  # sauroter
    else:
        b.cap(lo, None, 0, reverse=True)
    w, t = p["head_width"], max(0.06, r * 0.7)
    mid = z1 + (tip - z1) * 0.3
    blade = [b.vert(w, 0, mid), b.vert(0, t, mid), b.vert(-w, 0, mid), b.vert(0, -t, mid)]
    neck = b.ring(z1, r, 4, start=0.0)
    b.tube(neck, blade, 1)
    b.cap(blade, b.vert(0, 0, tip), 1)
    b.cap(hi, None, 0)


def build_sword(b, p):
    """Xiphos: leaf blade (zone 0) from the guard to the point at +Z; grip and pommel (zone 1)
    around the origin."""
    g, length, w = p["grip_length"], p["blade_length"], p["blade_width"]
    gr, half_t = 0.13, 0.07
    g0, g1 = -g * 0.5, g * 0.5
    grip_lo, grip_hi = b.ring(g0, gr, 6), b.ring(g1, gr, 6)
    b.tube(grip_lo, grip_hi, 1)
    b.cap(grip_lo, b.vert(0, 0, g0 - 0.25), 1, reverse=True)  # pommel
    b.cap(grip_hi, None, 1)
    gw, gt = p["guard_width"] * 0.5, 0.1  # guard: a flat bar across the blade's width (x)
    corners = [(-gw, -0.15), (gw, -0.15), (gw, 0.15), (-gw, 0.15)]
    lo = [b.vert(x, y, g1) for x, y in corners]
    hi = [b.vert(x, y, g1 + gt) for x, y in corners]
    b.tube(lo, hi, 0)
    b.cap(lo, None, 0, reverse=True)
    b.cap(hi, None, 0)
    base = g1 + gt
    # leaf: width along x, thickness along y; (fraction of length, width factor)
    sections = [(0.0, 0.7), (0.35, 0.8), (0.65, 1.0), (0.88, 0.6)]
    rings = [[b.vert(w * f * 0.5, 0, base + length * s), b.vert(0, half_t, base + length * s),
              b.vert(-w * f * 0.5, 0, base + length * s), b.vert(0, -half_t, base + length * s)]
             for s, f in sections]
    b.cap(rings[0], None, 0, reverse=True)
    for a, c in zip(rings, rings[1:]):
        b.tube(a, c, 0)
    b.cap(rings[-1], b.vert(0, 0, base + length), 0)


def build_shield(b, p):
    """Hoplon: a closed shell. Outer dome (zone 0, the face, planar UVs) and flat rim (zone 1)
    towards +X; the edge (zone 1); the inside (zone 2) back to ``back_x``. Face in the YZ plane
    around (centre_y, centre_z)."""
    radius, rim, depth, th = p["radius"], p["rim"], p["depth"], p["thickness"]
    bx, cy, cz = p["back_x"], p["centre_y"], p["centre_z"]
    rf = max(radius - rim, radius * 0.5)
    rings_n, sides = 6, 28
    xf = bx + th  # front plane of the rim

    def outer_x(r):
        return xf + depth * (1 - (r / rf) ** 2)

    def ring_at(r, x):
        return [b.vert(x, cy + r * math.sin(2 * math.pi * i / sides), cz + r * math.cos(2 * math.pi * i / sides))
                for i in range(sides)]

    def planar(q):  # seen from outside (+X) the viewer's right is PDX +Z (PDX space is mirrored), up is +Y
        return 0.5 + (q[2] - cz) / (2 * rf), 0.5 - (q[1] - cy) / (2 * rf)

    def quads(a, c, zone, uv=None):
        for i in range(sides):
            b.face([a[i], a[(i + 1) % sides], c[(i + 1) % sides], c[i]], zone, uv)

    def fan(centre, ring, zone, uv=None, reverse=False):
        for i in range(sides):
            tri = [centre, ring[i], ring[(i + 1) % sides]]
            b.face(list(reversed(tri)) if reverse else tri, zone, uv)

    radii = [rf * (i + 1) / rings_n for i in range(rings_n)]
    outer = [ring_at(r, outer_x(r)) for r in radii]
    inner = [ring_at(r, outer_x(r) - th) for r in radii]
    edge_front, edge_back = ring_at(radius, xf), ring_at(radius, bx)
    fan(b.vert(outer_x(0), cy, cz), outer[0], 0, planar)
    for a, c in zip(outer, outer[1:]):
        quads(a, c, 0, planar)
    quads(outer[-1], edge_front, 1)  # rim, front
    quads(edge_front, edge_back, 1)  # edge
    quads(edge_back, inner[-1], 2)  # rim, back
    for a, c in zip(reversed(inner), list(reversed(inner))[1:]):
        quads(a, c, 2)
    fan(b.vert(outer_x(0) - th, cy, cz), inner[0], 2, reverse=True)


def build_crest(b, p):
    """A horsehair fin (zone 0) on an arc over the helmet, in the head bone frame. Fore-aft: the
    arc runs along z; transverse: along x. ``centre_z`` moves it forward (-) or back (+)."""
    base, height = p["base"], p["height"]
    span, half = p["span_deg"], p["thickness"] * 0.5
    radius, steps = p["radius"], 10
    transverse = p.get("orientation") == "transverse"
    cz = p.get("centre_z", 0.0)

    def at(side, y, along):
        x, z = (along, side) if transverse else (side, along)
        return b.vert(x, y, z + cz)

    bottom, top = [], []
    for i in range(steps + 1):
        a = math.radians(-span + 2 * span * i / steps)
        along, y = radius * math.sin(a), base - radius * (1 - math.cos(a))
        h = height * (1 - 0.35 * abs(math.sin(a)))
        bottom.append((at(-half, y, along), at(half, y, along)))
        top.append((at(-half, y + h, along), at(half, y + h, along)))
    for i in range(steps):
        (bl0, br0), (bl1, br1) = bottom[i], bottom[i + 1]
        (tl0, tr0), (tl1, tr1) = top[i], top[i + 1]
        b.face([bl0, bl1, tl1, tl0], 0)   # one side
        b.face([br0, tr0, tr1, br1], 0)   # other side
        b.face([tl0, tl1, tr1, tr0], 0)   # top
        b.face([bl0, br0, br1, bl1], 0)   # bottom
    for (bl, br), (tl, tr) in ((bottom[0], top[0]), (bottom[-1], top[-1])):
        b.face([bl, tl, tr, br], 0)       # end caps


def build_ears(b, p):
    """Two horse ears in the head bone frame (the unit faces -Z): each a closed cone on an oval
    base, ``spread`` either side at height ``base``, tilted ``tilt_deg`` outwards and leaning
    ``lean_deg`` back. Faces turned to the front (-Z) take zone 1 (inner), the rest zone 0."""
    sides = 6
    tilt, lean = math.radians(p["tilt_deg"]), math.radians(p["lean_deg"])
    half_w, half_d = p["width"] * 0.5, p["depth"] * 0.5
    for s in (-1.0, 1.0):
        cx, cy, cz = s * p["spread"], p["base"], p.get("centre_z", 0.0)
        d = (s * math.sin(tilt), math.cos(tilt) * math.cos(lean), math.sin(lean))
        tip = b.vert(cx + d[0] * p["length"], cy + d[1] * p["length"], cz + d[2] * p["length"])
        ring = [b.vert(cx + half_w * math.cos(a) * math.cos(tilt), cy - s * half_w * math.cos(a) * math.sin(tilt),
                       cz + half_d * math.sin(a))
                for a in (2 * math.pi * i / sides for i in range(sides))]
        for i in range(sides):
            a, c = ring[i], ring[(i + 1) % sides]
            front = math.sin(2 * math.pi * (i + 0.5) / sides) < 0  # this face's side of the oval is towards -Z
            b.face([a, c, tip], 1 if front else 0)
        b.face(list(reversed(ring)), 0)


RIM_DROP = {  # style -> how far the rim hangs below the crown at the front, the side and the back
    "attic": (0.75, 2.05, 1.45),       # the brow, cheek guards to the jaw, a short neck guard
    "corinthian": (2.2, 2.15, 1.5),    # the face covered (eye slot: a zone-2 band), cheek and neck guards
    "pilos": (1.05, 1.05, 1.05),       # a level rim
}
SLOT = (1.15, 1.45)  # corinthian eye slot: this far below the crown, at the front


def build_helmet(b, p):
    """A Greek helmet in the head bone frame (the unit faces -Z): a bronze shell with thickness. A
    dome (an ellipsoid fitted to the bare head: ``top``, ``half_width``, ``front``/``back``) down to
    its equator, then a skirt that drops nearly straight to a rim whose depth varies around the
    head by style (``RIM_DROP``); pilos raises the crown into a cone; corinthian reaches further
    forward to cover the nose. An integrated crest (zone 1) runs fore-aft or across the crown.
    Zone 2: the eye slot (corinthian) and the inner lining."""
    n_az, dome_n, skirt_n = 12, 4, 3
    style = p.get("style", "attic")
    front = p["front"] - (0.22 if style == "corinthian" else 0.0)
    rx = p["half_width"]
    cz = (front + p["back"]) / 2
    rz = (p["back"] - front) / 2
    ry = rx * 1.05
    cy = p["top"] - ry
    th = p["thickness"]
    drops = RIM_DROP[style]

    def rim_y(phi):  # phi 0 = front (-Z), pi/2 = the side, pi = back
        c = math.cos(phi)
        f, sd, bk = drops
        return p["top"] - (sd + (f - sd) * c if c >= 0 else sd + (bk - sd) * -c)

    def point(phi, theta, inset):
        """Dome point at azimuth phi, elevation theta (0 = crown, 90 degrees = the equator), ``inset`` inwards."""
        sx, sy, sz = rx - inset, ry - inset, rz - inset
        st = math.sin(theta)
        y = cy + sy * math.cos(theta)
        if style == "pilos" and theta < math.radians(70):
            y += (1 - theta / math.radians(70)) * ry * 0.9  # the cone
        return (sx * st * math.sin(phi), y, cz - sz * st * math.cos(phi))

    def column(phi, inset):
        """The points from just below the crown down to the rim at azimuth phi."""
        bottom = rim_y(phi)
        if bottom >= cy:  # the rim is on the dome
            t_rim = math.acos(max(-1.0, min(1.0, (bottom - cy) / ry)))
            n = dome_n + skirt_n
            return [point(phi, t_rim * j / n, inset) for j in range(1, n + 1)]
        pts = [point(phi, math.radians(90) * j / dome_n, inset) for j in range(1, dome_n + 1)]
        ex, _, ez = pts[-1]
        for j in range(1, skirt_n + 1):
            f = j / skirt_n
            taper = 1 - 0.06 * f
            pts.append((ex * taper, cy + (bottom - cy) * f, cz + (ez - cz) * taper))
        return pts

    phis = [2 * math.pi * i / n_az for i in range(n_az)]
    crown_o = b.vert(*point(0, 0, 0))
    crown_i = b.vert(*point(0, 0, th))
    outer = [[b.vert(*q) for q in column(phi, 0)] for phi in phis]
    inner = [[b.vert(*q) for q in column(phi, th)] for phi in phis]
    rows = dome_n + skirt_n

    eye = p["top"] - (SLOT[0] + SLOT[1]) / 2

    def slot_row(i):  # the front columns' row nearest eye height (the slot is narrower than a row)
        return min(range(rows - 1), key=lambda j: abs((b.pdx[outer[i][j]][1] + b.pdx[outer[i][j + 1]][1]) / 2 - eye))

    def zone(i, j):
        if style != "corinthian" or math.cos(phis[i]) < 0.55:
            return 0
        return 2 if j == slot_row(i) else 0

    for i in range(n_az):
        k = (i + 1) % n_az
        b.face([crown_o, outer[i][0], outer[k][0]], 0)
        b.face([crown_i, inner[k][0], inner[i][0]], 2)
        for j in range(rows - 1):
            b.face([outer[i][j], outer[i][j + 1], outer[k][j + 1], outer[k][j]], zone(i, j))
            b.face([inner[i][j], inner[k][j], inner[k][j + 1], inner[i][j + 1]], 2)
        b.face([outer[i][-1], inner[i][-1], inner[k][-1], outer[k][-1]], 0)  # the rim's edge
    if p.get("crest", "fore_aft") != "none":
        transverse = p.get("crest") == "transverse"
        steps, half, h = 8, 0.12, p["crest_height"]
        span = math.radians(75)
        rows = []
        for i in range(steps + 1):
            a = -span + 2 * span * i / steps
            base = point(math.pi / 2 if transverse else 0.0, abs(a), -0.02)
            along = (base[0] if transverse else base[2])
            if a < 0:
                along = -along
            hh = h * (1 - 0.35 * abs(math.sin(a)))
            def at(side, up, _along=along, _y=base[1]):
                return b.vert(_along, _y + up, side) if transverse else b.vert(side, _y + up, _along)
            rows.append((at(-half, 0), at(half, 0), at(-half, hh), at(half, hh)))
        for (bl0, br0, tl0, tr0), (bl1, br1, tl1, tr1) in zip(rows, rows[1:]):
            b.face([bl0, bl1, tl1, tl0], 1)
            b.face([br0, tr0, tr1, br1], 1)
            b.face([tl0, tl1, tr1, tr0], 1)
            b.face([bl0, br0, br1, bl1], 1)
        for bl, br, tl, tr in (rows[0], rows[-1]):
            b.face([bl, tl, tr, br], 1)


BUILDERS = {"spear": build_spear, "sword": build_sword, "shield": build_shield, "crest": build_crest, "ears": build_ears,
            "helmet": build_helmet}


SHARP_ANGLE = 50.0  # degrees: sharper turns stay hard edges (blade edges, rims); the rest shades smooth


def smooth_normals(mesh):
    """Q4: smooth shading with hard edges only where two colour zones meet or the surface turns
    sharper than SHARP_ANGLE (the parts were flat-shaded: every face lit on its own)."""
    mesh.shade_smooth()
    mesh.set_sharp_from_angle(angle=math.radians(SHARP_ANGLE))
    zone = mesh.attributes.get("zone")
    if zone is None:
        return
    sharp = mesh.attributes.get("sharp_edge") or mesh.attributes.new("sharp_edge", "BOOLEAN", "EDGE")
    faces_of = {}
    for poly in mesh.polygons:
        for key in poly.edge_keys:
            faces_of.setdefault(key, []).append(zone.data[poly.index].value)
    for e in mesh.edges:
        zs = faces_of.get(tuple(sorted(e.vertices)), [])
        if len(set(zs)) > 1:
            sharp.data[e.index].value = True


def unwrap_atlas(obj, density):
    """Q1: a real UV unwrap (Smart UV Project, islands packed with a margin) and the texture size
    (a power of two, 64-512) that gives about ``density`` texels per mesh unit. Returns
    {"size", "density", "tris": [[zone, u0, v0, u1, v1, u2, v2], ...]} in image coordinates (v down)."""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.03)
    bpy.ops.object.mode_set(mode="OBJECT")
    me = obj.data
    uv = me.uv_layers["map1"].data
    zone = me.attributes.get("zone")
    area3, area_uv, tris = 0.0, 0.0, []
    for poly in me.polygons:
        pts = [uv[li].uv for li in poly.loop_indices]
        area3 += poly.area
        area_uv += abs((pts[1].x - pts[0].x) * (pts[2].y - pts[0].y) - (pts[2].x - pts[0].x) * (pts[1].y - pts[0].y)) / 2
        z = zone.data[poly.index].value if zone is not None else 0
        tris.append([z] + [c for p in pts for c in (p.x, 1.0 - p.y)])
    ideal = density * math.sqrt(area3 / area_uv) if area_uv > 0 else 64
    size = 64
    while size < 512 and size * 1.41 < ideal:
        size *= 2
    return {"size": size, "density": size * math.sqrt(area_uv / area3) if area3 else 0.0, "tris": tris}


def main() -> None:
    a = _pdx.args()
    ie = _pdx.io_module()
    out = []
    for part in a["parts"]:
        _pdx.clear_scene()
        params = part.get("params", {})
        builder = Builder(part["zone_uvs"], flip_z=params.get("point") == "down")
        BUILDERS[part["builder"]](builder, params)
        mesh = builder.finish(part["shape"])
        smooth_normals(mesh)
        obj = bpy.data.objects.new(part["shape"].removesuffix("Shape"), mesh)
        bpy.context.scene.collection.objects.link(obj)
        mesh.materials.append(_pdx.pdx_material(part["shape"] + "_mat", part.get("shader", "PdxMeshStandard"),
                                                part["textures"]))
        atlas = unwrap_atlas(obj, part.get("density", 32.0)) if part.get("layout") == "atlas" else None
        ie.export_meshfile(part["out_mesh"], exp_mesh=True, exp_skel=False, exp_locs=False)
        out.append({"kind": part["kind"], "out_mesh": part["out_mesh"], "verts": len(mesh.vertices),
                    "faces": len(mesh.polygons), **({"atlas": atlas} if atlas else {})})
    _pdx.write_result(a, {"parts": out})


main()
