"""fresh_horse.py's abandoned sources (Track C procedural/anatomy horses, Track M fitted masters; tests 15-24),
removed from the live job in Track P3. Reference only: not importable as is."""

class Readability:
    """The round 6 rule's settings and measurements (fresh_horse args "readability")."""

    def __init__(self, cfg, van, barding):
        cfg = cfg or {}
        self.auto = cfg.get("mode", "auto") == "auto"
        self.share = float(cfg.get("leg_share", hr.LEG_SHARE))
        self.depth_floor = float(cfg.get("depth_floor", DEPTH_FLOOR))
        self.hem_steps = [float(h) for h in cfg.get("hem_steps", HEM_STEPS)]
        self.bones = van["bones"]
        vco = np.array(van["p"], dtype=np.float64).reshape(-1, 3)
        vtri = np.array(van["tri"], dtype=np.int64).reshape(-1, 3)
        self.vanilla_arrays = (vco, vtri)
        self.vanilla = hr.leg_visible(vco, vtri, self.bones)["length"]
        self.vanilla_depth = None
        self.barding = barding if barding and barding.get("style") == "drape" else None
        self.log = {"mode": "auto" if self.auto else "reference", "leg_share": self.share,
                    "vanilla_leg": self.vanilla, "steps": []}

    def depth_ratio(self, depth):
        if self.vanilla_depth is None:
            import _horse_measure as hm
            v = self.vanilla_arrays
            self.vanilla_depth = hm.summary(hm.proportions(v[0], v[1], self.bones))["depth"]
        return depth / self.vanilla_depth

    def legs(self, co, tri, part=None, hem_length=None, extra=()):
        """Visible leg length with the drape (at ``hem_length``, default its own) hung on this body,
        and the ``extra`` pieces (_horse_barding arrays: the stirrups, the tail) beside it."""
        co, tri = np.asarray(co, dtype=np.float64), np.asarray(tri, dtype=np.int64)
        part = np.zeros(len(tri), dtype=np.int64) if part is None else np.asarray(part)
        pieces = list(extra)
        if self.barding is not None:
            kw = {k: v for k, v in self.barding.items() if k != "style"}
            if hem_length is not None:
                kw["hem_length"] = hem_length
            pieces.append(hb.drape(co, self.bones, **kw))  # on the body alone, as add_barding hangs it
        for b in pieces:
            co, tri = np.vstack([co, b["v"]]), np.vstack([tri, np.asarray(b["f"]) + len(co)])
            part = np.concatenate([part, np.asarray(b["part"])])
        return hr.leg_visible(co, tri, self.bones, part)["length"]

    def passes(self, length):
        return length >= self.share * self.vanilla

    def step(self, **kw):
        kw["leg_ratio"] = round(kw["leg"] / self.vanilla, 3)
        kw["passes"] = self.passes(kw["leg"])
        self.log["steps"].append(kw)
        return kw["passes"]


def bounds(obj):
    vs = [v.co for v in obj.data.vertices]
    lo = Vector((min(v.x for v in vs), min(v.y for v in vs), min(v.z for v in vs)))
    hi = Vector((max(v.x for v in vs), max(v.y for v in vs), max(v.z for v in vs)))
    return lo, hi


def fit(obj, vanilla):
    vlo, vhi = bounds(vanilla)
    lo, hi = bounds(obj)
    s = (vhi.z - vlo.z) / max(hi.z - lo.z, 1e-6)
    obj.data.transform(Matrix.Scale(s, 4))
    lo, hi = bounds(obj)
    off = Vector((-(lo.x + hi.x) / 2, vlo.y - lo.y, vlo.z - lo.z))
    obj.data.transform(Matrix.Translation(off))
    obj.data.update()
    return s, list(off)


def scale_height(obj, vanilla, bones):
    """Height and width scaled so the croup (the back 3-4 behind Saddle_Node, |x| < 1) is vanilla's:
    the box fit gets the length right, but the generated horse stands lower (side profiles)."""
    z = bones["Saddle_Node"][2]
    s = top_at(vanilla, z + 3.0, z + 4.0, 1.0) / top_at(obj, z + 3.0, z + 4.0, 1.0)
    for v in obj.data.vertices:
        v.co.x *= s
        v.co.y *= s
    obj.data.update()
    return s


def solidify(obj, voxel):
    """A voxel remesh into one closed manifold solid (thin sheets thickened, inner faces gone),
    largest piece kept. Its input must be closed: holes let the inside leak and the body vanishes."""
    activate(obj)
    mod = obj.modifiers.new("solid", "REMESH")
    mod.mode, mod.voxel_size = "VOXEL", float(voxel)
    bpy.ops.object.modifier_apply(modifier=mod.name)
    out = {"voxel": voxel, "tris": tri_count(obj), "pieces": count_pieces(obj)}
    keep_largest(obj)
    out["tris_kept"] = tri_count(obj)
    out["normals_flipped"] = outward(obj)
    return out


def outward(obj):
    """Face normals made consistent and pointing out of the (closed) solid; booleans and the
    voxel remesh read inside-out meshes inverted. -> whether they had pointed in."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    before = sum(f.normal.dot(f.calc_center_median()) for f in bm.faces)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    after = sum(f.normal.dot(f.calc_center_median()) for f in bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return (before > 0) != (after > 0)


def boolean(obj, cutter, operation="DIFFERENCE"):
    activate(obj)
    mod = obj.modifiers.new("bool", "BOOLEAN")
    mod.operation, mod.object, mod.solver = operation, cutter, "EXACT"
    mod.use_self, mod.use_hole_tolerant = True, True  # cutters made of overlapping capsules
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter, do_unlink=True)


def cut_rider_box(obj, cut):
    """The rider removed with a boolean: a box above ``cut.y`` between ``cut.z0`` and ``cut.z1``
    (head, neck and tail stay). On a closed solid the result is closed, with a flat cap at
    exactly ``cut.y``: the back the rider sits on."""
    before = tri_count(obj)
    boolean(obj, box("rider_cutter", (-20, cut["y"], cut["z0"]), (20, 60, cut["z1"])))
    return before - tri_count(obj)


def trim_skirts(obj, bones, hem_y, leg_radius):
    """Skirts and caparison end at ``hem_y`` (above the knees and hocks) so the legs swing free:
    a boolean cut of everything below the hem except capsules of ``leg_radius`` round the vanilla
    leg chains (the legs are on them after the reshape) and round the hanging tail."""
    B = bones
    cutter = box("skirt_cutter", (-20, -5, -40), (20, hem_y, 40))
    keep = []
    for leg in hs.LEGS:
        pts = hs.leg_chain({k: np.array(v) for k, v in B.items()}, leg)
        for p0, p1 in zip(pts[:-1], pts[1:]):
            if p0[1] < hem_y + 1:
                keep += capsule("leg_keep", p0.tolist(), p1.tolist(), leg_radius)
    tail = [B[t] for t in ("HorseTailMiddle", "HorseTailTip", "HorseTailEnd")]
    for p0, p1 in zip(tail[:-1], tail[1:]):
        keep += capsule("tail_keep", p0, p1, 0.6)
    keeper = join(keep)
    for o in (keeper, cutter):
        activate(o)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    boolean(cutter, keeper)
    before = tri_count(obj)
    boolean(obj, cutter)
    return before - tri_count(obj)


def clear_tail(obj, bones, gap):
    """The caparison pushed out to ``gap`` beyond the hanging tail (TailMiddle down; tail radius
    0.45): every vertex off the centre line (|x| > 0.45) closer than that moves straight away from
    the tail, so the tail and the caparison stay apart after the voxel solid."""
    B = {k: np.array(v) for k, v in bones.items()}
    co = hs.verts(obj)
    dist, point = hs.nearest_on_chains(co, [hs.chain(B, ["HorseTailMiddle", "HorseTailTip", "HorseTailEnd"])])
    move = (dist < 0.45 + gap) & (np.abs(co[:, 0]) > 0.45)
    away = co[move] - point[move]
    norm = np.linalg.norm(away, axis=1, keepdims=True)
    co[move] = point[move] + away / np.maximum(norm, 1e-6) * (0.45 + gap)
    hs.set_verts(obj, co)
    return int(move.sum())


def surface_top(obj, z, half=0.5, x=0.0):
    """The highest surface point straight below y 60 over (x, z +- ``half``): rays cast down on
    the faces (vertex samples miss a flat cap made of large faces). None if nothing is hit."""
    from mathutils.bvhtree import BVHTree
    bvh = BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
    hits = []
    for dz in (-half, -half / 2, 0.0, half / 2, half):
        loc = bvh.ray_cast(Vector((x, 60.0, z + dz)), Vector((0, -1, 0)))[0]
        if loc is not None:
            hits.append(loc.y)
    return max(hits) if hits else None


def match_saddle(obj, vanilla, bones, tolerance=0.05, limit=1.0, also=()):
    """Our back at the saddle (surface height on the centre line within 0.5 of Saddle_Node along
    z) onto vanilla's, by a smooth vertical shift of the back (full within 1.5 of the saddle
    along z, none from 3.5; only above y 9). Saddle_Node itself never moves. A shift over
    ``limit`` is refused (something else is wrong). The same shift field moves ``also`` (the
    bake source: it must keep matching the low mesh). -> (before, after, vanilla, note)."""
    z = bones["Saddle_Node"][2]
    want = surface_top(vanilla, z)
    before = surface_top(obj, z)
    if before is None or abs(want - before) > limit:
        return before, before, want, f"refused: a shift of {want - (before or 0):.2f} is over {limit}"
    for _ in range(4):
        delta = want - surface_top(obj, z)
        if abs(delta) <= tolerance / 2:
            break
        for o in (obj, *also):
            for v in o.data.vertices:
                if v.co.y > 9.0:
                    w = 1.0 - _smooth((abs(v.co.z - z) - 1.5) / 2.0)
                    v.co.y += delta * w * _smooth((v.co.y - 9.0) / 1.5)
            o.data.update()
    return before, surface_top(obj, z), want, "matched"


def _smooth(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def fill_holes(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    edges = [e for e in bm.edges if e.is_boundary]
    before = len(edges)
    if edges:
        res = bmesh.ops.holes_fill(bm, edges=edges, sides=0)
        bmesh.ops.triangulate(bm, faces=res["faces"])
    bm.normal_update()
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return before


def capsule(name, a, b, r):
    a, b = Vector(a), Vector(b)
    d = b - a
    length = max(d.length, 1e-3)
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=r, depth=length, location=(a + b) / 2)
    cyl = bpy.context.active_object
    cyl.rotation_mode = "QUATERNION"
    cyl.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d.normalized())
    parts = [cyl]
    for c in (a, b):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=6, radius=r, location=c)
        parts.append(bpy.context.active_object)
    return parts


def read_obj(path):
    """(positions, triangles) in PDX space from a preview-axes OBJ (as debug_dump writes them)."""
    co, tris = [], []
    for line in open(path, encoding="utf-8"):
        if line.startswith("v "):
            x, y, z = map(float, line.split()[1:4])
            co.append((x, y, -z))
        elif line.startswith("f "):
            idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            tris += [(idx[0], idx[i], idx[i + 1]) for i in range(1, len(idx) - 1)]
    return np.array(co, dtype=np.float64), np.array(tris, dtype=np.int64)


def procedural_horse(van, src):
    """Bake-off horse B: _horse_proc's tubes on the vanilla skeleton, fitted to the reference
    model (``fit_from``: a preview-axes OBJ of it on the vanilla rest pose, e.g. a model build's
    3_rider_cut stage), joined into one closed surface by a voxel remesh (``voxel``), relaxed
    (``relax``). -> (object, fit log)."""
    import _horse_proc as hp
    vco = np.array(van["p"], dtype=np.float64).reshape(-1, 3)
    vtri = np.array(van["tri"], dtype=np.int64).reshape(-1, 3)
    fit = {"ratios": None, "raw": None, "topline": None}
    if src.get("fit_from"):
        rco, rtri = read_obj(src["fit_from"])
        fit.update(hp.fit_ratios(rco, rtri, vco, vtri, van["bones"]))
        fit["topline"] = hp.topline(rco, rtri, vco, vtri, van["bones"])
    # the template's reins and tail are left out of the samples (by their dominant bone)
    skip = {"Rem", "Stigbyglar", *hs.TAIL}
    keep = []
    for t in vtri:
        score = {}
        for vi in t:
            for bone, w in van["weights"][vi].items():
                score[bone] = score.get(bone, 0.0) + w
        keep.append(max(score, key=score.get) not in skip if score else True)
    built = hp.build(vco, vtri, van["bones"], fit["ratios"], fit["topline"], tri_keep=np.array(keep))
    objs = [mesh_object(f"proc_{i}", v.tolist(), t.tolist()) for i, (v, t) in enumerate(built["tubes"])]
    obj = join(objs)
    solidify(obj, float(src.get("voxel", 0.15)))
    if src.get("relax", 6):
        activate(obj)
        mod = obj.modifiers.new("relax", "SMOOTH")
        mod.factor, mod.iterations = 0.5, int(src.get("relax", 6))
        bpy.ops.object.modifier_apply(modifier=mod.name)
    obj.name = "procedural_horse"
    return obj, {"scale": 1.0, "offset": [0, 0, 0], "note": "built on the skeleton", **fit, "build": built["log"]}


def shallower(ha, bones, seat, params, read):
    """The readability rule's first step (round 6): while the legs (with the caparison hung at its
    own hem) don't show enough, the barrel 0.05 x vanilla's depth shallower, never under
    ``read.depth_floor``; measured on the anatomy's tubes, before the remesh. -> (params, built)."""
    params = dict(params)
    for _ in range(8):
        built = ha.build(bones, seat, params)
        co, tri = ha._joined(built["tubes"])
        ratio = read.depth_ratio(ha.measure(built["tubes"], bones)["depth"])
        ok = read.step(stage="depth", depth=round(ratio, 3), barrel_depth=round(built["params"]["barrel_depth"], 4),
                       leg=read.legs(co, tri))
        if ok or ratio <= read.depth_floor + 0.005:
            return params, built
        want = max(read.depth_floor, ratio - 0.05)
        params["barrel_depth"] = built["params"]["barrel_depth"] * want / ratio
    return params, built


def anatomy_horse(van, vanilla, src, read=None):
    """Bake-off horse B, round 5: _horse_anatomy's own parametric horse on the vanilla skeleton
    (from vanilla only the bones and the seat height: its back's surface at Saddle_Node), its
    parameters driven by the reference model (``fit_from``: a preview-axes OBJ of it on the vanilla
    rest pose) and ``params`` overrides, joined by a voxel remesh (``voxel``) and relaxed
    (``relax``); with ``read`` (a Readability in auto mode) made shallower first (``shallower``).
    -> (object, fit log)."""
    import _horse_anatomy as ha
    bones = van["bones"]
    seat = surface_top(vanilla, bones["Saddle_Node"][2])
    params, fit = src.get("params") or {}, {"seat": seat}
    if src.get("fit_from"):
        rco, rtri = read_obj(src["fit_from"])
        fit.update(ha.fit(bones, seat, rco, rtri, params))
        params = fit["params"]
    if read is not None and read.auto:
        params, built = shallower(ha, bones, seat, params, read)
    else:
        built = ha.build(bones, seat, params)
    fit["achieved"] = ha.measure(built["tubes"], bones)
    objs = [mesh_object(f"anat_{i}", v.tolist(), t.tolist()) for i, (v, t) in enumerate(built["tubes"])]
    obj = join(objs)
    solidify(obj, float(src.get("voxel", 0.15)))
    if src.get("relax", 4):
        activate(obj)
        mod = obj.modifiers.new("relax", "SMOOTH")
        mod.factor, mod.iterations = 0.5, int(src.get("relax", 4))
        bpy.ops.object.modifier_apply(modifier=mod.name)
    obj.name = "anatomy_horse"
    fit["params"] = {k: round(v, 4) for k, v in built["params"].items()}
    return obj, {"scale": 1.0, "offset": [0, 0, 0], "note": "the app's own anatomy on the skeleton", **fit,
                 "build": built["log"]}


def control_horse(bones):
    """Capsules along the vanilla bones (radius by body part), joined and voxel-remeshed."""
    B = {k: Vector(v) for k, v in bones.items()}
    parts = []
    body_a = B["HorseSpineEnd"] + Vector((0, -1.6, 0.6))
    body_b = B["HorseSpineTop"] + Vector((0, -1.8, -1.2))
    parts += capsule("body", body_a, body_b, 2.7)
    parts += capsule("neck", B["HorseNeckBase"] + Vector((0, -0.8, 0)), B["HorseHead"], 1.25)
    head_tip = B["HorseHead"] + Vector((0, -2.6, -3.0))
    parts += capsule("head", B["HorseHead"], head_tip, 0.95)
    for side in ("Left", "Right"):
        for end in ("Front", "Back"):
            chain = [f"Horse{end}{side}{p}" for p in ("Up", "Down", "Knee", "Heel")]
            pts = [B[c] for c in chain] + [Vector((B[chain[-1]].x, 0.15, B[chain[-1]].z))]
            for i in range(len(pts) - 1):
                parts += capsule(f"leg{i}", pts[i], pts[i + 1], 0.75 if i == 0 else 0.42)
    tail = [B[t] for t in ("HorseTailTop", "HorseTailMiddle", "HorseTailTip", "HorseTailEnd")]
    for i in range(len(tail) - 1):
        parts += capsule(f"tail{i}", tail[i], tail[i + 1], 0.45 - 0.08 * i)
    activate(parts[0])
    for p in parts:
        p.select_set(True)
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    mod = obj.modifiers.new("remesh", "REMESH")
    mod.mode, mod.voxel_size = "VOXEL", 0.22
    bpy.ops.object.modifier_apply(modifier=mod.name)
    obj.name = "control_horse"
    return obj


def decimate(obj, target, log=None, quads=None):
    """Down to about ``target`` triangles with even-sized faces: QuadriFlow quads (target / 2),
    split into triangles. Collapse decimation leaves long slivers across flat areas, and an edge
    spanning several bones stretches badly when they bend (test 15, control horse). QuadriFlow
    needs a closed manifold input (a voxel remesh gives one) and otherwise does nothing; the
    final collapse then does all the work, which ``quadriflow_ran`` reports."""
    activate(obj)
    log = log if log is not None else {}
    log["before_quadriflow"] = tri_count(obj)
    status = bpy.ops.object.quadriflow_remesh(target_faces=max(100, int(quads or target // 2)), use_preserve_sharp=False,
                                              use_preserve_boundary=False, use_mesh_symmetry=False,
                                              smooth_normals=True, mode="FACES", seed=0)
    log["quadriflow"] = sorted(status)
    log["after_quadriflow"] = tri_count(obj)
    log["quadriflow_ran"] = log["after_quadriflow"] != log["before_quadriflow"]  # it refuses non-manifold input silently
    mod = obj.modifiers.new("tri", "TRIANGULATE")
    bpy.ops.object.modifier_apply(modifier=mod.name)
    if tri_count(obj) > target:  # QuadriFlow lands near, not on, the target
        mod = obj.modifiers.new("trim", "DECIMATE")
        mod.decimate_type, mod.ratio = "COLLAPSE", target / tri_count(obj)
        bpy.ops.object.modifier_apply(modifier=mod.name)
    return tri_count(obj)


def add_chanfron(obj, bones, lift=0.06, thickness=0.05, start=0.12, stop=0.86, width=0.78, nose=0.45,
                 rows=8, cols=6, corner=0.35, facing=None):
    """A chanfron merged into the mesh as its own piece (part = 1; 100% HorseHead after the
    transfer): a plate made from scratch, not the head's own faces (those gave a ragged outline,
    test 16). The head's axis runs from HorseHead to the muzzle (the body point furthest along
    vanilla's poll-to-muzzle direction); the plate covers ``start``..``stop`` of it, ``rows`` x
    ``cols`` points, each cast down onto the face from above and lifted ``lift`` off it. Its
    half-width is ``width`` x the head's half-width there (rays across), narrowing to ``nose`` x
    at the muzzle end, the corners rounded (``corner``); ``thickness`` with a rim.
    -> its triangle count."""
    from mathutils.bvhtree import BVHTree
    head = Vector(bones["HorseHead"])
    axis0 = Vector((0.0, -1.5, -3.3)).normalized()
    co = hs.verts(obj)
    rel = co - np.array(head)
    tip = Vector(co[int(np.argmax(rel @ np.array(axis0)))])
    axis = tip - head
    length = axis.length
    axis.normalize()
    side = Vector((1.0, 0.0, 0.0))
    up = axis.cross(side).normalized()
    if up.y < 0:
        up = -up
    bvh = BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
    grid = []
    for i in range(rows):
        u = start + (stop - start) * i / (rows - 1)
        c = head + axis * (u * length)
        hit = bvh.ray_cast(c + side * 4.0, -side, 8.0)[0]
        half = max((hit - c).dot(side), 0.2) if hit is not None else 0.6
        taper = width + (nose - width) * ((u - start) / (stop - start)) ** 1.5
        row = []
        for j in range(cols):
            v = -1.0 + 2.0 * j / (cols - 1)
            # rounded corners: the outer columns pull in at the two ends
            e = min((u - start) / (stop - start), (stop - u) / (stop - start)) / corner
            pull = 1.0 if abs(v) < 0.99 else 0.75 + 0.25 * min(e, 1.0)
            pos = c + side * (v * half * taper * pull)
            h = bvh.ray_cast(pos + up * 4.0, -up, 8.0)
            if h[0] is None:
                row.append(None)
                continue
            row.append(h[0] + h[1] * lift)
        grid.append(row)
    bm = bmesh.new()
    vs = [[bm.verts.new(p) if p is not None else None for p in row] for row in grid]
    for i in range(rows - 1):
        for j in range(cols - 1):
            q = [vs[i][j], vs[i][j + 1], vs[i + 1][j + 1], vs[i + 1][j]]
            if all(x is not None for x in q):
                bm.faces.new(q)
    bm.normal_update()
    if bm.faces and sum(f.normal.dot(up) for f in bm.faces) < 0:
        bmesh.ops.reverse_faces(bm, faces=list(bm.faces))
    me = bpy.data.meshes.new("chanfron")
    bm.to_mesh(me)
    bm.free()
    ch = bpy.data.objects.new("chanfron", me)
    bpy.context.scene.collection.objects.link(ch)
    activate(ch)
    mod = ch.modifiers.new("solid", "SOLIDIFY")
    mod.thickness, mod.offset, mod.use_rim = thickness, 1.0, True
    bpy.ops.object.modifier_apply(modifier=mod.name)
    mod = ch.modifiers.new("tri", "TRIANGULATE")
    bpy.ops.object.modifier_apply(modifier=mod.name)
    tris = tri_count(ch)
    for o, value in ((obj, 0), (ch, 1)):
        if o.data.attributes.get("part") is None:
            attr = o.data.attributes.new("part", "INT", "FACE")
            attr.data.foreach_set("value", [value] * len(o.data.polygons))
    activate(obj)
    ch.select_set(True)
    bpy.ops.object.join()
    return tris


def shorter_hem(obj, barding, read, extra=()):
    """The readability rule's second step (round 6): the drape's hem_length, from its own along
    ``read.hem_steps`` (each shorter), the first whose legs show enough on this body (the reduced
    one, with its chanfron, and the ``extra`` pieces still to join: _horse_barding arrays, e.g. the
    stirrups and the tail); in reference mode, or when none does, the shortest tried. -> hem_length."""
    own = float(barding.get("hem_length", 0.25))
    tri, part = [], []
    for poly, p in zip(obj.data.polygons, face_parts(obj)):
        v = poly.vertices
        tri += [(v[0], v[i], v[i + 1]) for i in range(1, len(v) - 1)]
        part += [p] * (len(v) - 2)
    co = hs.verts(obj)
    hem = own
    for hem in [own] + [h for h in read.hem_steps if h < own]:
        if read.step(stage="hem", hem_length=hem, leg=read.legs(co, tri, part, hem, extra)) or not read.auto:
            break
    return hem


def add_barding(obj, bones, barding, stirrups=None):
    """Round 3: the hanging panels and the stirrups (_horse_barding) joined in as their own
    pieces, with face attributes ``part`` and ``hem`` and a point attribute ``anchor`` (where each
    vertex takes its body weights from). -> the builders' logs."""
    body = hs.verts(obj)
    style = barding.get("style", "slab")
    kw = {k: v for k, v in barding.items() if k != "style"}
    built = [hb.drape(body, bones, **kw) if style == "drape" else hb.panels(body, bones, **kw)]
    if stirrups:
        built.append(hb.stirrups(**stirrups))
    if obj.data.attributes.get("part") is None:
        attr = obj.data.attributes.new("part", "INT", "FACE")
        attr.data.foreach_set("value", [0] * len(obj.data.polygons))
    log = {}
    for name, b in zip(("panels", "stirrups"), built):
        join_piece(obj, name, b)
        log[name] = b["log"]
    return log


def cut_tail_stub(obj, bones, radius):
    """A generated tail stub cut off (a capsule from TailMiddle past TailEnd: behind the rump, so
    the rump itself stays) before our own tail is built in its place. -> triangles before, after
    (the cut faces can make it more)."""
    B = {k: Vector(v) for k, v in bones.items()}
    end = B["HorseTailEnd"] + (B["HorseTailEnd"] - B["HorseTailTip"]).normalized() * 1.5
    before = tri_count(obj)
    boolean(obj, join(capsule("tail_cutter", B["HorseTailMiddle"], end, radius)))
    keep_largest(obj)
    return [before, tri_count(obj)]


def pin_chanfron(obj, bone="HorseHead"):
    """Every vertex of a part-1 face weighted 100% to ``bone``. -> how many."""
    parts = face_parts(obj)
    pinned = {v for poly, part in zip(obj.data.polygons, parts) if part == 1 for v in poly.vertices}
    group = obj.vertex_groups.get(bone) or obj.vertex_groups.new(name=bone)
    for vi in pinned:
        for g in list(obj.data.vertices[vi].groups):
            obj.vertex_groups[g.group].remove([vi])
        group.add([vi], 1.0, "REPLACE")
    return len(pinned)


def paint_textures(obj, bones, palette, hem_y, out_dir, name, size):
    """Step 8: the textures (_horse_paint) as PNGs in ``out_dir``. -> paths and coverage."""
    import _horse_paint as hp
    me = obj.data
    me.calc_loop_triangles()
    uv = me.uv_layers.active.data
    parts = face_parts(obj)
    co = hs.verts(obj)
    tris = np.array([lt.vertices[:] for lt in me.loop_triangles])
    uvs = np.array([[uv[li].uv[:] for li in lt.loops] for lt in me.loop_triangles])
    normals = np.array([lt.normal[:] for lt in me.loop_triangles])
    centres = co[tris].mean(axis=1)
    part = np.array([parts[lt.polygon_index] for lt in me.loop_triangles])
    hem_attr = me.attributes.get("hem")
    face_hem = None
    if hem_attr is not None:
        vals = [0.0] * len(me.polygons)
        hem_attr.data.foreach_get("value", vals)
        face_hem = np.array([vals[lt.polygon_index] for lt in me.loop_triangles])
    g = obj.vertex_groups.get("HorseHead")
    head_w = np.zeros(len(me.vertices))
    if g is not None:
        for v in me.vertices:
            for e in v.groups:
                if e.group == g.index:
                    head_w[v.index] = e.weight
    region = hp.regions(centres, normals, part, bones, hem_y, face_hem, head_w[tris].mean(axis=1))
    diff, spec, normal, coverage = hp.paint(uvs, tris, co, normals, region, palette, bones, size)
    paths = {}
    for kind, arr in (("diffuse", diff), ("specular", spec), ("normal", normal)):
        h, w = arr.shape[:2]
        rgba = arr if arr.shape[2] == 4 else np.concatenate([arr, np.full((h, w, 1), 255, np.uint8)], axis=2)
        img = bpy.data.images.new(f"{name}_{kind}", w, h, alpha=True)
        img.pixels.foreach_set((rgba[::-1].astype(np.float32) / 255.0).ravel())  # Blender rows run up
        path = os.path.join(out_dir, f"{name}_{kind}.png")
        img.filepath_raw, img.file_format = path, "PNG"
        img.save()
        paths[kind] = path
    counts = {int(r): int((region == r).sum()) for r in np.unique(region)}
    return {"paths": paths, "coverage": coverage, "faces_by_region": counts}


def smart_uv(obj):
    activate(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.01)
    bpy.ops.object.mode_set(mode="OBJECT")


def main():
    a = _pdx.args()
    _pdx.clear_scene()
    van = a["vanilla"]
    vp = van["p"]
    vanilla = mesh_object("vanilla", [vp[i:i + 3] for i in range(0, len(vp), 3)],
                          [van["tri"][i:i + 3] for i in range(0, len(van["tri"]), 3)])
    for vi, row in enumerate(van["weights"]):
        for bone, w in row.items():
            g = vanilla.vertex_groups.get(bone) or vanilla.vertex_groups.new(name=bone)
            g.add([vi], w, "REPLACE")
    if a["source"]["kind"] == "direct":
        return direct(a, van, vanilla)
    res = {"stages": {}}
    src = a["source"]
    read = Readability(a.get("readability"), van, a["barding"]) if a.get("barding") else None
    if src["kind"] == "model":
        obj = import_model(src["path"])
        if src.get("merge"):
            merge_seams(obj, float(src["merge"]))
        res["stages"]["imported"] = {"tris": tri_count(obj), "pieces": count_pieces(obj)}
        keep_largest(obj)
        res["stages"]["largest_piece"] = {"tris": tri_count(obj)}
        if a.get("fit", True):
            s, off = fit(obj, vanilla)
        else:  # already on the vanilla skeleton (Track M's game version): a box fit would move it off the bones
            s, off = 1.0, [0.0, 0.0, 0.0]
        res["fit"] = {"scale": s, "offset": off}
        if a.get("reshape"):
            res["fit"]["height_scale"] = scale_height(obj, vanilla, van["bones"])
            res["fit"]["reshape"] = hs.reshape(obj, vanilla, van["bones"], leg_top=float(a.get("leg_top", 6.0)))
        debug_dump(a, obj, "1_reshaped")
        if a.get("textured_out"):  # M21: the model's own UVs after the fit and reshape, for its texture's bake
            write_obj(obj, a["textured_out"], uv=True)
        if a.get("voxel_fine"):  # closed before any edit (TripoSG's pieces and open sheets)
            res["stages"]["solid_fine"] = solidify(obj, a["voxel_fine"])
        debug_dump(a, obj, "2_solid_fine")
        cut = a.get("cut")
        if cut:
            if cut.get("y") == "saddle":  # the cap at vanilla's back height at the saddle
                z = van["bones"]["Saddle_Node"][2]
                cut = {**cut, "y": top_at(vanilla, z - 0.5, z + 0.5, 1.0)}
            res["stages"]["rider_cut"] = {"y": cut["y"], "tris_removed": cut_rider_box(obj, cut)}
        debug_dump(a, obj, "3_rider_cut")
        lump = a.get("lump")
        if lump:
            res["stages"]["lump_flattened"] = hs.flatten_lump(
                obj, lump["z0"], lump["z1"], lump["y0"], lump["y1"], lump.get("margin", 0.3))
        if a.get("hem_y"):
            res["stages"]["skirts_trimmed"] = {"hem_y": a["hem_y"], "tris_removed": trim_skirts(
                obj, van["bones"], float(a["hem_y"]), float(a.get("leg_radius", 0.9)))}
            keep_largest(obj)
        if a.get("neck_allow"):  # round 3: the neck no thicker than vanilla's x neck_allow
            res["stages"]["neck_warp"] = hs.neck_warp(obj, vanilla, van["bones"], float(a["neck_allow"]))
        if a.get("flank_allow"):  # round 3: each flank no wider than vanilla's x flank_allow
            res["stages"]["flank_warp"] = hs.flank_warp(obj, vanilla, van["bones"], float(a["flank_allow"]))
        if a.get("leg_allow"):  # round 3: legs no thinner than vanilla's x leg_allow, before the coarse voxels
            res["stages"]["leg_inflate"] = hs.leg_inflate(obj, vanilla, van["bones"], float(a.get("hem_y") or 6.0),
                                                          float(a["leg_allow"]))
        tail = a.get("tail")
        build_tail = False
        if tail:  # round 3: our own tail when the generated one is missing, fused or a stub
            import _horse_measure as hm
            state = hm.tail_state(hs.verts(obj), hs.faces(obj), van["bones"])
            build_tail = tail.get("build", "auto") is True or (tail.get("build", "auto") == "auto" and state["state"] != "free")
            res["tail"] = {"source": state, "build": build_tail}
            if build_tail and state["state"] != "missing":
                res["tail"]["stub_cut_tris"] = cut_tail_stub(obj, van["bones"], float(tail.get("cut", 0.8)))
        debug_dump(a, obj, "4_trimmed")
        # the bake source: the fine surface after every edit so far; each later reshape of the
        # body moves it too (the voxel solid, relax and reduction don't: those are what the
        # normal map makes up for)
        high = obj.copy()
        high.data = obj.data.copy()
        high.name = "bake_source"  # not linked to the scene: the joins and selections below never see it
        if a.get("tail_gap"):
            res["stages"]["tail_cleared"] = {"vertices": clear_tail(obj, van["bones"], float(a["tail_gap"])),
                                             "bake_source_vertices": clear_tail(high, van["bones"], float(a["tail_gap"]))}
        if a.get("voxel"):  # merge what the edits made overlap; closed in, closed out
            res["stages"]["solid"] = solidify(obj, a["voxel"])
        else:
            res["stages"]["holes_filled"] = {"boundary_edges": fill_holes(obj), "tris": tri_count(obj)}
        debug_dump(a, obj, "5_solid")
    elif src["kind"] in ("procedural", "anatomy"):
        high, build_tail = None, False
        obj, res["fit"] = (anatomy_horse(van, vanilla, src, read) if src["kind"] == "anatomy" else procedural_horse(van, src))
        res["stages"]["built"] = {"tris": tri_count(obj), "pieces": count_pieces(obj)}
        high = obj.copy()  # the bake source: the smooth surface before the reduction
        high.data = obj.data.copy()
        high.name = "bake_source"
        if a.get("tail"):
            import _horse_measure as hm
            state = hm.tail_state(hs.verts(obj), hs.faces(obj), van["bones"])
            build_tail = a["tail"].get("build", "auto") is True or (a["tail"].get("build", "auto") == "auto" and state["state"] != "free")
            res["tail"] = {"source": state, "build": build_tail}
        debug_dump(a, obj, "5_solid")
    else:
        high, build_tail = None, False
        obj = control_horse(van["bones"])
        res["stages"]["built"] = {"tris": tri_count(obj), "pieces": count_pieces(obj)}
        res["fit"] = {"scale": 1.0, "offset": [0, 0, 0], "note": "built on the skeleton: no fit"}
    lo, hi = bounds(obj)
    vlo, vhi = bounds(vanilla)
    res["bounds"] = {"ours": [list(lo), list(hi)], "vanilla": [list(vlo), list(vhi)]}
    saddle = van["bones"]["Saddle_Node"]
    res["back"] = {}
    for label, (z0, z1) in (("withers", (saddle[2] - 3.4, saddle[2] - 2.4)), ("croup", (saddle[2] + 3.0, saddle[2] + 4.0)),
                            ("saddle", (saddle[2] - 0.5, saddle[2] + 0.5))):
        res["back"][label] = {"ours": top_at(obj, z0, z1, 1.0), "vanilla": top_at(vanilla, z0, z1, 1.0)}
    res["saddle_node_y"] = saddle[1]
    if a.get("relax"):  # even out voxel stair-steps before the reduction
        activate(obj)
        mod = obj.modifiers.new("relax", "SMOOTH")
        mod.factor, mod.iterations = 0.5, int(a["relax"])
        bpy.ops.object.modifier_apply(modifier=mod.name)
    log = {}
    res["stages"]["decimated"] = {"tris": decimate(obj, int(a["target_tris"]), log, a.get("quads")), **log}
    if count_pieces(obj) > 1:  # remeshing can leave a stray shell
        keep_largest(obj)
        res["stages"]["decimated"]["tris_one_piece"] = tri_count(obj)
    if src["kind"] == "model" and a.get("cut"):
        before, after, want, note = match_saddle(obj, vanilla, van["bones"], also=[high])
        res["saddle"] = {"before": before, "after": after, "vanilla": want, "note": note,
                         "saddle_node_y": van["bones"]["Saddle_Node"][1]}
    if a.get("chanfron"):
        res["chanfron"] = {"tris": add_chanfron(obj, van["bones"], **a["chanfron"])}
    if a.get("pieces"):  # Track M: the master's cloth as closed low slabs, fitted like the source model
        s_, off_ = res["fit"]["scale"], np.array(res["fit"]["offset"])
        res["pieces"] = {}
        for q in json.loads(open(a["pieces"], encoding="utf-8").read()):
            v = np.array(q["v"]) * s_ + off_
            f = np.array(q["f"], dtype=np.int64)
            join_piece(obj, q["name"], {"v": v, "f": f, "part": [q.get("part", hb.CLOTH_PANEL)] * len(f),
                                        "hem": [q["hem"] * s_ + off_[1]] * len(f), "anchor": np.array(q["anchor"]) * s_ + off_})
            res["pieces"][q["name"]] = {"tris": int(len(f))}
    if build_tail:  # found on the body alone, before the panels join it
        root, offset = tail_anchor(obj, van["bones"])
        b = hb.tail(van["bones"], root, offset=offset,
                    **{k: v for k, v in a["tail"].items() if k in ("sides", "rings", "radii", "flat")})
    if a.get("barding"):
        barding = dict(a["barding"])
        if read is not None and read.barding is not None:  # with what else hangs off the body
            extra = ([hb.stirrups(**a["stirrups"])] if a.get("stirrups") else []) + ([b] if build_tail else [])
            barding["hem_length"] = shorter_hem(obj, barding, read, extra)
        res["barding"] = add_barding(obj, van["bones"], barding, a.get("stirrups"))
    if read is not None:
        res["readability"] = read.log
    if build_tail:
        join_piece(obj, "tail", b)
        res["tail"]["built"] = {**b["log"], "offset": [round(x, 2) for x in offset]}
    smart_uv(obj)
    wstats = {}
    transfer_weights(obj, vanilla, a.get("smooth", 2), a.get("influences", 3),
                     van["bones"] if a.get("chain_weights") else None, wstats)
    res["weights"] = wstats
    if a.get("chanfron"):
        res["chanfron"]["pinned_vertices"] = pin_chanfron(obj)
    elif a.get("pieces") and any(q.get("part") == 1 for q in json.loads(open(a["pieces"], encoding="utf-8").read())):
        res["head_pieces_pinned"] = pin_chanfron(obj)  # the ears and plume: rigid with the head
    if a.get("barding") or a.get("pieces") or build_tail:  # panels, stirrups and the tail: their own weights
        res.setdefault("barding", {})["pinned_vertices"] = pin_barding(obj, van["bones"])
    if a.get("paint"):
        pt = a["paint"]
        res["textures"] = paint_textures(obj, van["bones"], pt["palette"], float(pt.get("hem_y", a.get("hem_y") or 0.0)),
                                         pt["out_dir"], pt["name"], int(pt.get("size", 512)))
    verts, tris, unweighted = export(obj, a["geometry_out"], a.get("preview_out"))
    if high is not None and a.get("bake_source_out"):
        write_obj(high, a["bake_source_out"])
        res["bake_source"] = {"path": a["bake_source_out"], "tris": tri_count(high)}
    res["final"] = {"vertices": verts, "tris": tris, "unweighted_vertices": unweighted, "pieces": count_pieces(obj)}
    # a reshape that moves neighbours apart steeply leaves steps on the fine surface: invisible
    # after the reduction, but the normal bake picks them up
    res["steep_reshapes"] = {k: v["steepness"] for k, v in res["stages"].items()
                             if isinstance(v, dict) and v.get("steepness", {}).get("max", 0) > STEEP}
    _pdx.write_result(a, res)
