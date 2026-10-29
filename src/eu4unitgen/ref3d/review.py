"""The animation review (the user, after T38): a soldier on foot through every animation it plays, next to a base-game
unit playing the same set (``core.stances.example_unit``: western_halbard's westerngfx_INFANTRY_1 for
polearm_twohanded, DLH_INFANTRY_1's spear and shield for spear_shield), with automatic flags.

One background Blender run per animation state (``Session.gallery`` shots): the rest pose and ``SAMPLES`` points
through the state, each from the front and the side, rendered and dumped (the posed meshes and bone heads). The
flags come from the dumps (pure Python), each against the base-game unit at the same point of the same animation:

  bends     the share of the body's surface squashed below ``SQUASH`` of its rest area
  clipping  the share of each hand part's points inside the body
  floating  how far each hand part is from the nearest hand node (a weapon runs through the hand; a shield is
            strapped to the forearm)

A flag is raised when ours is worse than the base game's by more than ``MARGIN``. -> ``review.json`` and
``review_sheet.png`` under ``<preview>/review``; the regression set reads the numbers.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

STATES = ("idle", "walk", "attack")
SAMPLES = 5
SQUASH = 0.8
MARGIN = {"bends": 0.08, "clipping": 0.05, "floating": 0.6}
VIEW_ELEVATION = 5.0  # degrees: nearly level, so arms and legs read clearly
CAMERA = {"target": [0, 0, 4.2], "ortho_scale": 12.0, "width": 360, "height": 400, "shadow": False}
HANDS = ("Right_hand_node", "Left_hand_node")


def shots(samples: int = SAMPLES) -> list[dict]:
    """The rest pose then ``samples`` points through the state, each front then side."""
    el = math.radians(VIEW_ELEVATION)
    cams = [dict(CAMERA, view=[0.0, math.cos(el), -math.sin(el)]), dict(CAMERA, view=[-math.cos(el), 0.0, -math.sin(el)])]
    out = [{"rest": True, "camera": c} for c in cams]
    for i in range(samples):
        t = i / max(samples - 1, 1)
        out += [{"t": t, "camera": c} for c in cams]
    return out


def _tri_areas(p: list, tri: list) -> list[float]:
    out = []
    for t in range(0, len(tri), 3):
        a, b, c = (p[3 * tri[t + k]:3 * tri[t + k] + 3] for k in range(3))
        u = [b[i] - a[i] for i in range(3)]
        v = [c[i] - a[i] for i in range(3)]
        x = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        out.append(0.5 * math.sqrt(sum(k * k for k in x)))
    return out


def squashed(rest: dict, posed: dict, below: float = SQUASH) -> float:
    """The share of the surface (by rest area) whose triangles shrank below ``below`` of their rest area."""
    a0, a1 = _tri_areas(rest["p"], rest["tri"]), _tri_areas(posed["p"], posed["tri"])
    tot = sum(a0) or 1.0
    return sum(x for x, y in zip(a0, a1) if x > 1e-9 and y / x < below) / tot


def split_dump(d: dict, hand_parts: set | None = None) -> tuple[dict | None, dict, dict]:
    """(the body: the first mesh, the parts on the hand nodes (``hand_parts``' entities; all but the body when None),
    the body's own bone heads: a weapon's skeleton has bones of its own, and merged they moved the hand nodes)."""
    meshes = [(k, v) for k, v in d.items() if k != "bones"]
    if not meshes:
        return None, {}, {}
    body_name = meshes[0][0]
    rigs = d.get("bones") or {}
    bones = rigs.get(body_name) or next(iter(rigs.values()), {})
    parts = {k: v for k, v in meshes[1:] if hand_parts is None or k in hand_parts}
    return meshes[0][1], parts, bones


def point_triangle(p, tri) -> float:
    """The distance from ``p`` to the triangle ``tri`` (three points): the closest-point regions (Ericson 5.1.5)."""
    a, b, c = tri
    sub = lambda u, v: [u[0] - v[0], u[1] - v[1], u[2] - v[2]]  # noqa: E731
    dot = lambda u, v: u[0] * v[0] + u[1] * v[1] + u[2] * v[2]  # noqa: E731
    ab, ac, ap = sub(b, a), sub(c, a), sub(p, a)
    d1, d2 = dot(ab, ap), dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return math.dist(p, a)
    bp = sub(p, b)
    d3, d4 = dot(ab, bp), dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return math.dist(p, b)
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        return math.dist(p, [a[k] + v * ab[k] for k in range(3)])
    cp = sub(p, c)
    d5, d6 = dot(ab, cp), dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return math.dist(p, c)
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        return math.dist(p, [a[k] + w * ac[k] for k in range(3)])
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return math.dist(p, [b[k] + w * (c[k] - b[k]) for k in range(3)])
    den = 1.0 / ((va + vb + vc) or 1e-12)
    v, w = vb * den, vc * den
    return math.dist(p, [a[k] + ab[k] * v + ac[k] * w for k in range(3)])


def shot_measures(rest: dict, posed: dict, hand_parts: set | None = None) -> dict:
    """One shot's numbers (module doc) from its dump and the rest pose's; ``hand_parts``: the entities on the hand
    nodes (a sword on the belt isn't held)."""
    from eu4unitgen.ref3d.quality import inside_count
    body0, _, _ = split_dump(rest)
    body, parts, bones = split_dump(posed, hand_parts)
    if body is None or body0 is None or len(body0["p"]) != len(body["p"]):
        return {}
    m = {"bends": round(squashed(body0, body), 4), "clipping": {}, "floating": {}}
    hands = [bones[h] for h in HANDS if h in bones]
    for name, part in parts.items():
        pts = [part["p"][i:i + 3] for i in range(0, len(part["p"]), 3)]
        if not pts:
            continue
        step = max(1, len(pts) // 300)
        sample = pts[::step]
        m["clipping"][name] = round(inside_count(sample, body["p"], body["tri"]) / len(sample), 4)
        if hands:  # to its surface: the base game's halberd shaft has vertices only at its ends (6.6 from the hand)
            tris = [[pts[part["tri"][t + k]] for k in range(3)] for t in range(0, len(part["tri"]), 3)]
            m["floating"][name] = round(min(point_triangle(h, t) for h in hands for t in tris), 3)
    return m


def flags(ours: dict, game: dict) -> list[str]:
    """A shot's flags: ours against the base game's at the same shot."""
    out = []
    if ours.get("bends", 0) > game.get("bends", 0) + MARGIN["bends"]:
        out.append(f"bad bends: {ours['bends']:.0%} squashed (the game's {game.get('bends', 0):.0%})")
    gc = max((game.get("clipping") or {}).values(), default=0.0)
    for k, v in (ours.get("clipping") or {}).items():
        if v > gc + MARGIN["clipping"]:
            out.append(f"clipping: {v:.0%} of {_short(k)} inside the body (the game's {gc:.0%})")
    gf = max((game.get("floating") or {}).values(), default=0.0)
    for k, v in (ours.get("floating") or {}).items():
        if v > gf + MARGIN["floating"]:
            out.append(f"floating: {_short(k)} {v:.1f} from the hand (the game's {gf:.1f})")
    return out


def _short(node: str) -> str:
    n = node.lower()
    return "the shield" if "shield" in n else "the weapon" if any(k in n for k in ("lance", "weapon", "spear", "halbard")) else node


def review(preview: Path, tier: int, stance: str, eu4: Path, name: str = "preview", utype: str = "infantry",
           samples: int = SAMPLES) -> dict:
    """The review of the staged soldier (``preview``'s ``<name>.json`` at ``tier``) against the base game's unit for
    ``stance`` (module doc). -> {"stance", "against", "states": {state: {"ours": [..], "game": [..], "flags": [..]}},
    "summary": {...}, "sheet"}."""
    from eu4unitgen.core import stances
    from eu4unitgen.core.character import CharacterSpec
    from eu4unitgen.export.pipeline import Session, Settings
    preview = Path(preview)
    out_dir = preview / "review"
    out_dir.mkdir(exist_ok=True)
    against = stances.example_unit(str(eu4), stance)
    ch = CharacterSpec.load(preview / f"{name}.json")
    s = Session(Settings(eu4_path=Path(eu4), owned_dlc=[], prefix="hr_", stage_dir=preview / "review_stage"))
    sh = shots(samples)
    res: dict = {"stance": stance, "against": against, "states": {}}
    images: dict = {}
    for state in STATES:
        dd = out_dir / state
        g = s.gallery(ch.set_spec(), units=[(utype, tier)], camera=None, lit={"normal_maps": True}, state=state,
                      shots=sh, entities=[against] if against else None, dump_dir=dd)
        for key, im in g.images.items():
            if len(key) == 3:
                p = dd / f"{key[0]}_{key[1]}_{key[2]}.png"
                im.save(p)
                images[(state, "ours" if key[0] != "entity" else "game", key[2])] = p
        st = {"ours": [], "game": [], "flags": []}
        held = {who: {e for b, e in (g.parts.get(k) or {}).items() if b in HANDS}
                for who, k in (("ours", (utype, tier)), ("game", ("entity", against)))}
        for who, prefix in (("ours", f"{utype}_{tier}"), ("game", against)):
            if not prefix:
                continue
            dumps = [dd / f"{prefix}_{i}_posed.json" for i in range(len(sh))]
            if not all(p.is_file() for p in dumps):
                continue
            rest = json.loads(dumps[0].read_text(encoding="utf-8"))
            for i in range(2, len(sh), 2):  # one measure per pose (the front shot's dump; the side's is the same pose)
                st[who].append(shot_measures(rest, json.loads(dumps[i].read_text(encoding="utf-8")), held[who]))
        for k, (o, gm) in enumerate(zip(st["ours"], st["game"])):
            for f in flags(o, gm):
                st["flags"].append({"pose": k, "t": round(k / max(samples - 1, 1), 2), "flag": f})
        res["states"][state] = st
    res["summary"] = summary(res)
    res["sheet"] = str(sheet(res, images, out_dir / "review_sheet.png", samples))
    (out_dir / "review.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def summary(res: dict) -> dict:
    """The worst of each measure over every state, ours and the game's, and the flag count (the regression's)."""
    def worst(who, key):
        vals = []
        for st in res["states"].values():
            for m in st[who]:
                v = m.get(key)
                vals.append(max(v.values(), default=0.0) if isinstance(v, dict) else (v or 0.0))
        return round(max(vals, default=0.0), 4)
    return {"flags": sum(len(st["flags"]) for st in res["states"].values()),
            **{f"{k}_{who}": worst(who, k) for k in ("bends", "clipping", "floating") for who in ("ours", "game")}}


def sheet(res: dict, images: dict, path: Path, samples: int) -> Path:
    """Rows: each state, ours and the game's, front and side; columns: the rest pose then the samples. A flagged
    pose is outlined red on our row with its flags under the sheet's state title."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font, bold = ImageFont.truetype("arial.ttf", 15), ImageFont.truetype("arialbd.ttf", 20)
    except OSError:
        font = bold = ImageFont.load_default()
    cw, chh, lw, title_h = 180, 200, 150, 34
    n_cols = samples + 1
    rows = []
    for state in STATES:
        st = res["states"].get(state) or {}
        rows.append(("title", state, st))
        for view in (0, 1):
            for who in ("ours", "game"):
                rows.append(("img", state, (who, view)))
    h = sum(title_h + 22 * max(1, min(4, len((r[2] or {}).get("flags") or []))) if r[0] == "title" else chh for r in rows)
    img = Image.new("RGB", (lw + n_cols * cw, h + 40), (238, 238, 242))
    d = ImageDraw.Draw(img)
    d.text((10, 8), f"Animation review: {res['stance']} (ours against the game's {res['against']})", fill=(10, 10, 14), font=bold)
    y = 40
    for kind, state, extra in rows:
        if kind == "title":
            fl = extra.get("flags") or []
            d.text((10, y + 6), f"{state}: {len(fl)} flag{'s' if len(fl) != 1 else ''}", fill=(10, 10, 14), font=bold)
            yy = y + title_h
            for f in fl[:4]:
                d.text((24, yy - 4), f"pose {f['pose'] + 1}: {f['flag']}", fill=(170, 20, 20), font=font)
                yy += 22
            y = yy if fl else y + title_h + 22
            continue
        who, view = extra
        d.text((10, y + chh // 2 - 20), f"{'ours' if who == 'ours' else 'the game'}\n{'front' if view == 0 else 'side'}",
               fill=(10, 10, 14), font=font)
        flagged = {f["pose"] for f in (res["states"].get(state) or {}).get("flags") or []}
        for c in range(n_cols):
            p = images.get((state, who, 2 * c + view))
            x = lw + c * cw
            if p and Path(p).is_file():
                im = Image.open(p).convert("RGB")
                im.thumbnail((cw - 4, chh - 4))
                img.paste(im, (x + (cw - im.width) // 2, y + (chh - im.height) // 2))
            if who == "ours" and c > 0 and (c - 1) in flagged:
                d.rectangle([x + 1, y + 1, x + cw - 2, y + chh - 2], outline=(200, 20, 20), width=3)
            if y == 40 + title_h + 22 or (c == 0 and view == 0 and who == "ours"):
                d.text((x + 4, y + 2), "rest" if c == 0 else f"pose {c}", fill=(90, 90, 96), font=font)
        y += chh
    img.save(path)
    return path


def review_version(vdir: Path, subject: dict, out: Path, eu4: Path, tier: int = 1) -> dict:
    """A built soldier version (``vdir``, on foot: ``trellis.to_foot`` done) staged on its own as the tier's infantry
    under ``out`` and reviewed (the regression set's form; the app's preview stage reviews its own staging)."""
    from eu4unitgen.ref3d import trellis, unitstage
    m = json.loads((Path(vdir) / "manifest.json").read_text(encoding="utf-8"))
    soldier = unitstage.rider_entry(Path(vdir), subject)
    sp = unitstage.spec(f"review of {Path(vdir).name}", tier, None, None, unitstage.unit_names("review"),
                        unitstage.TEST, soldier=soldier)
    st = unitstage.stage(Path(out), sp, "preview")
    if st["exit"] != 0:
        raise RuntimeError(f"staging the review failed; see {Path(out) / 'log.txt'}")
    return review(Path(out), tier, trellis.foot_stance(m), eu4)
