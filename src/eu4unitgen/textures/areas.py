"""Named texture areas of a body (SPEC Track R, the GUI step): head, torso, arms, legs, shins, ...

An area is a group of bones (``library/bodies.json`` "areas": defaults for the generic 28-bone
skeleton, overridable per body). Its mask is every triangle of the body whose vertices are mostly
skinned to the area's bones, drawn into texture space with a 2 px margin. Computed from the
body's own mesh (bone weights and UVs), so every body on the skeleton gets them without painting.
Texture recipes target them as ``area:<name>``.

Bands (``library/bodies.json`` "bands") are finer areas cut from a parent area along an axis:
``height`` (y) or ``reach`` (|x|, out from the body's middle; arms hang in an A-pose). A
triangle of the parent belongs to the band holding its centroid. ``adjust`` grows, shrinks and
feathers any mask (a recipe op's ``grow`` and ``feather``).
"""
from __future__ import annotations

from collections import Counter

from PIL import Image, ImageDraw

from eu4unitgen.library.data import library
from eu4unitgen.library.scanner import _walk, parse_pdx_mesh


def groups(base_file: str | None = None) -> dict[str, list[str]]:
    """Area name -> bones: the library's defaults, with the body's own overrides on top."""
    lib = library("bodies")
    out = {k: list(v) for k, v in lib.get("areas", {}).items()}
    if base_file:
        out.update({k: list(v) for k, v in lib["bodies"].get(base_file.lower(), {}).get("areas", {}).items()})
    return out


def bands(base_file: str | None = None) -> dict[str, dict]:
    """Band name -> {"of": parent area, "axis": "height" | "reach", "from", "to"}, the body's own on top."""
    lib = library("bodies")
    out = {k: dict(v) for k, v in lib.get("bands", {}).items()}
    if base_file:
        out.update({k: dict(v) for k, v in lib["bodies"].get(base_file.lower(), {}).get("bands", {}).items()})
    return out


def area_names() -> list[str]:
    names = set(library("bodies").get("areas", {})) | set(library("bodies").get("bands", {}))
    for body in library("bodies")["bodies"].values():
        names |= set(body.get("areas", {})) | set(body.get("bands", {}))
    return sorted(names)


def area_order(base_file: str | None = None) -> list[str]:
    """Every area and band, broad before narrow: the bone areas in library order, then the bands
    widest first. Finishes apply in this order, so a narrow area's finish wins over its parent's."""
    band_defs = bands(base_file)
    widest = sorted(band_defs, key=lambda b: -(band_defs[b]["to"] - band_defs[b]["from"]))
    return [a for a in groups(base_file) if a not in band_defs] + widest


GROW_MAX = 32  # px, either way
FEATHER_MAX = 32


def adjust(mask: Image.Image, grow: int = 0, feather: int = 0) -> Image.Image:
    """``mask`` grown (``grow`` > 0) or shrunk (< 0) by that many pixels, then its edge softened
    over ``feather`` pixels. Pixels are the mask's own."""
    from PIL import ImageFilter

    m = mask.convert("L")
    for _ in range(abs(int(grow))):  # 3x3 steps: a big rank filter is quadratic in its size
        m = m.filter(ImageFilter.MaxFilter(3) if grow > 0 else ImageFilter.MinFilter(3))
    if feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(feather / 2))
    return m


def skin_colour(diffuse: Image.Image, skin: Image.Image, where: list[Image.Image] | None = None) -> tuple[int, int, int]:
    """The body's own skin tone: the mean colour of its skin region, inside the ``where`` masks
    (the head and hands areas: the colour-based skin region also catches padding and hose)."""
    from PIL import ImageChops

    m = skin.resize(diffuse.size).point(lambda v: 255 if v >= 128 else 0)
    if where:
        inside = Image.new("L", diffuse.size, 0)
        for w in where:
            inside = ImageChops.lighter(inside, w.resize(diffuse.size))
        if ImageChops.multiply(m, inside).getbbox() is not None:
            m = ImageChops.multiply(m, inside)
    if m.getbbox() is None:
        return (196, 150, 120)  # no skin found: a neutral tan
    # the lit skin: 75th percentile by brightness (the mean sinks into beard and shadow; type1: 115 87 78
    # mean against 140 97 90 lit cheek)
    px = sorted((c for c, k in zip(diffuse.convert("RGB").get_flattened_data(), m.get_flattened_data()) if k),
                key=sum)
    return tuple(px[int(len(px) * 0.75)])


def overlay(diffuse: Image.Image, masks: dict[str, Image.Image], colours: dict[str, tuple],
            opacity: float = 0.5) -> Image.Image:
    """The body texture with the given areas tinted (later areas on top): the Character page's map."""
    out = diffuse.convert("RGB")
    for name, colour in colours.items():
        if name in masks:
            m = masks[name] if masks[name].size == out.size else masks[name].resize(out.size)
            out = Image.composite(Image.blend(out, Image.new("RGB", out.size, colour), opacity), out, m)
    return out


def area_masks(mesh: bytes, size: tuple[int, int], base_file: str | None = None,
               names: list[str] | None = None) -> dict[str, Image.Image]:
    """``{area: 'L' mask}`` at ``size`` for the body ``mesh`` (.mesh bytes): its bone areas, then its
    bands. A triangle belongs to the area(s) holding most of its three vertices' dominant bones;
    triangles in no area are left out."""
    tree = parse_pdx_mesh(mesh)
    # bands after the bone areas, widest first: finishes apply in row order, so narrow ones win
    all_bands = bands(base_file)
    band_defs = {b: all_bands[b] for b in area_order(base_file) if b in all_bands and (names is None or b in names)}
    masks = {a: Image.new("L", size, 0) for a in (names or groups(base_file)) if a not in band_defs}
    masks.update((b, Image.new("L", size, 0)) for b in band_defs)
    node = next((n for n in _walk(tree) if n["name"] == "mesh"), None)
    skeleton = next((n for n in _walk(tree) if n["name"] == "skeleton"), None)
    skin = next((c for c in node["children"] if c["name"] == "skin"), None) if node else None
    if skeleton is None or skin is None:  # an unskinned mesh has no bone areas
        return masks
    bone_of = {b["props"]["ix"][0]: b["name"] for b in skeleton["children"]}
    props = node["props"]
    skin = skin["props"]
    k = (skin.get("bones") or [4])[0]
    area_of_bone = {}
    wanted = set(masks) | {d["of"] for d in band_defs.values()}
    for area, bones in groups(base_file).items():
        if area in wanted:
            for b in bones:
                area_of_bone.setdefault(b, []).append(area)

    def dominant(v: int) -> str:
        ws = skin["w"][v * k:(v + 1) * k]
        return bone_of.get(skin["ix"][v * k + max(range(k), key=lambda j: ws[j])], "")

    pos = props["p"]

    def along(v: int, axis: str) -> float:
        return pos[v * 3 + 1] if axis == "height" else abs(pos[v * 3])

    # each band's parent extent on its axis, over the parent's own vertices
    nv = len(pos) // 3
    doms = [dominant(v) for v in range(nv)]
    extent = {}
    for b, d in band_defs.items():
        vals = [along(v, d["axis"]) for v in range(nv) if d["of"] in area_of_bone.get(doms[v], [])]
        extent[b] = (min(vals), max(vals)) if vals else None

    w, h = size
    draws = {a: ImageDraw.Draw(m) for a, m in masks.items()}
    u0, tri = props["u0"], props["tri"]
    for t in range(0, len(tri), 3):
        vs = tri[t:t + 3]
        votes = Counter(a for v in vs for a in area_of_bone.get(doms[v], []))
        if not votes:
            continue
        top = max(votes.values())
        won = {a for a, n in votes.items() if n == top}
        pts = [(u0[v * 2] * w, u0[v * 2 + 1] * h) for v in vs]
        for area in won:
            if area in draws and area not in band_defs:
                draws[area].polygon(pts, fill=255, outline=255, width=2)
        for b, d in band_defs.items():
            if d["of"] in won and extent[b] is not None:
                lo, hi = extent[b]
                f = (sum(along(v, d["axis"]) for v in vs) / 3 - lo) / ((hi - lo) or 1)
                if d["from"] <= f < d["to"] or (d["to"] >= 1 and f >= 1):
                    draws[b].polygon(pts, fill=255, outline=255, width=2)
    return masks
