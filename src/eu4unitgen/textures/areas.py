"""Named texture areas of a body (SPEC Track R, the GUI step): head, torso, arms, legs, shins, ...

An area is a group of bones (``library/bodies.json`` "areas": defaults for the generic 28-bone
skeleton, overridable per body). Its mask is every triangle of the body whose vertices are mostly
skinned to the area's bones, drawn into texture space with a 2 px margin. Computed from the
body's own mesh (bone weights and UVs), so every body on the skeleton gets them without painting.
Texture recipes target them as ``area:<name>``.
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


def area_names() -> list[str]:
    names = set(library("bodies").get("areas", {}))
    for body in library("bodies")["bodies"].values():
        names |= set(body.get("areas", {}))
    return sorted(names)


def skin_colour(diffuse: Image.Image, skin: Image.Image) -> tuple[int, int, int]:
    """The body's own skin tone: the mean colour of its skin region (face, hands)."""
    from PIL import ImageStat

    if skin.getbbox() is None:
        return (196, 150, 120)  # no skin found: a neutral tan
    mean = ImageStat.Stat(diffuse.convert("RGB"), skin.resize(diffuse.size).point(lambda v: 255 if v >= 128 else 0)).mean
    return tuple(round(c) for c in mean)


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
    """``{area: 'L' mask}`` at ``size`` for the body ``mesh`` (.mesh bytes). A triangle belongs to
    the area(s) holding most of its three vertices' dominant bones; triangles in no area are left out."""
    tree = parse_pdx_mesh(mesh)
    node = next(n for n in _walk(tree) if n["name"] == "mesh")
    skeleton = next(n for n in _walk(tree) if n["name"] == "skeleton")
    bone_of = {b["props"]["ix"][0]: b["name"] for b in skeleton["children"]}
    props = node["props"]
    skin = next(c for c in node["children"] if c["name"] == "skin")["props"]
    k = (skin.get("bones") or [4])[0]
    area_of_bone = {}
    for area, bones in groups(base_file).items():
        if names is None or area in names:
            for b in bones:
                area_of_bone.setdefault(b, []).append(area)

    def dominant(v: int) -> str:
        ws = skin["w"][v * k:(v + 1) * k]
        return bone_of.get(skin["ix"][v * k + max(range(k), key=lambda j: ws[j])], "")

    w, h = size
    masks = {a: Image.new("L", size, 0) for a in (names or groups(base_file))}
    draws = {a: ImageDraw.Draw(m) for a, m in masks.items()}
    u0, tri = props["u0"], props["tri"]
    for t in range(0, len(tri), 3):
        vs = tri[t:t + 3]
        votes = Counter(a for v in vs for a in area_of_bone.get(dominant(v), []))
        if not votes:
            continue
        top = max(votes.values())
        pts = [(u0[v * 2] * w, u0[v * 2 + 1] * h) for v in vs]
        for area, n in votes.items():
            if n == top and area in draws:
                draws[area].polygon(pts, fill=255, outline=255, width=2)
    return masks
