"""Texture recipes: what to do to each retexturable slot of a unit set (SPEC §10 Phase 2).

A recipe is a JSON file in the user's project (never in this repo):

    {"schema": 1,
     "slots": {
       "1/infantry": {
         "base": null,                      # default: the base culture's pdxmesh for that tier/role
         "protect": {"skin": true, "metal": false, "mask": null},
         "ops": [
           {"op": "palette", "image": "ref.jpg", "targets": ["country1", "country2"], "fixed": true},
           {"op": "recolour", "region": "cloth", "colour": [120, 30, 30], "strength": 1.0},
           {"op": "hue_shift", "region": "cloth", "degrees": 30},
           {"op": "pattern", "region": "mask:stripes_mask.png", "image": "stripes.png",
            "blend": "multiply", "opacity": 0.6, "scale": 1.0},
           {"op": "spec_paint", "channel": "country1", "mask": "primary.png", "mode": "set"},
           {"op": "gloss", "region": "metal", "preset": "metal"},      # or "mask": ..., "percent": 10
           {"op": "flatten", "region": "area:arms", "strength": 1.0}   # normal map toward flat (skin over mail)
         ]}}}

Slots are ``<tier>/<role>`` with role in MESH_ROLES (``infantry`` also dresses the cavalry
rider and the artillery crew). Regions are those of ``textures.regions``, ``mask:<png>`` or
``area:<name>`` (the body's named texture areas and bands, textures.areas). Any op with a
``region`` may also carry ``"grow": px`` (negative shrinks) and ``"feather": px`` (soft edge),
applied to that region's mask at its own size.
Paths are relative to the recipe file. Diffuse operations skip the protect mask; spec
operations (``spec_paint``, ``gloss``) don't.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from eu4unitgen.core.naming import MESH_ROLES, TIERS
from eu4unitgen.textures import recolour as rc
from eu4unitgen.textures.areas import adjust
from eu4unitgen.textures.regions import COUNTRY_REGIONS, REGIONS, area, auto_regions, protect_mask

SCHEMA = 1
OPS = ("palette", "recolour", "hue_shift", "pattern", "spec_paint", "gloss", "flatten")
SPEC_OPS = ("spec_paint", "gloss")


class RecipeError(ValueError):
    pass


@dataclass
class SlotRecipe:
    base: str | None = None
    protect_skin: bool = True
    protect_metal: bool = False
    protect_mask: str | None = None
    ops: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"base": self.base, "protect": {"skin": self.protect_skin, "metal": self.protect_metal,
                                               "mask": self.protect_mask}, "ops": self.ops}


@dataclass
class TextureRecipe:
    slots: dict[tuple[int, str], SlotRecipe] = field(default_factory=dict)
    root: Path = Path(".")  # relative paths in ops resolve here (the recipe file's folder)

    # ------------------------------------------------------------------ files
    @classmethod
    def load(cls, path: Path) -> "TextureRecipe":
        path = Path(path)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RecipeError(f"can't read texture recipe {path}: {exc}") from exc
        return cls.from_json(doc, path.parent)

    @classmethod
    def from_json(cls, doc: dict, root: Path = Path(".")) -> "TextureRecipe":
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
            raise RecipeError(f"texture recipe must be an object with \"schema\": {SCHEMA}")
        out = cls(root=Path(root))
        for key, raw in (doc.get("slots") or {}).items():
            slot_key = parse_slot(key)
            protect = raw.get("protect") or {}
            slot = SlotRecipe(base=raw.get("base"), protect_skin=bool(protect.get("skin", True)),
                              protect_metal=bool(protect.get("metal", False)), protect_mask=protect.get("mask"),
                              ops=list(raw.get("ops") or []))
            for op in slot.ops:
                check_op(op)
            out.slots[slot_key] = slot
        return out

    def to_json(self) -> dict:
        return {"schema": SCHEMA, "slots": {f"{t}/{r}": s.to_json() for (t, r), s in sorted(self.slots.items())}}

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ use
    def active(self) -> dict[tuple[int, str], SlotRecipe]:
        """Slots that change something."""
        return {k: s for k, s in sorted(self.slots.items()) if s.ops}

    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    def image(self, rel: str) -> Image.Image:
        try:
            with Image.open(self.path(rel)) as im:
                im.load()
                return im.copy()
        except OSError as exc:
            raise RecipeError(f"can't open image {self.path(rel)}: {exc}") from exc


def parse_slot(key: str) -> tuple[int, str]:
    tier, _, role = str(key).partition("/")
    if not tier.isdigit() or int(tier) not in TIERS or role not in MESH_ROLES:
        raise RecipeError(f"slot {key!r} must be <tier>/<role>, tier {TIERS}, role {MESH_ROLES}")
    return int(tier), role


def _check_region(op: dict, key: str = "region") -> None:
    from eu4unitgen.textures.areas import area_names

    region = op.get(key)
    ok = isinstance(region, str) and (region in REGIONS or (region.startswith("mask:") and len(region) > 5)
                                      or (region.startswith("area:") and region[5:] in area_names()))
    if not ok:
        raise RecipeError(f"{op.get('op')}: {key} {region!r} must be one of {REGIONS}, mask:<png> or "
                          f"area:<name> ({', '.join(area_names())})")
    from eu4unitgen.textures.areas import FEATHER_MAX, GROW_MAX

    for k, lo, hi in (("grow", -GROW_MAX, GROW_MAX), ("feather", 0, FEATHER_MAX)):
        if k in op and not (isinstance(op[k], int) and not isinstance(op[k], bool) and lo <= op[k] <= hi):
            raise RecipeError(f"{op.get('op')}: {k} must be a whole number of pixels, {lo} to {hi}")
    if "protect" in op and not isinstance(op["protect"], bool):
        raise RecipeError(f"{op.get('op')}: protect must be true or false (false: the slot's protect mask doesn't apply)")


def _check_colour(c: Any, what: str) -> None:
    if not (isinstance(c, (list, tuple)) and len(c) == 3 and all(isinstance(v, int) and 0 <= v <= 255 for v in c)):
        raise RecipeError(f"{what} must be [r, g, b] with 0-255 integers")


def check_op(op: dict) -> None:
    kind = op.get("op") if isinstance(op, dict) else None
    if kind not in OPS:
        raise RecipeError(f"unknown texture operation {op!r}; known: {OPS}")
    if kind == "palette":
        if not op.get("image"):
            raise RecipeError("palette: needs a reference image")
        targets = op.get("targets", list(COUNTRY_REGIONS))
        if not targets:
            raise RecipeError("palette: needs at least one target region")
        for t in targets:
            _check_region({"op": "palette", "region": t})
    elif kind == "recolour":
        _check_region(op)
        _check_colour(op.get("colour"), "recolour: colour")
        if not isinstance(op.get("detail", 1.0), (int, float)) or not 0 <= op.get("detail", 1.0) <= 1:
            raise RecipeError("recolour: detail must be 0-1")
    elif kind == "hue_shift":
        _check_region(op)
        if not isinstance(op.get("degrees"), (int, float)):
            raise RecipeError("hue_shift: degrees must be a number")
    elif kind == "pattern":
        _check_region(op)
        if not op.get("image"):
            raise RecipeError("pattern: needs an image")
        if op.get("blend", "multiply") not in rc.BLENDS:
            raise RecipeError(f"pattern: blend must be one of {rc.BLENDS}")
    elif kind == "spec_paint":
        if op.get("channel") not in (*COUNTRY_REGIONS, "all"):
            raise RecipeError(f"spec_paint: channel must be one of {COUNTRY_REGIONS} or all")
        if op.get("channel") == "all" and op.get("mode", "set") != "clear":
            raise RecipeError("spec_paint: channel all only clears (mode clear)")
        if "mask" not in op and "region" not in op:
            raise RecipeError("spec_paint: needs a mask (PNG) or a region")
        if "region" in op:
            _check_region(op)
        if op.get("mode", "set") not in ("set", "clear"):
            raise RecipeError("spec_paint: mode must be set or clear")
    elif kind == "gloss":
        if "mask" not in op:
            _check_region(op)
        if "preset" in op and op["preset"] not in rc.GLOSS_PRESETS:
            raise RecipeError(f"gloss: preset must be one of {sorted(rc.GLOSS_PRESETS)}")
        if "preset" not in op and not isinstance(op.get("percent"), (int, float)):
            raise RecipeError("gloss: needs a preset or a percent")
    elif kind == "flatten":
        _check_region(op)
        if not isinstance(op.get("strength", 1.0), (int, float)) or not 0 <= op.get("strength", 1.0) <= 1:
            raise RecipeError("flatten: strength must be 0-1")


# --------------------------------------------------------------------------- applying a slot

@dataclass
class SlotResult:
    diffuse: Image.Image | None
    specular: Image.Image | None
    regions: dict[str, Image.Image]
    protect: Image.Image
    notes: list[str] = field(default_factory=list)
    normal: Image.Image | None = None


def apply_slot(recipe: TextureRecipe, slot: SlotRecipe, diffuse: Image.Image, specular: Image.Image | None,
               shader: str, role: str = "infantry", areas=None, normal=None) -> SlotResult:
    """Run a slot's operations on decoded base textures. Returns new images (None = unchanged).
    On a horse, the ``skin`` region is the whole hide (protected by default). ``areas``: the base
    body's named texture areas (textures.areas), a dict or a callable returning one. ``normal``:
    the base normal map (or a callable returning it), read only by ``flatten``."""
    colour_shader = shader == "PdxMeshColor"
    regions = auto_regions(diffuse, specular, colour_shader, hide=role == "horse")
    override = recipe.image(slot.protect_mask) if slot.protect_mask else None
    protect = protect_mask(regions, slot.protect_skin, slot.protect_metal, override)
    size = diffuse.size
    d, s = diffuse.convert("RGBA"), (specular.convert("RGBA") if specular is not None else None)
    d_changed = s_changed = False
    n = None  # the edited normal map, once a flatten op asks for it
    notes: list[str] = []

    area_cache: dict = {}

    edge = (0, 0)  # the current op's grow and feather

    def region(name: str, spec_size: bool = False) -> Image.Image:
        target = s.size if (spec_size and s is not None) else size
        if name.startswith("mask:"):
            m, how = recipe.image(name[5:]).convert("L"), Image.Resampling.BILINEAR
        elif name.startswith("area:"):
            if not area_cache:
                area_cache.update((areas() if callable(areas) else areas) or {})
            if name[5:] not in area_cache:
                raise RecipeError(f"area {name[5:]!r} isn't known for this body")
            m, how = area_cache[name[5:]], Image.Resampling.NEAREST
        else:
            m, how = regions[name], Image.Resampling.BILINEAR
        if edge != (0, 0):
            m, how = adjust(m, *edge), Image.Resampling.BILINEAR
        return m if m.size == target else m.resize(target, how)

    def mask_or_region(op: dict) -> Image.Image:
        if "mask" in op:
            return recipe.image(op["mask"]).convert("L").resize(s.size, Image.Resampling.BILINEAR)
        return region(op["region"], spec_size=True)

    def guard(op: dict):  # "protect": false on an op: the slot's protect mask doesn't apply to it
        return protect if op.get("protect", True) else None

    for op in slot.ops:
        check_op(op)
        kind = op["op"]
        edge = (op.get("grow", 0), op.get("feather", 0))
        if kind in SPEC_OPS and s is None:
            raise RecipeError(f"{kind}: this base has no specular map")
        if kind == "spec_paint" and not colour_shader:
            raise RecipeError(f"spec_paint: the base uses {shader}, which ignores spec R/G/B (no country colours)")
        if kind == "palette":
            targets = list(op.get("targets", list(COUNTRY_REGIONS)))
            if not colour_shader:
                targets = [t for t in targets if t not in COUNTRY_REGIONS] or ["cloth"]
            masks = {t: rc.effective(region(t), protect) for t in targets}
            order = sorted(targets, key=lambda t: -area(masks[t]))
            colours = rc.dominant_colours(recipe.image(op["image"]), len(order))
            for t, colour in zip(order, colours):
                if area(masks[t]) == 0:
                    notes.append(f"palette: region {t} is empty on this base")
                    continue
                d = rc.recolour(d, masks[t], colour)
                d_changed = True
                if t in COUNTRY_REGIONS and op.get("fixed", True) and s is not None:
                    s = rc.spec_paint(s, masks[t].resize(s.size), t, "clear")
                    s_changed = True
            notes.append("palette: " + ", ".join(f"{t}={c}" for t, c in zip(order, colours)))
        elif kind == "recolour":
            d = rc.recolour(d, rc.effective(region(op["region"]), guard(op)), tuple(op["colour"]),
                            float(op.get("strength", 1.0)), float(op.get("detail", 1.0)))
            d_changed = True
        elif kind == "hue_shift":
            d = rc.hue_shift(d, float(op["degrees"]), rc.effective(region(op["region"]), guard(op)))
            d_changed = True
        elif kind == "pattern":
            d = rc.pattern_overlay(d, rc.effective(region(op["region"]), guard(op)), recipe.image(op["image"]),
                                   op.get("blend", "multiply"), float(op.get("opacity", 1.0)),
                                   float(op.get("scale", 1.0)))
            d_changed = True
        elif kind == "spec_paint":
            channels = list(COUNTRY_REGIONS) if op["channel"] == "all" else [op["channel"]]
            for ch in channels:
                s = rc.spec_paint(s, mask_or_region(op), ch, op.get("mode", "set"))
            s_changed = True
        elif kind == "gloss":
            pct = rc.GLOSS_PRESETS[op["preset"]] if "preset" in op else float(op["percent"])
            s = rc.gloss(s, mask_or_region(op), pct)
            s_changed = True
        elif kind == "flatten":
            if n is None:
                n = normal() if callable(normal) else normal
                if n is None:
                    raise RecipeError("flatten: this base has no normal map")
                n = n.convert("RGBA")
            m = region(op["region"])
            n = rc.flatten(n, m if m.size == n.size else m.resize(n.size, Image.Resampling.NEAREST),
                           float(op.get("strength", 1.0)))
    return SlotResult(d if d_changed else None, s if s_changed else None, regions, protect, notes, n)


# --------------------------------------------------------------------------- the Textures screen's form

@dataclass
class SimpleSlot:
    """One slot as the GUI's fixed form shows it. ``to_slot`` writes the operations in a fixed
    order: palette, cloth recolour, pattern, spec masks, gloss, then any operations the form
    can't show (``extra``, kept as they were)."""
    base: str | None = None
    protect_skin: bool = True
    protect_metal: bool = False
    protect_mask: str | None = None
    palette_image: str | None = None
    palette_targets: list[str] = field(default_factory=lambda: list(COUNTRY_REGIONS))
    palette_fixed: bool = True
    cloth_colour: tuple[int, int, int] | None = None
    cloth_strength: float = 1.0
    pattern_image: str | None = None
    pattern_region: str = "cloth"
    pattern_blend: str = "multiply"
    pattern_opacity: float = 1.0
    pattern_scale: float = 1.0
    spec_masks: dict[str, str] = field(default_factory=dict)  # country1-3 -> PNG (set)
    gloss_preset: str | None = None
    gloss_region: str = "metal"
    gloss_mask: str | None = None
    extra: list[dict] = field(default_factory=list)

    def to_slot(self) -> SlotRecipe:
        ops: list[dict] = []
        if self.palette_image:
            ops.append({"op": "palette", "image": self.palette_image, "targets": list(self.palette_targets),
                        "fixed": self.palette_fixed})
        if self.cloth_colour is not None:
            ops.append({"op": "recolour", "region": "cloth", "colour": list(self.cloth_colour),
                        "strength": self.cloth_strength})
        if self.pattern_image:
            ops.append({"op": "pattern", "region": self.pattern_region, "image": self.pattern_image,
                        "blend": self.pattern_blend, "opacity": self.pattern_opacity, "scale": self.pattern_scale})
        for channel in COUNTRY_REGIONS:
            if self.spec_masks.get(channel):
                ops.append({"op": "spec_paint", "channel": channel, "mask": self.spec_masks[channel], "mode": "set"})
        if self.gloss_preset:
            g = {"op": "gloss", "preset": self.gloss_preset}
            g.update({"mask": self.gloss_mask} if self.gloss_mask else {"region": self.gloss_region})
            ops.append(g)
        ops += self.extra
        for op in ops:
            check_op(op)
        return SlotRecipe(self.base, self.protect_skin, self.protect_metal, self.protect_mask, ops)

    @classmethod
    def from_slot(cls, slot: SlotRecipe) -> "SimpleSlot":
        f = cls(base=slot.base, protect_skin=slot.protect_skin, protect_metal=slot.protect_metal,
                protect_mask=slot.protect_mask)
        seen: set[str] = set()
        for op in slot.ops:
            kind = op.get("op")
            if "finish" in op:  # the Character page's area finishes stay as they are (core.finishes)
                f.extra.append(op)
                continue
            if kind == "palette" and "palette" not in seen:
                f.palette_image, f.palette_fixed = op["image"], bool(op.get("fixed", True))
                f.palette_targets = list(op.get("targets", list(COUNTRY_REGIONS)))
            elif kind == "recolour" and op.get("region") == "cloth" and "recolour" not in seen:
                f.cloth_colour, f.cloth_strength = tuple(op["colour"]), float(op.get("strength", 1.0))
            elif kind == "pattern" and "pattern" not in seen:
                f.pattern_image, f.pattern_region = op["image"], op["region"]
                f.pattern_blend, f.pattern_opacity = op.get("blend", "multiply"), float(op.get("opacity", 1.0))
                f.pattern_scale = float(op.get("scale", 1.0))
            elif (kind == "spec_paint" and op.get("mode", "set") == "set" and op.get("mask")
                  and op["channel"] not in f.spec_masks):
                f.spec_masks[op["channel"]] = op["mask"]
                continue
            elif kind == "gloss" and "gloss" not in seen and "preset" in op:
                f.gloss_preset, f.gloss_mask = op["preset"], op.get("mask")
                f.gloss_region = op.get("region", "metal")
            else:
                f.extra.append(op)
                continue
            seen.add(kind)
        return f
