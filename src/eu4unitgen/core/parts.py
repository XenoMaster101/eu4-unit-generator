"""Procedural rigid parts: weapons, shields and helmet crests attached to a bone (SPEC §10 Phase 3).

A part is our own geometry (built by the Blender job ``build_parts``) and our own textures
(``textures.procedural``): nothing is derived from game files. Each part becomes a static
.mesh in the set's model folder, a pdxmesh and an entity that unit entities attach to a bone:

    attach = { Right_hand_node = "hr_mac_dory_1"  Left_hand_node = "hr_mac_hoplon_1"  head = "hr_mac_crest_1" }

Conventions (measured on 1.37.5 base-game meshes, SPEC §10 Phase 3; confirmed in-game by test 7
for the dory and the crest):
  * pdxmesh scale 0.5, no bones or locators, like vanilla weapons and shields.
  * Hand weapons in the ``Right_hand_node`` frame, grip at the origin. Infantry polearms point
    along +Z (western_halbard z -10.8..13.9, muslim_spear -10.8..12.1). The cavalry sabre's blade
    runs along -Z (cavalry_saber z -7.8..0.7); rider weapons default to that.
  * Shields in the ``Left_hand_node`` frame (the mesh axes): the face in the YZ plane, domed
    towards +X (indian_generic_01_type1_shield x -0.1..1.4, y -1.8..4.2, z -3.0..2.9). Vanilla's
    south_american_generic_01_type1_shield is a static PdxMeshColor shield, so a country-coloured
    face has precedent.
  * Head parts in the ``head`` bone frame (the mesh axes); the unit faces -Z (attach_back sits
    at +Z; spears level towards -Z in the mesh's frame).

Every default below was set on the western tier-1 body (core.fitting.REFERENCE_HEIGHT); fitting
scales the ``scaled`` lengths to other bodies and measures the head for crests.
"""
from __future__ import annotations

import base64
import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from eu4unitgen.core.generator import OWNED, GeneratedFile, GenerationError
from eu4unitgen.core.naming import LOADOUT_ROLES, PART_KINDS, Naming
from eu4unitgen.library.scanner import strip_mesh_props
from eu4unitgen.textures.paint import paint
from eu4unitgen.textures.procedural import (
    COUNTRY_CHANNELS, SIZE, TEXEL_DENSITY, Zone, atlas_images, encode_part_textures, part_images, shader_for,
    zone_map, zone_uvs,
)

PART_SCALE = 0.5  # vanilla weapons' and shields' pdxmesh scale


class PartError(ValueError):
    pass


@dataclass(frozen=True)
class Param:
    default: float
    lo: float
    hi: float
    label: str
    step: float = 0.05
    tip: str = ""


@dataclass(frozen=True)
class Choice:
    options: tuple[str, ...]
    default: str
    label: str
    tip: str = ""


@dataclass(frozen=True)
class ZoneDef:
    name: str
    label: str
    colour: tuple[int, int, int]
    gloss: float  # percent
    channel: str | None = None  # default country channel (None: fixed colour)
    material: str = "plain"  # textures.paint pattern class (Q2): wood, bronze, iron, hair, leather, paint, plain


@dataclass(frozen=True)
class PartKind:
    name: str
    label: str
    bone: str
    roles: tuple[str, ...]
    builder: str  # the Blender job's geometry builder
    layout: str  # textures.procedural layout
    params: dict[str, Param | Choice]
    zones: tuple[ZoneDef, ...]
    scaled: tuple[str, ...] = ()  # lengths fitting scales with the body height
    blazon: bool = False


WOOD, BRONZE, IRON, HAIR = (110, 75, 40), (180, 130, 60), (165, 165, 170), (170, 20, 25)
POINTS = ("up", "down")
DEFAULT_POINTS = {"infantry": "up", "crew": "up", "rider": "down"}  # what "auto" means unless the loadout says
POINT = Choice(("auto", *POINTS), "auto", "Point",
               "up: +Z, like the halberd and spears; down: -Z, like the cavalry sabre. auto: the loadout's direction "
               "for this body (default: up for infantry and crews, down for riders). In game (test 8) a rider's "
               "'down' is levelled forward, 'up' upright and back")


def _spear(tip, butt, head_length, head_width, shaft_radius, spike) -> dict:
    return {"tip": Param(tip, 4, 60, "Tip", 0.5, "z of the point; the grip is at 0"),
            "butt": Param(butt, -30, -1, "Butt", 0.5, "z of the shaft's end, behind the grip"),
            "head_length": Param(head_length, 0.5, 10, "Head length", 0.1),
            "head_width": Param(head_width, 0.1, 1.5, "Head width"),
            "shaft_radius": Param(shaft_radius, 0.05, 0.5, "Shaft radius", 0.01),
            "spike": Param(spike, 0, 4, "Butt spike", 0.1, "length of the sauroter"),
            "point": POINT}


SPEAR_ZONES = (ZoneDef("shaft", "Shaft", WOOD, 3, material="wood"),
               ZoneDef("metal", "Head and butt spike", BRONZE, 13, material="bronze"))

KINDS: dict[str, PartKind] = {k.name: k for k in (
    PartKind("dory", "Dory (spear)", "Right_hand_node", LOADOUT_ROLES, "spear", "atlas",
             _spear(15.0, -11.5, 4.0, 0.45, 0.17, 1.3), SPEAR_ZONES,
             scaled=("tip", "butt", "head_length", "spike")),
    PartKind("sarissa", "Sarissa (pike)", "Right_hand_node", ("infantry", "crew"), "spear", "atlas",
             _spear(30.0, -13.0, 2.5, 0.32, 0.15, 1.5), SPEAR_ZONES,
             scaled=("tip", "butt", "head_length", "spike")),
    PartKind("xyston", "Xyston (cavalry lance)", "Right_hand_node", LOADOUT_ROLES, "spear", "atlas",
             _spear(22.0, -16.0, 3.0, 0.4, 0.16, 2.0), SPEAR_ZONES,
             scaled=("tip", "butt", "head_length", "spike")),
    PartKind("xiphos", "Xiphos (sword)", "Right_hand_node", LOADOUT_ROLES, "sword", "atlas",
             {"blade_length": Param(5.0, 1.5, 10, "Blade length", 0.1),
              "blade_width": Param(0.55, 0.15, 1.5, "Blade width", tip="at its widest, two thirds along (leaf blade)"),
              "grip_length": Param(1.0, 0.4, 2.5, "Grip length", 0.1),
              "guard_width": Param(0.9, 0.2, 2.5, "Guard width"),
              "point": POINT},
             (ZoneDef("blade", "Blade and guard", IRON, 13, material="iron"),
              ZoneDef("hilt", "Grip and pommel", WOOD, 3, material="leather")),
             scaled=("blade_length", "grip_length")),
    PartKind("hoplon", "Hoplon (round shield)", "Left_hand_node", ("infantry", "crew"), "shield", "disc",
             {"radius": Param(3.8, 1.5, 6.0, "Radius", 0.1),
              "depth": Param(0.9, 0.0, 2.5, "Dome depth", tip="how far the face bulges outwards (+X)"),
              "rim": Param(0.35, 0.0, 1.0, "Rim width"),
              "thickness": Param(0.12, 0.04, 0.5, "Thickness", 0.01),
              "back_x": Param(-0.1, -2.0, 2.0, "Back plane", tip="x of the inside rim; the hand is at 0"),
              "centre_y": Param(1.2, -4.0, 6.0, "Centre height", 0.1, "above the hand"),
              "centre_z": Param(0.0, -4.0, 4.0, "Centre fore/aft", 0.1, "forward (-) or back (+) of the hand"),
              "blazon_rotate": Choice(("0", "90", "180", "270"), "0", "Blazon rotation", "degrees, counter-clockwise"),
              "blazon_mirror": Choice(("no", "yes"), "no", "Mirror the blazon")},
             (ZoneDef("face", "Face", (150, 110, 60), 10, "country1", material="paint"),
              ZoneDef("rim", "Rim", BRONZE, 13, material="bronze"),
              ZoneDef("back", "Inside", (120, 85, 50), 3, material="wood")),
             scaled=("radius", "centre_y"), blazon=True),
    PartKind("crest", "Crest (helmet)", "head", LOADOUT_ROLES, "crest", "atlas",
             {"base": Param(1.9, -1.0, 5.0, "Base height", tip="above the head joint; fitted: the helmet top less 0.25"),
              "height": Param(1.2, 0.2, 4.0, "Crest height"),
              "radius": Param(1.4, 0.5, 3.0, "Arc radius", tip="fitted: 1.1 x the head's front or back reach"),
              "span_deg": Param(70.0, 10.0, 90.0, "Arc half-span", 1.0, "degrees each side of the top"),
              "thickness": Param(0.28, 0.08, 1.2, "Thickness", 0.01),
              "centre_z": Param(0.0, -2.0, 2.0, "Centre fore/aft", tip="forward (-) or back (+) of the head joint"),
              "orientation": Choice(("fore_aft", "transverse"), "fore_aft", "Orientation")},
             (ZoneDef("hair", "Horsehair", HAIR, 3, material="hair"),)),
    PartKind("horse_ears", "Horse ears (pair)", "head", LOADOUT_ROLES, "ears", "atlas",
             {"base": Param(1.7, -1.0, 4.0, "Base height", tip="above the head joint; fitted: the helmet top less 0.45"),
              "spread": Param(0.62, 0.1, 2.0, "Spread", 0.02, "x of each ear's base; fitted: 0.63 x the head's half-width"),
              "length": Param(1.3, 0.3, 3.0, "Length", tip="base to tip"),
              "width": Param(0.5, 0.1, 1.5, "Width"),
              "depth": Param(0.2, 0.05, 0.8, "Depth", 0.01, "front-to-back thickness at the base"),
              "tilt_deg": Param(25.0, 0.0, 60.0, "Outward tilt", 1.0, "degrees from vertical, away from the head"),
              "lean_deg": Param(10.0, -30.0, 45.0, "Backward lean", 1.0, "degrees from vertical, towards the back (+Z)"),
              "centre_z": Param(0.1, -1.5, 1.5, "Fore/aft", tip="forward (-) or back (+) of the head joint")},
             (ZoneDef("outer", "Outer", (95, 70, 50), 3, material="hair"),
              ZoneDef("inner", "Inner (front)", (150, 105, 95), 3, material="leather")),
             scaled=("length", "width", "depth")),
    PartKind("greek_helmet", "Greek helmet", "head", LOADOUT_ROLES, "helmet", "atlas",
             {"style": Choice(("attic", "corinthian", "pilos"), "attic", "Style",
                              "attic: open face, cheek guards; corinthian: the face covered, an eye slot; pilos: a cone, "
                              "level rim"),
              "top": Param(1.92, 0.5, 4.0, "Top", tip="y of the crown above the head joint; fitted: the head top + 0.12"),
              "half_width": Param(0.98, 0.3, 2.0, "Half width", 0.02, "fitted: the head's half-width + 0.13"),
              "front": Param(-1.12, -2.5, 0.0, "Front", 0.02, "z of the brow (the unit faces -Z); fitted from the head"),
              "back": Param(1.05, 0.0, 2.5, "Back", 0.02, "z of the back; fitted: the head's back + 0.2 (type1's old "
                            "helmet back, where the tuft comes out)"),
              "thickness": Param(0.08, 0.02, 0.3, "Thickness", 0.01),
              "crest": Choice(("fore_aft", "transverse", "none"), "fore_aft", "Crest"),
              "crest_height": Param(0.9, 0.2, 3.0, "Crest height")},
             (ZoneDef("metal", "Bronze", BRONZE, 13, material="bronze"),
              ZoneDef("crest", "Crest", HAIR, 3, material="hair"),
              ZoneDef("slot", "Eye slot and lining", (25, 20, 15), 3, material="leather"))),
)}
assert set(KINDS) == set(PART_KINDS), "naming.PART_KINDS and parts.KINDS must list the same kinds"


# --------------------------------------------------------------------------- a part as the user specifies it

@dataclass
class PartSpec:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)  # overrides: numbers, or choice strings
    colours: dict[str, Any] = field(default_factory=dict)  # zone -> [r, g, b] or "country1".."country3"
    blazon: str | None = None  # PNG painted on the face (disc parts), relative to the loadout file

    def check(self) -> None:
        k = KINDS.get(self.kind)
        if k is None:
            raise PartError(f"unknown part kind {self.kind!r}; known: {', '.join(KINDS)}")
        for name, value in self.params.items():
            p = k.params.get(name)
            if p is None:
                raise PartError(f"{self.kind}: unknown param {name!r}; known: {', '.join(k.params)}")
            if isinstance(p, Choice):
                if str(value) not in p.options:
                    raise PartError(f"{self.kind}: {name} must be one of {p.options}, not {value!r}")
            elif not isinstance(value, (int, float)) or isinstance(value, bool) or not p.lo <= value <= p.hi:
                raise PartError(f"{self.kind}: {name} must be a number from {p.lo} to {p.hi}, not {value!r}")
        zones = {z.name for z in k.zones}
        for zone, value in self.colours.items():
            if zone not in zones:
                raise PartError(f"{self.kind}: unknown colour zone {zone!r}; known: {', '.join(sorted(zones))}")
            if isinstance(value, str):
                if value not in COUNTRY_CHANNELS:
                    raise PartError(f"{self.kind}: zone {zone} must be [r, g, b] or one of {COUNTRY_CHANNELS}")
            elif not (isinstance(value, (list, tuple)) and len(value) == 3
                      and all(isinstance(v, int) and 0 <= v <= 255 for v in value)):
                raise PartError(f"{self.kind}: zone {zone} colour must be [r, g, b] with 0-255 integers")
        if self.blazon and not k.blazon:
            raise PartError(f"{self.kind} takes no blazon")

    def to_json(self) -> dict:
        out: dict[str, Any] = {"kind": self.kind}
        if self.params:
            out["params"] = dict(self.params)
        if self.colours:
            out["colours"] = {z: (v if isinstance(v, str) else list(v)) for z, v in self.colours.items()}
        if self.blazon:
            out["blazon"] = self.blazon
        return out

    @classmethod
    def from_json(cls, raw: dict) -> "PartSpec":
        if not isinstance(raw, dict) or "kind" not in raw:
            raise PartError(f"a part must be an object with a \"kind\", not {raw!r}")
        spec = cls(raw["kind"], dict(raw.get("params") or {}), dict(raw.get("colours") or {}), raw.get("blazon"))
        spec.check()
        return spec


@dataclass
class ResolvedPart:
    """A spec with every value decided: defaults < fitting < the user's overrides."""
    kind: str
    role: str
    params: dict[str, Any]
    zones: list[Zone]
    fitted: dict[str, float]  # what fitting set (before the user's overrides)

    @property
    def shader(self) -> str:
        return shader_for(self.zones)

    @property
    def bone(self) -> str:
        return KINDS[self.kind].bone


def resolve(spec: PartSpec, role: str, body=None, points: dict[str, str] | None = None) -> ResolvedPart:
    """``body``: a core.fitting.BodyFit of the body this part attaches to (None: defaults).
    ``points``: role -> "up"/"down", what a weapon's "auto" point means (default DEFAULT_POINTS)."""
    from eu4unitgen.core.fitting import fitted_params

    spec.check()
    k = KINDS[spec.kind]
    if role not in k.roles:
        raise PartError(f"a {spec.kind} can't go on the {role} (only: {', '.join(k.roles)})")
    params: dict[str, Any] = {n: p.default for n, p in k.params.items()}
    fitted = fitted_params(spec.kind, body)
    params.update(fitted)
    params.update(spec.params)
    if params.get("point") == "auto":
        params["point"] = {**DEFAULT_POINTS, **(points or {})}[role]
    if "tip" in params and "butt" in params and params["tip"] - params["head_length"] <= 0:
        raise PartError(f"{spec.kind}: the head ({params['head_length']}) must start above the grip (tip {params['tip']})")
    zones = []
    for zd in k.zones:
        value = spec.colours.get(zd.name)
        if isinstance(value, str):
            zones.append(Zone(zd.colour, zd.gloss, value, zd.material))
        elif value is not None:
            zones.append(Zone(tuple(value), zd.gloss, None, zd.material))
        else:
            zones.append(Zone(zd.colour, zd.gloss, zd.channel, zd.material))
    return ResolvedPart(spec.kind, role, params, zones, fitted)


# --------------------------------------------------------------------------- building (background Blender)

@dataclass
class PartBuild:
    """One part mesh to build: ``name`` is its entity name and the stem of its other names."""
    name: str
    tier: int
    part: ResolvedPart
    blazon: Image.Image | None = None


@dataclass
class BuiltParts:
    meshes: list[dict]  # pdxmesh template records
    entities: list[dict]  # entity template records
    files: list[GeneratedFile]
    raw_meshes: dict[str, bytes] = field(default_factory=dict)  # entity name -> .mesh bytes (also in files)
    atlas: dict[str, dict] = field(default_factory=dict)  # entity name -> {"size", "density", "tris"} (atlas parts)
    maps: dict[str, dict] = field(default_factory=dict)  # entity name -> baked maps {ao, curv, up, pattern}: PNG bytes
    log: str = ""

    def images(self, b: "PartBuild") -> dict[str, Image.Image]:
        """The part's {diffuse, normal, specular} images: painted on its unwrap (atlas layout), else
        the swatch or disc layout."""
        p = b.part.params
        atlas = self.atlas.get(b.name)
        layout = KINDS[b.part.kind].layout
        if atlas is not None:
            imgs = atlas_images(b.part.zones, atlas["size"], atlas["tris"])
        else:
            layout = "strip" if layout == "atlas" else layout
            imgs = part_images(b.part.zones, layout, b.blazon,
                               int(p.get("blazon_rotate", 0)), p.get("blazon_mirror") == "yes")
        maps = self.maps.get(b.name)
        if maps:  # Q2: painted in vanilla's style from the baked maps
            size = imgs["diffuse"].size[0]
            zmap = zone_map(b.part.zones, layout, size, atlas["tris"] if atlas else None)
            imgs["diffuse"] = paint(imgs["diffuse"], zmap, b.part.zones, maps)
        return imgs


def _names(naming: Naming, set_name: str, b: PartBuild) -> dict[str, str]:
    kind, role = b.part.kind, b.part.role
    return {"diff": naming.part_texture(set_name, kind, b.tier, "diffuse", role),
            "n": naming.part_texture(set_name, kind, b.tier, "normal", role),
            "spec": naming.part_texture(set_name, kind, b.tier, "specular", role)}


def job_entry(naming: Naming, set_name: str, b: PartBuild) -> dict:
    """The Blender job's description of one part (no paths: also the cache key)."""
    k = KINDS[b.part.kind]
    return {"kind": b.part.kind, "builder": k.builder, "shape": f"{b.name}Shape", "shader": b.part.shader,
            "zones": len(b.part.zones), "layout": k.layout, "density": TEXEL_DENSITY,
            "materials": [z.material for z in b.part.zones], "bake": True,
            "bake_size": SIZE.get(k.layout, 64),
            "zone_uvs": [list(uv) for uv in zone_uvs(len(b.part.zones), k.layout)],
            "params": b.part.params, "textures": _names(naming, set_name, b)}


def _key(entry: dict) -> str:
    return hashlib.sha256(json.dumps(entry, sort_keys=True).encode("utf-8")).hexdigest()


def build_parts(runner, naming: Naming, set_name: str, builds: list[PartBuild], version: int,
                cache: dict[str, bytes] | None = None) -> BuiltParts:
    """Build ``builds`` in background Blender (``runner``: blender.runner.BlenderRunner) and make
    their textures. ``cache`` (job entry hash -> .mesh bytes) skips Blender for parts built before."""
    out = BuiltParts([], [], [])
    entries = [job_entry(naming, set_name, b) for b in builds]
    todo = [i for i, e in enumerate(entries) if cache is None or _key(e) not in cache]
    meshes: dict[int, bytes] = {i: cache[_key(e)] for i, e in enumerate(entries) if i not in todo}
    atlases: dict[int, dict] = {i: json.loads(cache[_key(e) + "#atlas"]) for i, e in enumerate(entries)
                                if i not in todo and _key(e) + "#atlas" in cache}
    baked: dict[int, dict] = {i: json.loads(cache[_key(e) + "#maps"]) for i, e in enumerate(entries)
                              if i not in todo and _key(e) + "#maps" in cache}
    if todo and runner is None:
        raise GenerationError("parts need Blender to build (Settings: Blender path)")
    if todo:
        with tempfile.TemporaryDirectory(prefix="eu4unitgen_parts_") as tmp:
            job = [{**entries[i], "out_mesh": str(Path(tmp) / f"{builds[i].name}.mesh")} for i in todo]
            res = runner.run_job("build_parts", {"parts": job}, timeout=300)
            out.log = res.log
            for i, built in zip(todo, res.result["parts"]):
                meshes[i] = strip_mesh_props(Path(built["out_mesh"]).read_bytes())
                if built.get("atlas"):
                    atlases[i] = built["atlas"]
                if built.get("maps"):
                    baked[i] = {k: base64.b64encode(Path(v).read_bytes()).decode("ascii") for k, v in built["maps"].items()}
                if cache is not None:
                    cache[_key(entries[i])] = meshes[i]
                    if built.get("atlas"):
                        cache[_key(entries[i]) + "#atlas"] = json.dumps(built["atlas"]).encode("utf-8")
                    if i in baked:
                        cache[_key(entries[i]) + "#maps"] = json.dumps(baked[i]).encode("utf-8")
    for i, b in enumerate(builds):
        kind, role, tier = b.part.kind, b.part.role, b.tier
        mesh_rel = naming.part_mesh_file(set_name, kind, tier, role)
        out.files.append(GeneratedFile(mesh_rel, meshes[i], OWNED, procedural=True))
        out.raw_meshes[b.name] = meshes[i]
        if i in atlases:
            out.atlas[b.name] = atlases[i]
        if i in baked:
            out.maps[b.name] = {k: base64.b64decode(v) for k, v in baked[i].items()}
        dds = encode_part_textures(out.images(b))
        for tex, data in dds.items():
            rel = f"{naming.part_dir(set_name)}/{naming.part_texture(set_name, kind, tier, tex, role)}"
            out.files.append(GeneratedFile(rel, data, OWNED, procedural=True))
        pdx = naming.part_pdxmesh(set_name, kind, tier, role)
        out.meshes.append({"name": pdx, "file": mesh_rel, "animations": {}, "scale": PART_SCALE, "extra": {},
                           "settings": []})
        out.entities.append({"name": b.name, "pdxmesh": pdx, "version": version, "states": [], "attaches": []})
    return out


def attach(entity: dict, bone: str, part: str) -> None:
    """Put ``part`` on ``entity``'s ``bone``, replacing whatever it attached there (e.g. the halberd)."""
    kept = []
    for a in entity["attaches"]:
        nodes = [(n, t) for n, t in a["nodes"] if n.lower() != bone.lower()]
        if nodes:
            kept.append({**a, "nodes": nodes})
    unnamed = next((a for a in kept if a["name"] is None), None)
    if unnamed is None:
        unnamed = {"name": None, "nodes": []}
        kept.append(unnamed)
    unnamed["nodes"] = unnamed["nodes"] + [(bone, part)]
    entity["attaches"] = kept
