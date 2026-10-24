"""Loadouts: which procedural parts each body of a unit set carries, per tier (SPEC §10 Phase 3).

A loadout is a JSON file in the user's project (never in this repo):

    {"schema": 1,
     "slots": {
       "1/infantry": {"stance": "auto",
                      "parts": [{"kind": "dory"},
                                {"kind": "hoplon", "colours": {"face": "country1"}, "blazon": "lion.png"},
                                {"kind": "crest", "params": {"orientation": "transverse"}, "colours": {"hair": [20, 20, 20]}}]},
       "1/rider": {"parts": [{"kind": "xiphos"}, {"kind": "crest"}]},
       "1/crew":  {"parts": [{"kind": "crest"}]}}}

Slots are ``<tier>/<role>``; roles are ``infantry`` (the infantry entity, which is also the gun
crew unless a crew slot exists), ``rider`` (the cavalry rider) and ``crew``. A crew slot gives
the gun crew its own entity (``TAG_ARTILLERY_n_crew``, the base infantry plus the crew's parts),
so a hoplite's shield needn't follow him to the guns. Each part kind has one bone
(core.parts.KINDS); a part replaces what the base attached there (the halberd, the sabre). A
hand weapon replaces the base's whole hand loadout: western tiers 2-4 carry their musket on
``Left_hand_node``, which a sarissa in the right hand mustn't leave behind.

Stance picks the animation set of the infantry or crew pdxmesh for idle/walk/attack (death and
retreat stay). ``auto`` follows what is carried: a hoplon -> ``spear_shield`` (vanilla's
indiangfx spear-and-shield set, same skeleton); a dory or sarissa -> ``polearm_twohanded`` (the
western halberd set, in-game test 7's dory); a xiphos -> ``polearm_onehanded``; nothing in the
hands -> the base culture's own (on western tiers 2-4 that is the musket set). Or name one.
Changing it makes a pdxmesh on the same vanilla .mesh (or edits the set's retextured one).

Loadout-wide settings (both optional):

    "bodies": {"2": "western_generic_01_type1", ...}     tier -> the base-game infantry pdxmesh its
        infantry, rider and crew wear, instead of the base culture's own for that tier (western tiers
        2-4 are 17th-18th century hat bodies). It must share the generic 28-bone skeleton; the
        texture recipe and fitting follow it.
    "weapon_point": {"rider": "up"}     role -> what a weapon's "auto" point means (up/down;
        default up for infantry and crews, down for riders).
    "derived": {"1": ["tail", "tuft"]}     tier -> skinned additions (core.derived_body.ADDITION_KINDS)
        appended to that tier's infantry body: a derived body (SPEC Track R). Or {"add": [...],
        "remove": ["helmet"]}: also drop named base parts (core.derived_body.REMOVALS, per base body:
        type1's helmet). Parts on that infantry are fitted to the derived body. The infantry only:
        the rider and a crew slot's crew keep the base body (a tail on a rider is untested).

Paths (blazons) are relative to the loadout file.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from eu4unitgen.core.generator import OWNED, GeneratedFile, GenerationError
from eu4unitgen.core.naming import LOADOUT_ROLES, TIERS, Naming
from eu4unitgen.core.naming import check_identifier
from eu4unitgen.core.parts import (
    DEFAULT_POINTS, KINDS, POINTS, PartBuild, PartError, PartSpec, attach, build_parts, resolve,
)

SCHEMA = 1
STANCE_ROLES = ("infantry", "crew")
STANCE_TYPES = {  # base-game animation types (1.37.5), same 28-bone skeleton as every generic infantry
    "spear_shield": {"idle": "spear_shield_idle_animation", "walk": "spear_shield_moving_animation",
                     "attack": "spear_shield_attack_animation"},
    "polearm_onehanded": {"idle": "polearm_onehanded_idle_animation", "walk": "polearm_onehanded_moving_animation",
                          "attack": "polearm_onehanded_attack_animation"},
    "polearm_twohanded": {"idle": "polearm_twohanded_idle_animation", "walk": "polearm_twohanded_moving_animation",
                          "attack": "polearm_twohanded_attack_animation"},
}
STANCES = ("auto", "base", *STANCE_TYPES)


class LoadoutError(ValueError):
    pass


@dataclass
class RoleLoadout:
    parts: list[PartSpec] = field(default_factory=list)
    stance: str = "auto"

    def resolved_stance(self) -> str:
        """``base`` (keep the pdxmesh's animations) or a STANCE_TYPES key."""
        if self.stance != "auto":
            return self.stance
        kinds = {p.kind for p in self.parts}
        if "hoplon" in kinds:
            return "spear_shield"
        if kinds & {"dory", "sarissa", "xyston"}:
            return "polearm_twohanded"
        if "xiphos" in kinds:
            return "polearm_onehanded"
        return "base"

    def to_json(self) -> dict:
        out: dict = {"parts": [p.to_json() for p in self.parts]}
        if self.stance != "auto":
            out["stance"] = self.stance
        return out


def parse_slot(key: str) -> tuple[int, str]:
    tier, _, role = str(key).partition("/")
    if not tier.isdigit() or int(tier) not in TIERS or role not in LOADOUT_ROLES:
        raise LoadoutError(f"slot {key!r} must be <tier>/<role>, tier {TIERS}, role {LOADOUT_ROLES}")
    return int(tier), role


def check_slot(tier: int, role: str, slot: RoleLoadout) -> None:
    where = f"slot {tier}/{role}"
    if slot.stance not in STANCES:
        raise LoadoutError(f"{where}: stance {slot.stance!r} must be one of {STANCES}")
    if slot.stance not in ("auto", "base") and role not in STANCE_ROLES:
        raise LoadoutError(f"{where}: only {' and '.join(STANCE_ROLES)} take a stance (riders use the horse_* set)")
    bones: dict[str, str] = {}
    for p in slot.parts:
        try:
            p.check()
        except PartError as exc:
            raise LoadoutError(f"{where}: {exc}") from exc
        k = KINDS[p.kind]
        if role not in k.roles:
            raise LoadoutError(f"{where}: a {p.kind} can't go on the {role} (only: {', '.join(k.roles)})")
        if k.bone in bones:
            raise LoadoutError(f"{where}: {bones[k.bone]} and {p.kind} both go on {k.bone}")
        bones[k.bone] = p.kind


HORSE_KEYS = {"geometry", "textures", "base", "rights", "source", "unit_colours"}


def check_horse(key, raw) -> dict:
    """A loadout's fresh horse (Track C, test 15): ``geometry`` (the fresh_horse job's JSON),
    ``textures`` {diffuse, specular, normal} (PNG paths), ``base`` (the base-game horse pdxmesh
    whose skeleton the geometry was built on; default cavalry_horse), and the reference's
    ``rights`` and ``source`` (CLAUDE.md rule 8: required, written to the notes); optionally ``unit_colours``
    ([[r, g, b] x 3]: the horse's cloth is the country colour's mask (SPEC M21) and these keep its look; the
    set's tags without colours of their own get them: export.pipeline)."""
    if not isinstance(raw, dict) or set(raw) - HORSE_KEYS:
        raise LoadoutError(f"horses: tier {key} must be an object with keys from {sorted(HORSE_KEYS)}")
    if not raw.get("geometry") or not isinstance(raw.get("textures"), dict) or             set(raw["textures"]) != {"diffuse", "specular", "normal"}:
        raise LoadoutError(f"horses: tier {key} needs geometry and textures {{diffuse, specular, normal}}")
    if not str(raw.get("rights") or "").strip():
        raise LoadoutError(f"horses: tier {key} needs the reference's rights (original, licensed or authorised; "
                           "CLAUDE.md rule 8)")
    uc = raw.get("unit_colours")
    if uc is not None and not (isinstance(uc, list) and len(uc) == 3 and all(
            isinstance(c, list) and len(c) == 3 and all(isinstance(x, int) and 0 <= x <= 255 for x in c) for c in uc)):
        raise LoadoutError(f"horses: tier {key}: unit_colours must be three [r, g, b] triples, 0-255")
    return {"base": "cavalry_horse", **raw}


def slot_warnings(slot: RoleLoadout) -> list[str]:
    """Hints for a slot as the form shows it (the validator checks the generated result)."""
    kinds = {KINDS[p.kind].bone: p.kind for p in slot.parts}
    if "Left_hand_node" in kinds and "Right_hand_node" not in kinds:
        return [f"a {kinds['Left_hand_node']} but no weapon part: the right hand keeps only what the base holds "
                "there, and western tiers 2-4 hold nothing there (their musket is in the left hand, which the "
                "shield replaces), so the unit carries no weapon"]
    return []


@dataclass
class Loadout:
    slots: dict[tuple[int, str], RoleLoadout] = field(default_factory=dict)
    root: Path = Path(".")
    bodies: dict[int, str] = field(default_factory=dict)  # tier -> base-game infantry pdxmesh
    points: dict[str, str] = field(default_factory=dict)  # role -> "up"/"down" for weapons left on "auto"
    derived: dict[int, dict] = field(default_factory=dict)  # tier -> {"add": [...], "remove": [...]} on its infantry body
    accept_clipping: list[str] = field(default_factory=list)  # addition kinds whose clipping (C2) is a known limit
    hair: dict = field(default_factory=dict)  # a character spec's "hair" (swatch colours); empty: the library's
    horses: dict[int, dict] = field(default_factory=dict)  # tier -> a fresh horse (core.fresh_mesh: geometry, textures, rights)

    # ------------------------------------------------------------------ files
    @classmethod
    def load(cls, path: Path) -> "Loadout":
        path = Path(path)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LoadoutError(f"can't read loadout {path}: {exc}") from exc
        return cls.from_json(doc, path.parent)

    @classmethod
    def from_json(cls, doc: dict, root: Path = Path(".")) -> "Loadout":
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
            raise LoadoutError(f"a loadout must be an object with \"schema\": {SCHEMA}")
        out = cls(root=Path(root))
        for key, raw in (doc.get("slots") or {}).items():
            tier, role = parse_slot(key)
            if not isinstance(raw, dict):
                raise LoadoutError(f"slot {key}: must be an object")
            try:
                parts = [PartSpec.from_json(p) for p in raw.get("parts") or []]
            except PartError as exc:
                raise LoadoutError(f"slot {key}: {exc}") from exc
            slot = RoleLoadout(parts, raw.get("stance", "auto"))
            check_slot(tier, role, slot)
            out.slots[(tier, role)] = slot
        for key, body in (doc.get("bodies") or {}).items():
            if not str(key).isdigit() or int(key) not in TIERS:
                raise LoadoutError(f"bodies: {key!r} must be a tier {TIERS}")
            try:
                out.bodies[int(key)] = check_identifier(body, "body pdxmesh")
            except ValueError as exc:
                raise LoadoutError(f"bodies: {exc}") from exc
        from eu4unitgen.core.derived_body import ADDITION_KINDS

        for key, raw in (doc.get("derived") or {}).items():
            if not str(key).isdigit() or int(key) not in TIERS:
                raise LoadoutError(f"derived: {key!r} must be a tier {TIERS}")
            spec = {"add": raw, "remove": []} if isinstance(raw, list) else raw
            if not isinstance(spec, dict) or set(spec) - {"add", "remove"}:
                raise LoadoutError(f"derived: tier {key} must be a list of additions or an object with add/remove "
                                   f"lists, not {raw!r}")
            add, remove = list(spec.get("add") or []), list(spec.get("remove") or [])
            if any(k not in ADDITION_KINDS for k in add) or len(set(add)) != len(add):
                raise LoadoutError(f"derived: tier {key} must list additions from {sorted(ADDITION_KINDS)}, "
                                   f"each once, not {add!r}")
            if not all(isinstance(r, str) for r in remove) or len(set(remove)) != len(remove) or not (add or remove):
                raise LoadoutError(f"derived: tier {key} needs additions or removals (each once), not {raw!r}")
            out.derived[int(key)] = {"add": add, "remove": remove}
        accept = doc.get("accept_clipping") or []
        if not isinstance(accept, list) or any(k not in ADDITION_KINDS for k in accept):
            raise LoadoutError(f"accept_clipping must list addition kinds from {sorted(ADDITION_KINDS)}, not {accept!r}")
        out.accept_clipping = list(accept)
        for key, raw in (doc.get("horses") or {}).items():
            if not str(key).isdigit() or int(key) not in TIERS:
                raise LoadoutError(f"horses: {key!r} must be a tier {TIERS}")
            out.horses[int(key)] = check_horse(key, raw)
        for role, point in (doc.get("weapon_point") or {}).items():
            if role not in LOADOUT_ROLES or point not in POINTS:
                raise LoadoutError(f"weapon_point: {role!r} must be one of {LOADOUT_ROLES} and {point!r} one of {POINTS}")
            out.points[role] = point
        return out

    def to_json(self) -> dict:
        out = {"schema": SCHEMA}
        if self.bodies:
            out["bodies"] = {str(t): b for t, b in sorted(self.bodies.items())}
        if self.derived:
            out["derived"] = {str(t): (list(d["add"]) if not d["remove"]
                                       else {"add": list(d["add"]), "remove": list(d["remove"])})
                              for t, d in sorted(self.derived.items())}
        if self.accept_clipping:
            out["accept_clipping"] = list(self.accept_clipping)
        if self.horses:
            out["horses"] = {str(t): dict(h) for t, h in sorted(self.horses.items())}
        points = {r: p for r, p in sorted(self.points.items()) if p != DEFAULT_POINTS[r]}
        if points:
            out["weapon_point"] = points
        out["slots"] = {f"{t}/{r}": s.to_json() for (t, r), s in sorted(self.slots.items())}
        return out

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ use
    def active(self) -> dict[tuple[int, str], RoleLoadout]:
        """Slots that change something. A crew slot always does (the crew gets its own entity)."""
        return {k: s for k, s in sorted(self.slots.items())
                if s.parts or s.resolved_stance() != "base" or k[1] == "crew"}

    def changes_anything(self) -> bool:
        return bool(self.active() or self.bodies or self.derived or self.horses)

    def needs_blender(self) -> bool:
        return any(d["add"] for d in self.derived.values()) or any(s.parts for s in self.active().values())

    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    def image(self, rel: str) -> Image.Image:
        try:
            with Image.open(self.path(rel)) as im:
                im.load()
                return im.copy()
        except OSError as exc:
            raise LoadoutError(f"can't open image {self.path(rel)}: {exc}") from exc


# --------------------------------------------------------------------------- applying a loadout to a set

@dataclass
class PartRecord:
    """What the validator needs to know about one generated part."""
    entity: str
    kind: str
    tier: int
    role: str
    bone: str
    shader: str
    mesh_file: str
    params: dict
    channels: list[str]  # country channels its zones use
    fitted: dict
    zones: dict = field(default_factory=dict)  # zone name -> (r, g, b) or a country channel
    blazon: str | None = None


@dataclass
class LoadoutResult:
    parts: list[PartRecord] = field(default_factory=list)
    crews: dict[str, str] = field(default_factory=dict)  # crew entity -> artillery entity that attaches it
    stances: dict[str, tuple[str, str]] = field(default_factory=dict)  # entity -> (stance, pdxmesh)
    files: list[GeneratedFile] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    bodies: dict = field(default_factory=dict)  # mesh file -> core.fitting.BodyFit
    slots: list[tuple[int, str]] = field(default_factory=list)  # the (tier, role) slots that were applied
    chosen_bodies: dict[int, str] = field(default_factory=dict)  # tier -> body pdxmesh the loadout picked
    carried: dict[tuple[int, str], list[str]] = field(default_factory=dict)  # slot -> part entities it carries
    derived: dict[int, "DerivedRecord"] = field(default_factory=dict)  # tier -> its infantry's derived body
    horses: dict = field(default_factory=dict)  # tier -> core.fresh_mesh.FreshHorse (rule fresh_mesh)
    texture_files: list[GeneratedFile] = field(default_factory=list)  # derived bodies' derived textures (Q5)


@dataclass
class DerivedRecord:
    """What the validator needs to know about one derived body (rule derived_body)."""
    entity: str
    pdxmesh: str
    file: str
    base_file: str
    additions: list[tuple[str, tuple[str, ...], int]]  # (kind, chain bones, vertex count), in file order
    removed: list[str] = field(default_factory=list)  # named base parts removed (core.derived_body.REMOVALS)
    deform: dict | None = None  # core.deform_check result on the infantry's idle/walk/attack (None: not run)
    textures: dict = field(default_factory=dict)  # kind -> derived texture path (Q5 hair swatch), or empty
    accept_clipping: list[str] = field(default_factory=list)  # kinds whose C2 findings are known limits


class _Meshes:
    """The set's generated pdxmesh records (retextured ones first), looked up with vanilla's."""

    def __init__(self, records: list[dict], idx):
        self.records, self.idx = records, idx
        self.by_name = {m["name"]: m for m in records}
        self.original: dict[str, dict] = {}  # pdxmesh -> its animations before any stance change

    def get(self, name: str) -> dict:
        rec = self.by_name.get(name) or self.idx.pdxmesh(name, base_only=True)
        if rec is None:
            raise GenerationError(f"pdxmesh {name!r} isn't a base-game or generated pdxmesh")
        self.original.setdefault(name, dict(rec.get("animations") or {}))
        return rec

    def anims(self, name: str) -> dict:
        return dict(self.get(name).get("animations") or {})

    def wanted(self, source: str, stance: str) -> dict:
        self.get(source)
        anims = dict(self.original[source])
        if stance in STANCE_TYPES:
            for t in STANCE_TYPES[stance].values():
                if self.idx.animation_type(t, base_only=True) is None:
                    raise GenerationError(f"stance {stance}: animation type {t!r} isn't in the base game")
            anims.update(STANCE_TYPES[stance])
        return anims

    def ensure_crew_id(self, source: str, new_name: str) -> str:
        """A chosen body without ``artillery_attack`` (every base-game body outside westerngfx) gets
        it, mapped to the body's own attack type as western bodies map it; the gun crew needs it."""
        anims = self.anims(source)
        if "artillery_attack" in anims or "attack" not in anims:
            return source
        name = self.with_anims(source, new_name, {**anims, "artillery_attack": anims["attack"]})
        self.original[name] = self.anims(name)
        return name

    def with_anims(self, source: str, new_name: str, anims: dict) -> str:
        """``source`` if it already has ``anims``; else ``new_name`` with them: the set's own record
        edited in place, or a new record on the same .mesh (scale, extra keys, meshsettings kept)."""
        if self.anims(source) == anims:
            return source
        if new_name in self.by_name:
            self.by_name[new_name]["animations"] = dict(anims)
            return new_name
        src = self.get(source)
        rec = {"name": new_name, "file": src["file"], "base": src.get("base", source), "animations": dict(anims),
               "scale": src.get("scale"), "extra": dict(src.get("extra") or {}),
               "settings": [dict(s) for s in src.get("settings") or []]}
        self.records.append(rec)
        self.by_name[new_name] = rec
        return new_name


HAND_NODES = ("right_hand_node", "left_hand_node")
BODY_BONES = ("head", "Right_hand_node", "Left_hand_node")  # what parts hang on


def check_body(name: str, game, idx) -> dict:
    """A body a loadout may pick: a base-game pdxmesh (never DLC, SPEC §9) on a base-game .mesh,
    with the infantry and rider animation ids. Returns its pdxmesh record."""
    rec = idx.pdxmesh(name, base_only=True)
    if rec is None:
        raise GenerationError(f"body {name!r} isn't a base-game pdxmesh (DLC bodies are never used)")
    if not rec.get("file") or not game.in_base_game(rec["file"]):
        raise GenerationError(f"body {name!r}: mesh {rec.get('file')!r} isn't in the base game")
    ids = set(rec.get("animations") or {})
    missing = sorted({"idle", "walk", "attack", "horse_idle", "horse_walk", "horse_attack"} - ids)
    if missing:
        raise GenerationError(f"body {name!r} lacks animation ids {missing}: not an infantry/rider body")
    return rec


def apply_bodies(naming: Naming, owner: str, entities: list[dict], loadout: Loadout, game, idx) -> list[str]:
    """Put each tier's chosen body on its infantry and rider (the crew is the infantry). Runs before
    retexturing, so a recipe retextures the chosen body. Returns notes."""
    by_name = {e["name"]: e for e in entities}
    notes = []
    for tier, body in sorted(loadout.bodies.items()):
        check_body(body, game, idx)
        for name in (naming.unit_entity(owner, "infantry", tier), naming.rider_entity(owner, tier)):
            by_name[name]["pdxmesh"] = body
        notes.append(f"tier {tier}: infantry, rider and crew wear {body} (loadout body)")
    return notes


@dataclass
class _Prebuilt:
    data: bytes
    adds: list
    removed: list[int]
    base_file: str
    fit: object  # core.fitting.BodyFit of the derived body (what parts on this infantry are fitted to)
    textures: dict = field(default_factory=dict)  # kind -> (bare name, DDS bytes, base texture path): Q5 hair swatch


def _prebuild(naming: Naming, set_name: str, tier: int, inf: dict, spec: dict, pdx: "_Meshes", game,
              runner, hair: dict | None = None, retextured: dict | None = None) -> _Prebuilt:
    """Build the tier's derived body before its parts are fitted (a removed helmet changes the head):
    the infantry pdxmesh's base-game .mesh without ``spec["remove"]`` parts and with ``spec["add"]``
    additions (core.derived_body)."""
    from eu4unitgen.core.derived_body import (
        DerivedBodyError, append_additions, build_derived, removed_triangles, retexture_material,
    )
    from eu4unitgen.core.fitting import measure_body

    src = pdx.get(inf["pdxmesh"])
    base_file = src.get("file") or ""
    # a texture recipe's retextured body: its meshsettings name the recipe's textures (generated, bare
    # names in the base mesh's folder, where the derived .mesh goes too); the derived body's material
    # names them instead, and a hair swatch is painted on top of them
    recipe_tex = {}
    for st in src.get("settings") or []:
        folder = base_file.rsplit("/", 1)[0]
        for kind in ("diffuse", "normal", "specular"):
            if st.get(kind):
                recipe_tex[kind] = f"{folder}/{st[kind]}"
    if not game.in_base_game(base_file):
        raise GenerationError(f"tier {tier}: derived bodies come from base-game meshes only, not {base_file!r}")
    if runner is None and spec["add"]:
        raise GenerationError("derived bodies need Blender to build their additions (Settings: Blender path)")
    base = game.read(base_file)
    textures: dict = {}
    try:
        removed = removed_triangles(base, base_file, spec["remove"]) if spec["remove"] else []
        if spec["add"]:
            swatch = _hair_swatch(base, base_file, removed, game, hair, recipe_tex, retextured or {})
            if swatch is None:
                data, adds = build_derived(runner, base, spec["add"], remove=removed, base_file=base_file)
            else:
                rect, size, images = swatch
                data, adds = build_derived(runner, base, spec["add"], remove=removed, swatch=rect, tex=size)
                names = {k: naming.derived_body_texture(set_name, tier, k) for k in images}
                data = retexture_material(data, names)
                textures = {k: (names[k], dds, path) for k, (dds, path) in images.items()}
        else:
            data, adds = append_additions(base, [], removed), []
        if recipe_tex and not textures:  # no swatch: the recipe's textures as they are
            data = retexture_material(data, {k: p.rsplit("/", 1)[-1] for k, p in recipe_tex.items()})
            textures = {k: (p.rsplit("/", 1)[-1], None, None) for k, p in recipe_tex.items()}
    except DerivedBodyError as exc:
        raise GenerationError(f"tier {tier}: derived body: {exc}") from exc
    file = naming.derived_body_file(set_name, tier, base_file)
    # fitted on the base after removals, without the additions (head-skinned ears would pass for the head)
    return _Prebuilt(data, adds, removed, base_file, measure_body(append_additions(base, [], removed), file), textures)


SWATCH_MIN_WIDTH = 8  # px; and at least an eighth of the texture tall: else keep the base texture's hair region


def _hair_swatch(base: bytes, base_file: str, removed: list[int], game, hair: dict | None = None,
                 recipe_tex: dict | None = None, retextured: dict | None = None):
    """Q5: derived copies of the base body's diffuse/normal/specular with a painted hair swatch in
    the largest texture area no remaining base triangle uses (a removed part's area is free too).
    -> (rect, size, {kind: (dds bytes, base texture path)}), or None when there's no room."""
    from eu4unitgen.core.derived_body import append_additions, free_rect
    from eu4unitgen.library.scanner import parse_pdx_mesh, summarize_mesh
    from eu4unitgen.textures.dds import DDSError, DDSFormat, read_dds, read_header, write_dds
    from eu4unitgen.textures.hair import paint_swatch

    shape = summarize_mesh(parse_pdx_mesh(base))["shapes"][0]
    folder = base_file.rsplit("/", 1)[0]
    paths = {k: f"{folder}/{shape[k]}" for k in ("diffuse", "normal", "specular") if shape.get(k)}
    raw = {k: game.read(p) for k, p in paths.items()}
    if len(raw) != 3 or any(v is None or not game.in_base_game(paths[k]) for k, v in raw.items()):
        return None
    for kind, path in (recipe_tex or {}).items():  # paint over a texture recipe's output; keep its base-game origin
        f = (retextured or {}).get(path)
        if f is not None and f.derived_from:
            raw[kind], paths[kind] = f.data, f.derived_from
    size = read_header(raw["diffuse"])["width"]
    rect = free_rect(append_additions(base, [], removed) if removed else base, size)
    if rect is None or rect[2] - rect[0] < SWATCH_MIN_WIDTH or rect[3] - rect[1] < size // 8:
        return None
    try:
        imgs = {k: read_dds(v) for k, v in raw.items()}
    except (OSError, ValueError, DDSError):  # undecodable base textures: keep the base texture's hair region
        return None
    scaled = {k: (im if im.size == (size, size) else im.resize((size, size))) for k, im in imgs.items()}
    d, n, s = paint_swatch(scaled["diffuse"], scaled["normal"], scaled["specular"], rect, hair)
    out = {}
    for k, im in (("diffuse", d), ("normal", n), ("specular", s)):
        fmt = DDSFormat.of(read_header(raw[k]))
        out[k] = (write_dds(im.resize((fmt.width, fmt.height)) if im.size != (fmt.width, fmt.height) else im, fmt),
                  paths[k])
    return rect, size, out


def _derive(naming: Naming, set_name: str, tier: int, inf: dict, spec: dict, built: _Prebuilt, pdx: "_Meshes",
            res: LoadoutResult, game=None, idx=None, runner=None, accept: list[str] | None = None) -> GeneratedFile:
    """Give ``inf`` (the tier's infantry entity, after its stance) its derived body, on a new pdxmesh
    with the same animations, scale and keys and no meshsettings (the copied material names the
    base game's textures, which resolve in the base body's folder)."""
    source = inf["pdxmesh"]
    src = pdx.get(source)
    file = naming.derived_body_file(set_name, tier, built.base_file)
    name = naming.derived_body_pdxmesh(set_name, tier)
    rec = {"name": name, "file": file, "base": src.get("base", source), "animations": dict(src.get("animations") or {}),
           "scale": src.get("scale"), "extra": dict(src.get("extra") or {}), "settings": []}
    pdx.records.append(rec)
    pdx.by_name[name] = rec
    inf["pdxmesh"] = name
    if inf["name"] in res.stances:
        res.stances[inf["name"]] = (res.stances[inf["name"]][0], name)
    deform = None
    if built.adds and runner is not None and idx is not None:
        from eu4unitgen.core.deform_check import anim_files, run_check, summary
        from eu4unitgen.core.derived_body import ADDITION_KINDS, base_vertex_count

        anims = [(state, game.read(path)) for state, path in anim_files(rec, idx)]
        deform = run_check(runner, built.data, base_vertex_count(built.data, built.adds), built.adds, ADDITION_KINDS,
                           [(s, d) for s, d in anims if d is not None])
        res.notes.append(f"tier {tier}: deformation check on {', '.join(s for s, _ in anims)}: {summary(deform)}")
    res.derived[tier] = DerivedRecord(
        inf["name"], name, file, built.base_file, [(a.name, a.chain, a.count) for a in built.adds],
        removed=list(spec["remove"]), deform=deform,
        accept_clipping=[k for k in (accept or []) if k in {a.name for a in built.adds}],
        textures={k: f"{file.rsplit('/', 1)[0]}/{tex}" for k, (tex, _, _) in built.textures.items()})
    what = [f"- {r}" for r in spec["remove"]] + [f"+ {k}" for k in spec["add"]]
    res.notes.append(f"tier {tier}: {inf['name']} wears a derived body ({file}: {built.base_file} {' '.join(what)})")
    folder = file.rsplit("/", 1)[0]
    res.texture_files += [GeneratedFile(f"{folder}/{name}", dds, OWNED, derived_from=base_path)
                          for name, dds, base_path in built.textures.values() if dds is not None]
    return GeneratedFile(file, built.data, OWNED, derived_from=built.base_file)


def _drop_hand_attaches(entity: dict) -> list[str]:
    """Remove the base's hand-held items (a weapon part replaces the whole hand loadout); returns them."""
    dropped, kept = [], []
    for a in entity["attaches"]:
        if a["name"] is None:
            dropped += [t for n, t in a["nodes"] if n.lower() in HAND_NODES]
            nodes = [(n, t) for n, t in a["nodes"] if n.lower() not in HAND_NODES]
            if nodes:
                kept.append({**a, "nodes": nodes})
        else:
            kept.append(a)
    entity["attaches"] = kept
    return dropped


def apply_loadout(naming: Naming, set_name: str, owner: str, entities: list[dict], meshes: list[dict],
                  loadout: Loadout, game, idx, version: int, runner, cache: dict | None = None,
                  retextured: dict | None = None) -> LoadoutResult:
    """Dress ``owner``'s entities (from ``unit_entities``, after retexturing) with ``loadout``.
    ``entities`` and ``meshes`` (the set's generated pdxmesh records) are changed in place: part
    entities, crew entities and stance pdxmeshes are added. Parts are built in background Blender
    through ``runner``; identical parts at one tier are built once and shared between roles."""
    from eu4unitgen.core.fitting import measure_body

    res = LoadoutResult(chosen_bodies=dict(loadout.bodies))
    by_name = {e["name"]: e for e in entities}
    pdx = _Meshes(meshes, idx)

    def body(pdx_name: str):
        file = pdx.get(pdx_name)["file"]
        if file not in res.bodies:
            data = game.read(file)
            if data is None:
                raise GenerationError(f"body mesh {file} not found in the EU4 install")
            res.bodies[file] = measure_body(data, file)
            res.notes += [f"fitting: {n}" for n in res.bodies[file].notes]
        return res.bodies[file]

    builds: list[PartBuild] = []
    derived_files: list[GeneratedFile] = []
    shared: dict[str, str] = {}  # part identity -> entity
    assigned: list[tuple[str, str, str]] = []  # (entity, bone, part entity)
    active = loadout.active()
    for tier in TIERS:
        inf = by_name[naming.unit_entity(owner, "infantry", tier)]
        if tier in loadout.bodies:
            before = inf["pdxmesh"]
            inf["pdxmesh"] = pdx.ensure_crew_id(before, naming.pdxmesh(set_name, "infantry", tier))
            if inf["pdxmesh"] != before:
                res.notes.append(f"tier {tier}: {loadout.bodies[tier]} has no artillery_attack; "
                                 f"{inf['pdxmesh']} adds it (= its attack) for the gun crew")
        base_inf = copy.deepcopy(inf)
        prebuilt = (_prebuild(naming, set_name, tier, inf, loadout.derived[tier], pdx, game, runner, loadout.hair,
                              retextured or {})
                    if tier in loadout.derived else None)
        for role in ("infantry", "crew", "rider"):
            slot = active.get((tier, role))
            if slot is None:
                continue
            res.slots.append((tier, role))
            if role == "crew":
                crew = naming.crew_entity(owner, tier)
                art = by_name[naming.unit_entity(owner, "artillery", tier)]
                for a in art["attaches"]:
                    if a["name"] == "gunman":
                        a["nodes"] = [(n, crew if n == "infantry_node" else t) for n, t in a["nodes"]]
                entities.append({**copy.deepcopy(base_inf), "name": crew})
                by_name[crew] = entities[-1]
                res.crews[crew] = art["name"]
                target = crew
            else:
                target = inf["name"] if role == "infantry" else naming.rider_entity(owner, tier)
            fit = prebuilt.fit if prebuilt is not None and role == "infantry" else body(by_name[target]["pdxmesh"])
            weapon = next((p.kind for p in slot.parts if KINDS[p.kind].bone == "Right_hand_node"), None)
            if weapon is not None:
                dropped = _drop_hand_attaches(by_name[target])
                if dropped:
                    res.notes.append(f"{target}: the {weapon} replaces the base's {', '.join(dropped)}")
            for spec in slot.parts:
                try:
                    part = resolve(spec, role, fit, loadout.points)
                except PartError as exc:
                    raise GenerationError(f"loadout slot {tier}/{role}: {exc}") from exc
                identity = json.dumps([tier, spec.kind, part.params, [(z.colour, z.gloss, z.channel) for z in part.zones],
                                       spec.blazon, spec.mesh], sort_keys=True, default=list)
                name = shared.get(identity)
                if name is None:
                    name = naming.part(set_name, spec.kind, tier, role)
                    shared[identity] = name
                    blazon = loadout.image(spec.blazon) if spec.blazon else None
                    own = None
                    if spec.mesh:  # M23: geometry and textures of its own
                        try:
                            geometry = json.loads(loadout.path(spec.mesh["geometry"]).read_text(encoding="utf-8"))
                        except (OSError, ValueError) as exc:
                            raise GenerationError(f"loadout slot {tier}/{role}: {spec.kind}: can't read its geometry: {exc}") from exc
                        own = {"geometry": {k: geometry[k] for k in ("p", "tri", "uv")},
                               "images": {k: loadout.image(spec.mesh[k]) for k in ("diffuse", "normal", "specular")}}
                    builds.append(PartBuild(name, tier, part, blazon, own))
                    res.parts.append(PartRecord(
                        name, spec.kind, tier, role, part.bone, part.shader,
                        naming.part_mesh_file(set_name, spec.kind, tier, role), part.params,
                        sorted({z.channel for z in part.zones if z.channel}), part.fitted,
                        {zd.name: z.channel or z.colour for zd, z in zip(KINDS[spec.kind].zones, part.zones)},
                        spec.blazon))
                assigned.append((target, part.bone, name))
                res.carried.setdefault((tier, role), []).append(name)
            if role in STANCE_ROLES:
                stance = slot.resolved_stance()
                source = base_inf["pdxmesh"]
                want = pdx.wanted(source, stance)
                if role == "infantry":
                    inf["pdxmesh"] = pdx.with_anims(source, naming.pdxmesh(set_name, "infantry", tier), want)
                    chosen = inf["pdxmesh"]
                elif pdx.anims(inf["pdxmesh"]) == want:  # the crew moves like the infantry: share its pdxmesh
                    chosen = by_name[target]["pdxmesh"] = inf["pdxmesh"]
                else:
                    chosen = by_name[target]["pdxmesh"] = pdx.with_anims(
                        source, naming.pdxmesh(set_name, "crew", tier), want)
                if stance != "base":
                    res.stances[target] = (stance, chosen)
        if prebuilt is not None:
            derived_files.append(_derive(naming, set_name, tier, inf, loadout.derived[tier], prebuilt, pdx, res,
                                         game, idx, runner, loadout.accept_clipping))
    if loadout.horses:
        from eu4unitgen.core.fresh_mesh import apply_horses

        files, res.horses, notes = apply_horses(naming, set_name, owner, by_name, pdx.records, pdx.by_name, loadout,
                                                game, idx)
        derived_files += files
        res.notes += notes
    built = build_parts(runner, naming, set_name, builds, version, cache)
    for target, bone, name in assigned:
        attach(by_name[target], bone, name)
    meshes.extend(built.meshes)
    entities.extend(built.entities)
    res.files = built.files + derived_files + res.texture_files
    for file, fit in sorted(res.bodies.items()):
        head = f", head top {fit.head['top']} above the joint" if fit.head else ""
        res.notes.append(f"fitting: parts sized for {file} (stature {fit.stature}, scale {fit.scale:.3f}{head})")
    return res
