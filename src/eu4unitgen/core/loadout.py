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

Paths (blazons) are relative to the loadout file.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from eu4unitgen.core.generator import GeneratedFile, GenerationError
from eu4unitgen.core.naming import LOADOUT_ROLES, TIERS, Naming
from eu4unitgen.core.parts import KINDS, PartBuild, PartError, PartSpec, attach, build_parts, resolve

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
        if kinds & {"dory", "sarissa"}:
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


@dataclass
class Loadout:
    slots: dict[tuple[int, str], RoleLoadout] = field(default_factory=dict)
    root: Path = Path(".")

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
        return out

    def to_json(self) -> dict:
        return {"schema": SCHEMA, "slots": {f"{t}/{r}": s.to_json() for (t, r), s in sorted(self.slots.items())}}

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ use
    def active(self) -> dict[tuple[int, str], RoleLoadout]:
        """Slots that change something. A crew slot always does (the crew gets its own entity)."""
        return {k: s for k, s in sorted(self.slots.items())
                if s.parts or s.resolved_stance() != "base" or k[1] == "crew"}

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
    carried: dict[tuple[int, str], list[str]] = field(default_factory=dict)  # slot -> part entities it carries


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
                  loadout: Loadout, game, idx, version: int, runner, cache: dict | None = None) -> LoadoutResult:
    """Dress ``owner``'s entities (from ``unit_entities``, after retexturing) with ``loadout``.
    ``entities`` and ``meshes`` (the set's generated pdxmesh records) are changed in place: part
    entities, crew entities and stance pdxmeshes are added. Parts are built in background Blender
    through ``runner``; identical parts at one tier are built once and shared between roles."""
    from eu4unitgen.core.fitting import measure_body

    res = LoadoutResult()
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
    shared: dict[str, str] = {}  # part identity -> entity
    assigned: list[tuple[str, str, str]] = []  # (entity, bone, part entity)
    active = loadout.active()
    for tier in TIERS:
        inf = by_name[naming.unit_entity(owner, "infantry", tier)]
        base_inf = copy.deepcopy(inf)
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
            fit = body(by_name[target]["pdxmesh"])
            weapon = next((p.kind for p in slot.parts if KINDS[p.kind].bone == "Right_hand_node"), None)
            if weapon is not None:
                dropped = _drop_hand_attaches(by_name[target])
                if dropped:
                    res.notes.append(f"{target}: the {weapon} replaces the base's {', '.join(dropped)}")
            for spec in slot.parts:
                try:
                    part = resolve(spec, role, fit)
                except PartError as exc:
                    raise GenerationError(f"loadout slot {tier}/{role}: {exc}") from exc
                identity = json.dumps([tier, spec.kind, part.params, [(z.colour, z.gloss, z.channel) for z in part.zones],
                                       spec.blazon], sort_keys=True, default=list)
                name = shared.get(identity)
                if name is None:
                    name = naming.part(set_name, spec.kind, tier, role)
                    shared[identity] = name
                    blazon = loadout.image(spec.blazon) if spec.blazon else None
                    builds.append(PartBuild(name, tier, part, blazon))
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
    built = build_parts(runner, naming, set_name, builds, version, cache)
    for target, bone, name in assigned:
        attach(by_name[target], bone, name)
    meshes.extend(built.meshes)
    entities.extend(built.entities)
    res.files = built.files
    for file, fit in sorted(res.bodies.items()):
        head = f", head top {fit.head['top']} above the joint" if fit.head else ""
        res.notes.append(f"fitting: parts sized for {file} (stature {fit.stature}, scale {fit.scale:.3f}{head})")
    return res
