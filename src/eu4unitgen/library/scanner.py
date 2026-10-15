"""Index the user's EU4 install (base game + DLC zips) and build base_models.json.

Everything is read in place. Nothing from the game is copied: the output holds only
names, paths, bone names and texture-header facts.

Script text is read with pdxscript.parser.

Usage:
    python -m eu4unitgen.library.scanner --eu4 "<EU4 install>" --out base_models.json
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from eu4unitgen.pdxscript.parser import Block, Scalar, compact, parse_bytes

SCHEMA_VERSION = 2
COLOUR_SHADER = "PdxMeshColor"
UNIT_TYPES = ("infantry", "cavalry", "artillery")


def _num(s: Any) -> int | float | None:
    try:
        return int(s)
    except (TypeError, ValueError):
        try:
            return float(s)
        except (TypeError, ValueError):
            return None


# --------------------------------------------------------------------------- binary .mesh

def parse_pdx_mesh(data: bytes) -> dict:
    """Parse Paradox's binary mesh container (@@b@) into a node tree.

    Layout as implemented by ross-g/io_pdx_mesh: '!' property = len-prefixed name,
    type char (i/f/s), int32 count, payload; '[' * depth + NUL-terminated name = object.
    """
    if data[:4] != b"@@b@":
        raise ValueError("not a PDX binary mesh (missing @@b@ header)")
    pos = 4
    root: dict = {"name": "root", "props": {}, "children": []}
    stack = [root]
    while pos < len(data):
        marker = data[pos:pos + 1]
        if marker == b"!":
            pos += 1
            name_len = data[pos]
            pos += 1
            name = data[pos:pos + name_len].decode("latin-1")
            pos += name_len
            dtype = chr(data[pos])
            pos += 1
            (count,) = struct.unpack_from("<i", data, pos)
            pos += 4
            if dtype == "i":
                val: list = list(struct.unpack_from(f"<{count}i", data, pos))
                pos += 4 * count
            elif dtype == "f":
                val = list(struct.unpack_from(f"<{count}f", data, pos))
                pos += 4 * count
            elif dtype == "s":
                val = []
                for _ in range(count):
                    (slen,) = struct.unpack_from("<i", data, pos)
                    pos += 4
                    val.append(data[pos:pos + slen].rstrip(b"\0").decode("latin-1"))
                    pos += slen
            else:
                raise ValueError(f"unknown property type {dtype!r} at byte {pos}")
            stack[-1]["props"][name] = val
        elif marker == b"[":
            depth = 0
            while data[pos:pos + 1] == b"[":
                depth += 1
                pos += 1
            end = data.index(b"\0", pos)
            node = {"name": data[pos:end].decode("latin-1"), "props": {}, "children": []}
            pos = end + 1
            del stack[depth:]
            stack[-1]["children"].append(node)
            stack.append(node)
        else:
            raise ValueError(f"unexpected byte {marker!r} at {pos}")
    return root


def write_pdx_mesh(tree: dict) -> bytes:
    """The inverse of ``parse_pdx_mesh`` (byte-identical on every vanilla unit mesh; tested).
    Property types: ints stay 'i', floats 'f', strings 's'."""
    out = bytearray(b"@@b@")

    def props(node: dict) -> None:
        for name, vals in node["props"].items():
            raw = name.encode("latin-1")
            if vals and all(isinstance(v, str) for v in vals):
                dtype, body = b"s", b"".join(struct.pack("<i", len(v.encode("latin-1")) + 1) + v.encode("latin-1") + b"\0"
                                             for v in vals)
            elif vals and all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
                dtype, body = b"i", struct.pack(f"<{len(vals)}i", *vals)
            else:
                dtype, body = b"f", struct.pack(f"<{len(vals)}f", *vals)
            out.extend(b"!" + bytes([len(raw)]) + raw + dtype + struct.pack("<i", len(vals)) + body)

    def nodes(node: dict, depth: int) -> None:
        for child in node["children"]:
            out.extend(b"[" * depth + child["name"].encode("latin-1") + b"\0")
            props(child)
            nodes(child, depth + 1)

    props(tree)
    nodes(tree, 1)
    return bytes(out)


def strip_mesh_props(data: bytes, names: tuple[str, ...] = ("boundingsphere",)) -> bytes:
    """Drop properties vanilla EU4 meshes never carry (io_pdx_mesh 0.91 always writes
    ``boundingsphere``; none of the 167 vanilla unit meshes has it)."""
    tree = parse_pdx_mesh(data)
    for node in _walk(tree):
        for n in names:
            node["props"].pop(n, None)
    return write_pdx_mesh(tree)


def _walk(node: dict) -> Iterator[dict]:
    yield node
    for child in node["children"]:
        yield from _walk(child)


def bind_pose(tx: list[float]) -> list[float]:
    """A bone's ``tx`` (inverse bind matrix; columns tx[0:3], tx[3:6], tx[6:9], translation
    tx[9:12], as io_pdx_mesh reads it) -> its bind pose in mesh space (Y-up), as a row-major
    3x4 [r00 r01 r02 t0, r10 r11 r12 t1, r20 r21 r22 t2]."""
    a = [[tx[0], tx[3], tx[6]], [tx[1], tx[4], tx[7]], [tx[2], tx[5], tx[8]]]
    t = tx[9:12]
    det = (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
           + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    if abs(det) < 1e-12:
        raise ValueError("singular bone transform")
    inv = [[(a[(j + 1) % 3][(i + 1) % 3] * a[(j + 2) % 3][(i + 2) % 3]
             - a[(j + 1) % 3][(i + 2) % 3] * a[(j + 2) % 3][(i + 1) % 3]) / det for j in range(3)] for i in range(3)]
    pos = [-sum(inv[r][c] * t[c] for c in range(3)) for r in range(3)]
    return [v for r in range(3) for v in (*inv[r], pos[r])]


def summarize_mesh(tree: dict) -> dict:
    """Shapes (shader + texture names), bone names in file order, locators with parent bone,
    the skeleton (parent and bind pose per bone) and the bounding box over all shapes."""
    shapes, bones, locators, skeleton = [], [], {}, {}
    lo, hi = [float("inf")] * 3, [float("-inf")] * 3
    for node in _walk(tree):
        if node["name"] == "object":
            for shape in node["children"]:
                for part in shape["children"]:
                    for box in (c for c in part["children"] if c["name"] == "aabb"):
                        mn, mx = box["props"].get("min"), box["props"].get("max")
                        if mn and mx:
                            lo = [min(a, b) for a, b in zip(lo, mn)]
                            hi = [max(a, b) for a, b in zip(hi, mx)]
                    if part["name"] == "skeleton" and not skeleton:
                        by_index = {(b["props"].get("ix") or [None])[0]: b["name"] for b in part["children"]}
                        for b in part["children"]:
                            pa = (b["props"].get("pa") or [None])[0]
                            tx = b["props"].get("tx")
                            try:  # vanilla attach_free has an all-zero tx: no bind pose
                                bind = bind_pose(tx) if tx and len(tx) == 12 else None
                            except ValueError:
                                bind = None
                            skeleton[b["name"]] = {"parent": by_index.get(pa) if pa is not None else None,
                                                   "bind": bind}
                    for mat in (c for c in part["children"] if c["name"] == "material"):
                        p = mat["props"]
                        shapes.append({
                            "shape": shape["name"],
                            "shader": (p.get("shader") or [None])[0],
                            "diffuse": (p.get("diff") or [None])[0],
                            "normal": (p.get("n") or [None])[0],
                            "specular": (p.get("spec") or [None])[0],
                        })
                    if part["name"] == "skeleton" and not bones:
                        bones = [b["name"] for b in part["children"]]
        elif node["name"] == "locator":
            for loc in node["children"]:
                locators[loc["name"]] = (loc["props"].get("pa") or [None])[0]
    aabb = {"min": lo, "max": hi} if lo[0] != float("inf") else None
    return {"shapes": shapes, "bones": bones, "locators": locators, "skeleton": skeleton, "aabb": aabb}


# --------------------------------------------------------------------------- DDS header

def read_dds_header(data: bytes) -> dict:
    if data[:4] != b"DDS ":
        raise ValueError("not a DDS file")
    height, width = struct.unpack_from("<2I", data, 12)
    (mips,) = struct.unpack_from("<I", data, 28)
    (pf_flags,) = struct.unpack_from("<I", data, 80)
    fourcc = data[84:88]
    if pf_flags & 0x4:  # DDPF_FOURCC
        fmt = fourcc.decode("latin-1")
    else:
        (bits,) = struct.unpack_from("<I", data, 88)
        fmt = f"uncompressed_{bits}bpp"
    header = {"width": width, "height": height, "mips": max(mips, 1), "format": fmt, "dx10_header": fourcc == b"DX10"}
    if header["dx10_header"]:
        (header["dxgi_format"],) = struct.unpack_from("<I", data, 128)
    return header


# --------------------------------------------------------------------------- file access

class GameFiles:
    """Read-only, case-insensitive access to the base game plus DLC zips.

    Paths in game scripts are written with inconsistent case ("Units" vs "units");
    Windows ignores it, Linux doesn't, so every lookup goes through a lowercase map.
    """

    def __init__(self, root: Path, include_dlc: bool = True):
        self.root = Path(root)
        self._dir_cache: dict[Path, dict[str, str]] = {}
        self.zips: dict[str, zipfile.ZipFile] = {}
        self._zip_index: dict[str, list[tuple[str, str]]] = {}  # lower path -> [(dlc, member)]
        if include_dlc and (self.root / "dlc").is_dir():
            for zpath in sorted((self.root / "dlc").glob("*/*.zip")):
                dlc = zpath.parent.name
                zf = zipfile.ZipFile(zpath)
                self.zips[dlc] = zf
                for member in zf.namelist():
                    if not member.endswith("/"):
                        self._zip_index.setdefault(member.lower(), []).append((dlc, member))

    def close(self) -> None:
        for zf in self.zips.values():
            zf.close()

    def _resolve_base(self, rel: str) -> Path | None:
        cur = self.root
        for part in PurePosixPath(rel.replace("\\", "/")).parts:
            if cur not in self._dir_cache:
                try:
                    self._dir_cache[cur] = {p.name.lower(): p.name for p in cur.iterdir()}
                except (FileNotFoundError, NotADirectoryError):
                    return None
            real = self._dir_cache[cur].get(part.lower())
            if real is None:
                return None
            cur = cur / real
        return cur if cur.is_file() else None

    def read(self, rel: str) -> bytes | None:
        """Base game first, then the first DLC zip (alphabetical) that has the file."""
        path = self._resolve_base(rel)
        if path is not None:
            return path.read_bytes()
        hits = self._zip_index.get(rel.replace("\\", "/").lower(), [])
        if hits:
            dlc, member = hits[0]
            return self.zips[dlc].read(member)
        return None

    def in_base_game(self, rel: str) -> bool:
        """The file is in the base game folder (not only in a DLC zip)."""
        return self._resolve_base(rel) is not None

    def exists(self, rel: str) -> bool:
        if self._resolve_base(rel) is not None:
            return True
        return bool(self._zip_index.get(rel.replace("\\", "/").lower()))

    def iter_files(self, rel_dir: str, suffixes: tuple[str, ...], recursive: bool = False) -> Iterator[tuple[str, str | None, bytes]]:
        """Yield (path relative to root, dlc or None, bytes) for base game then DLC zips."""
        base_dir = self._resolve_base_dir(rel_dir)
        if base_dir is not None:
            files = base_dir.rglob("*") if recursive else base_dir.iterdir()
            for p in sorted(files):
                if p.is_file() and p.suffix.lower() in suffixes:
                    yield p.relative_to(self.root).as_posix(), None, p.read_bytes()
        prefix = rel_dir.rstrip("/").lower() + "/"
        for dlc, zf in self.zips.items():
            for member in sorted(zf.namelist()):
                low = member.lower()
                if not low.startswith(prefix) or not low.endswith(suffixes):
                    continue
                if not recursive and "/" in low[len(prefix):]:
                    continue
                yield member, dlc, zf.read(member)

    def _resolve_base_dir(self, rel: str) -> Path | None:
        cur = self.root
        for part in PurePosixPath(rel).parts:
            if not cur.is_dir():
                return None
            match = next((p for p in cur.iterdir() if p.name.lower() == part.lower()), None)
            if match is None:
                return None
            cur = match
        return cur if cur.is_dir() else None


# --------------------------------------------------------------------------- index

@dataclass
class Index:
    entities: dict[str, list[dict]] = field(default_factory=dict)
    pdxmeshes: dict[str, list[dict]] = field(default_factory=dict)
    animation_types: dict[str, list[dict]] = field(default_factory=dict)
    sprite_packs: dict[str, list[dict]] = field(default_factory=dict)
    pack_order: dict[str, list[dict]] = field(default_factory=dict)
    graphical_cultures: list[str] = field(default_factory=list)
    sprite_levels: dict[str, dict] = field(default_factory=dict)
    game_version: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def _pick(defs: list[dict], base_only: bool) -> dict | None:
        pool = [d for d in defs if d["dlc"] is None] if base_only else defs
        if not pool:
            return None
        # Highest version wins; missing version treated as 0 (UNVERIFIED default).
        # Among equal versions the later-loaded one is kept (UNVERIFIED tie rule).
        return max(enumerate(pool), key=lambda e: ((e[1].get("version") or 0), e[0]))[1]

    def entity(self, name: str, base_only: bool = True) -> dict | None:
        return self._pick(self.entities.get(name, []), base_only)

    def pdxmesh(self, name: str, base_only: bool = True) -> dict | None:
        return self._pick(self.pdxmeshes.get(name, []), base_only)

    def animation_type(self, name: str, base_only: bool = True) -> dict | None:
        return self._pick(self.animation_types.get(name, []), base_only)


def _blocks(items: Block, key: str) -> list[Block]:
    return [v for v in items.get_all(key) if isinstance(v, Block)]


def _entity_record(items: Block, path: str, dlc: str | None) -> dict:
    rec: dict[str, Any] = {"name": items.get_str("name"), "defined_in": path, "dlc": dlc}
    for key in ("clone", "pdxmesh"):
        if items.get_str(key) is not None:
            rec[key] = items.get_str(key)
    if items.get_str("version") is not None:
        rec["version"] = _num(items.get_str("version"))
    states = []
    for st in _blocks(items, "state"):
        state: dict[str, Any] = {"name": st.get_str("name")}
        if st.get_str("animation") is not None:
            state["animation"] = st.get_str("animation")
        events = [{k: (_num(v) if k == "time" else v) for k, v in ev.scalar_map().items()} for ev in _blocks(st, "event")]
        if events:
            state["events"] = events
        states.append(state)
    if states:
        rec["states"] = states
    attaches = [a.scalar_map() for a in _blocks(items, "attach")]
    if attaches:
        rec["attach"] = attaches
    known = {"name", "clone", "pdxmesh", "version", "state", "attach"}
    extra = {p.key.value: p.value.value for p in items.pairs() if p.key.value not in known and isinstance(p.value, Scalar)}
    if extra:
        rec["extra"] = extra
    return rec


def _pdxmesh_record(items: Block, path: str, dlc: str | None) -> dict:
    rec = {"name": items.get_str("name"), "defined_in": path, "dlc": dlc, "file": items.get_str("file"),
           "scale": _num(items.get_str("scale")),
           "animations": {a.get_str("id"): a.get_str("type") for a in _blocks(items, "animation")}}
    known = {"name", "file", "scale", "animation", "meshsettings"}
    extra = {p.key.value: p.value.value for p in items.pairs() if p.key.value not in known and isinstance(p.value, Scalar)}
    if extra:
        rec["extra"] = extra  # e.g. cull_distance
    settings = [m.scalar_map() for m in _blocks(items, "meshsettings")]
    if settings:
        rec["meshsettings"] = settings  # per-shape texture/shader override (vanilla: dharma_mapitems.gfx)
    return rec


def _sprite_pack_record(name: str, items: Block, path: str, dlc: str | None) -> dict:
    rec: dict[str, Any] = {"name": name, "defined_in": path, "dlc": dlc}
    trig = items.get_block("country_trigger")
    rec["country_trigger"] = compact(trig) if trig is not None else None
    gcs = items.get_block("graphical_cultures")
    if gcs is not None:
        rec["graphical_cultures"] = gcs.bare()
    for utype in UNIT_TYPES:
        sprites = items.get_block(f"sprite_{utype}")
        if sprites is not None:
            rec[utype] = sprites.scalar_map()
    return rec


def build_index(files: GameFiles) -> Index:
    idx = Index()

    for path, dlc, data in files.iter_files("gfx/entities", (".asset",)):
        for seq, val in enumerate(_blocks(parse_bytes(data), "entity")):
            if val.get_str("name"):
                rec = _entity_record(val, path, dlc)
                rec["seq"] = seq  # position in the file: load order within it (clone resolution)
                idx.entities.setdefault(rec["name"], []).append(rec)

    for path, dlc, data in files.iter_files("interface", (".gfx",), recursive=True):
        for obj_types in _blocks(parse_bytes(data), "objectTypes"):
            for pdx in _blocks(obj_types, "pdxmesh"):
                if pdx.get_str("name"):
                    rec = _pdxmesh_record(pdx, path, dlc)
                    idx.pdxmeshes.setdefault(rec["name"], []).append(rec)

    for path, dlc, data in files.iter_files("gfx/models", (".asset",), recursive=True):
        folder = PurePosixPath(path).parent
        for val in _blocks(parse_bytes(data), "animation"):
            if val.get_str("name") and val.get_str("file"):
                rec = {"name": val.get_str("name"), "defined_in": path, "dlc": dlc,
                       "file": str(folder / val.get_str("file"))}
                idx.animation_types.setdefault(rec["name"], []).append(rec)

    for path, dlc, data in files.iter_files("gfx/sprite_packs", (".txt",)):
        for p in parse_bytes(data).pairs():
            if isinstance(p.value, Block):
                idx.sprite_packs.setdefault(p.key.value, []).append(_sprite_pack_record(p.key.value, p.value, path, dlc))

    for path, dlc, data in files.iter_files("gfx/sprite_packs_order", (".txt",)):
        for p in parse_bytes(data).pairs():
            if isinstance(p.value, Scalar) and _num(p.value.value) is not None:
                idx.pack_order.setdefault(p.key.value, []).append({"order": _num(p.value.value), "defined_in": path, "dlc": dlc})

    gct = files.read("common/graphicalculturetype.txt")
    if gct:
        idx.graphical_cultures = parse_bytes(gct).bare()

    mil = files.read("common/technologies/mil.txt")
    if mil:
        for n, tech in enumerate(_blocks(parse_bytes(mil), "technology")):
            level = tech.get_str("sprite_level")
            if level is not None:
                idx.sprite_levels[str(level)] = {"mil_tech": n, "year": _num(tech.get_str("year"))}

    for key, fname in (("eu4_branch", "eu4_branch.txt"), ("eu4_rev", "eu4_rev.txt")):
        raw = files.read(fname)
        if raw:
            idx.game_version[key] = raw.decode("latin-1").strip()
    return idx


# --------------------------------------------------------------------------- base_models.json

def resolved(idx: Index, name: str, field_name: str, base_only: bool = True) -> Any:
    """Field value following the clone chain: own field, else the clone source's.

    UNVERIFIED: whether a clone's attach blocks replace the source's by attach name or
    merge. This takes the clone's own attach list if it has one, else the source's.
    """
    seen = set()
    while name and name not in seen:
        seen.add(name)
        ent = idx.entity(name, base_only)
        if ent is None:
            return None
        if field_name in ent:
            return ent[field_name]
        name = ent.get("clone")
    return None


def _attach_targets(attaches: list[dict] | None) -> dict[str, str]:
    return {k: v for a in (attaches or []) for k, v in a.items() if k != "name"}


def _entity_closure(idx: Index, roots: list[str]) -> list[str]:
    out, todo = [], list(roots)
    while todo:
        name = todo.pop()
        if name in out or idx.entity(name) is None:
            continue
        out.append(name)
        ent = idx.entity(name)
        if ent.get("clone"):
            todo.append(ent["clone"])
        todo.extend(_attach_targets(ent.get("attach")).values())
    return sorted(out)


def find_pack(idx: Index, *, tag: str | None = None, gfx: str | None = None) -> dict | None:
    for defs in idx.sprite_packs.values():
        for pack in defs:
            if pack["dlc"] is not None:
                continue
            if tag and pack.get("country_trigger") == f"tag = {tag}":
                return pack
            if gfx and gfx in pack.get("graphical_cultures", []):
                return pack
    return None


def _describe_slot(idx: Index, utype: str, entity: str) -> dict:
    pdx = resolved(idx, entity, "pdxmesh")
    attach = _attach_targets(resolved(idx, entity, "attach"))
    slot: dict[str, Any] = {"entity": entity, "clones": (idx.entity(entity) or {}).get("clone")}
    if utype == "infantry":
        node, weapon = next(iter(attach.items()), (None, None))
        slot.update(pdxmesh=pdx, weapon_node=node, weapon_entity=weapon)
    elif utype == "cavalry":
        rider = attach.get("Saddle_Node")
        rider_attach = _attach_targets(resolved(idx, rider, "attach")) if rider else {}
        slot.update(horse_pdxmesh=pdx, rider_entity=rider,
                    rider_pdxmesh=resolved(idx, rider, "pdxmesh") if rider else None,
                    rider_weapon_entity=rider_attach.get("Right_hand_node"))
    else:
        cannon = attach.get("cannon_node")
        slot.update(base_pdxmesh=pdx, crew_entity=attach.get("infantry_node"), cannon_entity=cannon,
                    cannon_pdxmesh=resolved(idx, cannon, "pdxmesh") if cannon else None)
    return slot


def pack_tiers(idx: Index, pack: dict) -> dict:
    tiers: dict[str, dict] = {}
    for utype in UNIT_TYPES:
        for i, ent in sorted(pack.get(utype, {}).items()):
            tiers.setdefault(str(int(i) + 1), {})[utype] = _describe_slot(idx, utype, ent)
    return tiers


def _strip(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k not in ("name", "dlc", "seq")}


def build_base_models(files: GameFiles, idx: Index, tag: str, gfx: str) -> dict:
    tag_pack = find_pack(idx, tag=tag)
    gfx_pack = find_pack(idx, gfx=gfx)
    if tag_pack is None or gfx_pack is None:
        raise LookupError(f"no base-game sprite pack for tag {tag!r} and/or graphical culture {gfx!r}")

    root_entities = [e for p in (tag_pack, gfx_pack) for u in UNIT_TYPES for e in p.get(u, {}).values()]
    entity_names = _entity_closure(idx, root_entities)
    entities = {n: _strip(idx.entity(n)) for n in entity_names}

    pdx_names = sorted({e["pdxmesh"] for e in entities.values() if "pdxmesh" in e})
    pdxmeshes, textures, anim_names, mesh_bones, mesh_skel = {}, {}, set(), {}, {}
    for name in pdx_names:
        rec = idx.pdxmesh(name)
        if rec is None:
            pdxmeshes[name] = {"missing": True}
            continue
        entry = _strip(rec)
        raw = files.read(rec["file"])
        if raw is None:
            entry["mesh_missing"] = True
        else:
            summary = summarize_mesh(parse_pdx_mesh(raw))
            shape = summary["shapes"][0] if summary["shapes"] else {}
            folder = str(PurePosixPath(rec["file"]).parent)
            entry.update(
                shader=shape.get("shader"),
                uses_country_colour=shape.get("shader") == COLOUR_SHADER,
                shape=shape.get("shape"),
                textures={k: shape.get(k) for k in ("diffuse", "normal", "specular")},
                texture_dir=folder,
                mesh_locators=summary["locators"],
            )
            if summary["aabb"]:
                entry["aabb"] = {k: [round(v, 4) for v in vals] for k, vals in summary["aabb"].items()}
            if len(summary["shapes"]) > 1:
                entry["extra_shapes"] = summary["shapes"][1:]
            mesh_bones[name] = summary["bones"]
            mesh_skel[name] = summary["skeleton"]
            for tex in entry["textures"].values():
                if tex and tex not in textures:
                    data = files.read(f"{folder}/{tex}")
                    textures[tex] = read_dds_header(data) if data else {"missing": True}
        anim_names.update(entry.get("animations", {}).values())
        pdxmeshes[name] = entry

    # Group identical skeletons; name each group after the first mesh file using it,
    # graphical-culture default meshes first so names read "western_generic_01_type1", not "DAN_01".
    gfx_tiers = pack_tiers(idx, gfx_pack)
    gfx_pdx = {v for slots in gfx_tiers.values() for s in slots.values() for k, v in s.items() if k.endswith("pdxmesh")}
    skeletons: dict[str, dict] = {}
    attach_keys = {k for e in entities.values() for a in e.get("attach", []) for k in a if k != "name"}
    for name in sorted(mesh_bones, key=lambda n: (n not in gfx_pdx, n)):
        bones = mesh_bones[name]
        if not bones:
            pdxmeshes[name]["skeleton"] = None
            continue
        sid = next((s for s, v in skeletons.items() if v["bones"] == bones), None)
        if sid is None:
            sid = PurePosixPath(pdxmeshes[name]["file"]).stem
            skel = mesh_skel.get(name) or {}
            skeletons[sid] = {
                "bones": bones, "attach_bones": [b for b in bones if b in attach_keys], "used_by": [],
                # Hierarchy and bind pose (mesh space, Y-up, row-major 3x4) as in this group's first mesh.
                "parents": {b: (skel.get(b) or {}).get("parent") for b in bones},
                "bind": {b: [round(v, 4) for v in (skel.get(b) or {}).get("bind")] if (skel.get(b) or {}).get("bind")
                         else None for b in bones},
                "bind_from": name,
            }
        skeletons[sid]["used_by"].append(name)
        pdxmeshes[name]["skeleton"] = sid
    for skel in skeletons.values():
        skel["used_by"].sort()

    animation_types = {}
    for a in sorted(anim_names):
        rec = idx.animation_type(a)
        animation_types[a] = {"file": rec["file"], "file_exists": files.exists(rec["file"]),
                              "defined_in": rec["defined_in"]} if rec else {"missing": True}

    tag_order = idx.pack_order.get(tag_pack["name"])
    dlc_overrides = []
    for pname, defs in sorted(idx.sprite_packs.items()):
        for pack in defs:
            if pack["dlc"] is None:
                continue
            listed = {e for u in UNIT_TYPES for e in pack.get(u, {}).values()}
            if not listed & set(root_entities):
                continue
            order = idx.pack_order.get(pname)
            redefined = sorted(
                (n, d.get("version")) for n in entity_names for d in idx.entities.get(n, []) if d["dlc"] == pack["dlc"])
            dlc_overrides.append({
                "dlc": pack["dlc"], "pack": pname, "pack_order": order[-1]["order"] if order else None,
                "country_trigger": pack["country_trigger"],
                "partial": {u: sorted(int(i) for i in pack.get(u, {})) for u in UNIT_TYPES},
                "redefined_entities": [{"name": n, "version": v} for n, v in redefined],
            })

    recolour_bases = {}
    for tier, slots in gfx_tiers.items():
        inf, cav, art = slots.get("infantry", {}), slots.get("cavalry", {}), slots.get("artillery", {})
        base = {
            "infantry": inf.get("pdxmesh"),
            "cavalry_horse": cav.get("horse_pdxmesh"),
            "cavalry_rider": cav.get("rider_pdxmesh"),
            "artillery_cannon": art.get("cannon_pdxmesh"),
        }
        recolour_bases[tier] = {
            role: {"pdxmesh": p, "uses_country_colour": (pdxmeshes.get(p) or {}).get("uses_country_colour")}
            for role, p in base.items()
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "eu4unitgen.library.scanner",
        "game_version": idx.game_version,
        "legal_note": "Metadata only (names, paths, bone names, texture header facts). No game files or script text.",
        "sprite_levels": idx.sprite_levels,
        "graphical_cultures": idx.graphical_cultures,
        "reference_nation": {
            "tag": tag,
            "graphical_culture": gfx,
            "sprite_pack": {"name": tag_pack["name"], "file": tag_pack["defined_in"],
                            "country_trigger": tag_pack["country_trigger"],
                            "order": tag_order[-1]["order"] if tag_order else None},
            "tiers": pack_tiers(idx, tag_pack),
            "dlc_overrides": dlc_overrides,
        },
        "graphical_culture_defaults": {
            "graphical_culture": gfx,
            "sprite_pack": {"name": gfx_pack["name"], "file": gfx_pack["defined_in"],
                            "country_trigger": gfx_pack["country_trigger"],
                            "graphical_cultures": gfx_pack.get("graphical_cultures", [])},
            "tiers": gfx_tiers,
        },
        "recolour_bases": {
            "note": f"Default base meshes for recolour/kitbash: the {gfx} generic set. "
                    f"Only meshes with uses_country_colour=true pick up country colours.",
            "tiers": recolour_bases,
        },
        "skeletons": skeletons,
        "pdxmeshes": pdxmeshes,
        "animation_types": animation_types,
        "entities": entities,
        "textures": dict(sorted(textures.items())),
    }


# --------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eu4", default=os.environ.get("EU4_PATH"), help="EU4 install folder (or set EU4_PATH)")
    ap.add_argument("--tag", default="DAN")
    ap.add_argument("--gfx", default="westerngfx")
    ap.add_argument("--no-dlc", action="store_true", help="ignore dlc/*/*.zip")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("base_models.json"))
    args = ap.parse_args(argv)
    if not args.eu4:
        ap.error("pass --eu4 or set EU4_PATH")
    files = GameFiles(Path(args.eu4), include_dlc=not args.no_dlc)
    try:
        doc = build_base_models(files, build_index(files), args.tag, args.gfx)
    finally:
        files.close()
    args.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(doc['entities'])} entities, {len(doc['pdxmeshes'])} pdxmeshes, "
          f"{len(doc['textures'])} textures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
