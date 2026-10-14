"""The game's view of the files after export, and the dependency graph of the generated set.

``World`` layers three sources the way the game's virtual file system does:
vanilla (+ DLC zips) < the mod's existing files < the generated files. A file in a
higher layer replaces the same path below it; everything else adds up.

``build_graph`` walks from the generated sprite packs (and every generated entity)
along pack -> entity -> clone/attach/pdxmesh -> .mesh -> textures, and
state -> animation id -> animation type -> .anim, recording whether each edge resolves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Iterator

from eu4unitgen.core.generator import SNIPPET, GeneratedSet
from eu4unitgen.core.naming import UNIT_TYPES
from eu4unitgen.library.scanner import GameFiles, Index, build_index, parse_pdx_mesh, summarize_mesh

VANILLA, MOD, GENERATED = "vanilla", "mod", "generated"
_INDEX_FIELDS = ("entities", "pdxmeshes", "animation_types", "sprite_packs", "pack_order")


def _norm(rel: str) -> str:
    return rel.replace("\\", "/").lower()


def meshsettings_texture_path(mesh_file: str, texture: str) -> str:
    """Where the engine looks for a meshsettings texture: the mesh's folder + the name, always.
    In-game test 4 (2026-10-14): a full path is prefixed too ("gfx/models/Units/gfx/models/...")
    and the unit renders black. Rule texture_paths requires bare names."""
    return f"{PurePosixPath(mesh_file).parent}/{texture}"


class MemoryFiles:
    """GameFiles-compatible reader over an in-memory {path: bytes} map (the generated set)."""

    def __init__(self, files: dict[str, bytes]):
        self._files = {_norm(p): (p, d) for p, d in files.items()}

    def read(self, rel: str) -> bytes | None:
        hit = self._files.get(_norm(rel))
        return hit[1] if hit else None

    def exists(self, rel: str) -> bool:
        return _norm(rel) in self._files

    def iter_files(self, rel_dir: str, suffixes: tuple[str, ...], recursive: bool = False) -> Iterator[tuple[str, None, bytes]]:
        prefix = _norm(rel_dir).rstrip("/") + "/"
        for low, (path, data) in sorted(self._files.items()):
            if low.startswith(prefix) and low.endswith(suffixes) and (recursive or "/" not in low[len(prefix):]):
                yield path, None, data

    def paths(self) -> list[str]:
        return [p for p, _ in self._files.values()]

    def close(self) -> None:
        pass


def _layer_paths(files: GameFiles | MemoryFiles | None) -> set[str]:
    if files is None:
        return set()
    if isinstance(files, MemoryFiles):
        return {_norm(p) for p in files.paths()}
    return {_norm(p.relative_to(files.root).as_posix()) for p in files.root.rglob("*") if p.is_file()}


class World:
    def __init__(self, game: GameFiles, game_idx: Index, mod: GameFiles | None = None,
                 generated: GeneratedSet | None = None, owned_dlc: list[str] | None = None):
        self.game, self.mod = game, mod
        self.owned_dlc = set(owned_dlc or [])
        self.generated = generated
        gen_files = {p: f.data for p, f in (generated.files.items() if generated else []) if f.role != SNIPPET}
        self.gen = MemoryFiles(gen_files)
        self.mod_paths = _layer_paths(mod)
        self.gen_paths = _layer_paths(self.gen)

        layers = [(VANILLA, game_idx, self.mod_paths | self.gen_paths)]
        if mod is not None:
            layers.append((MOD, build_index(mod), self.gen_paths))
        layers.append((GENERATED, build_index(self.gen), set()))
        self.idx = Index()
        for origin, idx, hidden in layers:
            for name in _INDEX_FIELDS:
                merged = getattr(self.idx, name)
                for key, defs in getattr(idx, name).items():
                    for d in defs:
                        if _norm(d["defined_in"]) in hidden:
                            continue  # same path in a higher layer replaces this file
                        merged.setdefault(key, []).append({**d, "origin": origin})
        self.graphical_cultures = self._list_file("common/graphicalculturetype.txt")
        self._mesh_cache: dict[str, dict | None] = {}

    # ------------------------------------------------------------------ files
    def read(self, rel: str) -> bytes | None:
        for layer in (self.gen, self.mod):
            if layer is not None:
                data = layer.read(rel)
                if data is not None:
                    return data
        return self.game.read(rel)

    def exists(self, rel: str) -> bool:
        return any(layer is not None and layer.exists(rel) for layer in (self.gen, self.mod, self.game))

    def origin_of_file(self, rel: str) -> str | None:
        if self.gen.exists(rel):
            return GENERATED
        if self.mod is not None and self.mod.exists(rel):
            return MOD
        return VANILLA if self.game.exists(rel) else None

    def _list_file(self, rel: str) -> list[str]:
        from eu4unitgen.pdxscript.parser import parse_bytes
        data = self.read(rel)
        return parse_bytes(data).bare() if data else []

    # ------------------------------------------------------------------ definitions
    def loaded(self, rec: dict) -> bool:
        """DLC files only load for owners."""
        return rec.get("dlc") is None or rec["dlc"] in self.owned_dlc

    def defs(self, kind: str, name: str) -> list[dict]:
        return getattr(self.idx, kind).get(name, [])

    def pick(self, kind: str, name: str) -> dict | None:
        """The definition the game uses: highest version wins, later-loaded on ties (UNVERIFIED)."""
        pool = [d for d in self.defs(kind, name) if self.loaded(d)]
        if not pool:
            return None
        return max(enumerate(pool), key=lambda e: ((e[1].get("version") or 0), e[0]))[1]

    def entity(self, name: str) -> dict | None:
        return self.pick("entities", name)

    @staticmethod
    def load_key(rec: dict) -> tuple[str, int]:
        """Where an entity definition sits in the engine's load order: ``gfx/entities`` files
        by path (base game and DLC interleaved, mod files in their own name's slot), then
        top to bottom in the file. Fits all 4,760 vanilla + DLC clones on 1.37.5, including
        the 7 whose parent is only in another file (``wonup.asset`` -> ``western_units.asset``,
        ``_musup_units.asset`` -> ``_hindup_units.asset``, each later file -> earlier one),
        and the in-game failure of ``hr_*.asset`` cloning ``westerngfx_*`` (SPEC §6).
        UNVERIFIED: case-sensitive vs case-insensitive name order (no vanilla file tells
        them apart); this uses lowercase."""
        return _norm(rec["defined_in"]), rec.get("seq", 0)

    def clone_parent_loaded_before(self, rec: dict) -> dict | None:
        """The parent definition the engine can see when it reaches ``rec``'s clone line,
        or None ("Couldn't find parent clone entity")."""
        key = self.load_key(rec)
        earlier = [d for d in self.defs("entities", rec["clone"]) if self.loaded(d) and self.load_key(d) < key]
        return max(earlier, key=self.load_key) if earlier else None

    def pdxmesh(self, name: str) -> dict | None:
        return self.pick("pdxmeshes", name)

    def resolve(self, entity: str, field_name: str):
        """A field's effective value: own, else inherited along the clone chain."""
        seen, name = set(), entity
        while name and name not in seen:
            seen.add(name)
            rec = self.entity(name)
            if rec is None:
                return None
            if field_name in rec:
                return rec[field_name]
            name = rec.get("clone")
        return None

    def mesh_summary(self, rel: str) -> dict | None:
        if rel not in self._mesh_cache:
            data = self.read(rel)
            try:
                self._mesh_cache[rel] = summarize_mesh(parse_pdx_mesh(data)) if data else None
            except (ValueError, IndexError):
                self._mesh_cache[rel] = {"error": True, "shapes": [], "bones": [], "locators": {}}
        return self._mesh_cache[rel]


# --------------------------------------------------------------------------- graph

@dataclass(frozen=True)
class Node:
    kind: str  # pack | entity | pdxmesh | file | anim_id | anim_type
    name: str


@dataclass
class Edge:
    src: Node
    dst: Node
    label: str
    ok: bool
    origin: str | None = None  # where dst was found: vanilla | mod | generated | <dlc name>


@dataclass
class Graph:
    roots: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    reached: set[Node] = field(default_factory=set)

    def missing(self) -> list[Edge]:
        return [e for e in self.edges if not e.ok]

    def edges_from(self, node: Node) -> list[Edge]:
        return [e for e in self.edges if e.src == node]


def _rec_origin(rec: dict | None) -> str | None:
    if rec is None:
        return None
    return rec.get("dlc") or rec.get("origin")


def build_graph(world: World, generated: GeneratedSet) -> Graph:
    g = Graph()
    gen_packs = [n for n, defs in world.idx.sprite_packs.items() if any(d["origin"] == GENERATED for d in defs)]
    gen_entities = [n for n, defs in world.idx.entities.items() if any(d["origin"] == GENERATED for d in defs)]
    g.roots = [Node("pack", n) for n in sorted(gen_packs)] + [Node("entity", n) for n in sorted(gen_entities)]
    todo = list(g.roots)

    def edge(src: Node, dst: Node, label: str, ok: bool, origin: str | None = None) -> None:
        g.edges.append(Edge(src, dst, label, ok, origin))
        if ok and dst not in g.reached:
            todo.append(dst)

    while todo:
        node = todo.pop(0)
        if node in g.reached:
            continue
        g.reached.add(node)
        if node.kind == "pack":
            pack = next(d for d in world.idx.sprite_packs[node.name] if d["origin"] == GENERATED)
            for utype in UNIT_TYPES:
                for index, ent in sorted(pack.get(utype, {}).items()):
                    rec = world.entity(ent)
                    edge(node, Node("entity", ent), f"sprite_{utype}[{index}]", rec is not None, _rec_origin(rec))
        elif node.kind == "entity":
            rec = world.entity(node.name)
            if rec is None:
                continue
            if rec.get("clone"):
                src = world.entity(rec["clone"])
                edge(node, Node("entity", rec["clone"]), "clone", src is not None, _rec_origin(src))
            for attach in world.resolve(node.name, "attach") or []:
                for bone, target in attach.items():
                    if bone != "name":
                        t = world.entity(target)
                        edge(node, Node("entity", target), f"attach:{bone}", t is not None, _rec_origin(t))
            pdx_name = world.resolve(node.name, "pdxmesh")
            if pdx_name:
                pdx = world.pdxmesh(pdx_name)
                edge(node, Node("pdxmesh", pdx_name), "pdxmesh", pdx is not None, _rec_origin(pdx))
                for state in world.resolve(node.name, "states") or []:
                    anim_id = state.get("animation")
                    if anim_id and pdx is not None:
                        ok = anim_id in (pdx.get("animations") or {})
                        edge(node, Node("anim_id", f"{pdx_name}:{anim_id}"), f"state:{state.get('name')}", ok)
        elif node.kind == "anim_id":
            pdx_name, anim_id = node.name.split(":", 1)
            atype = (world.pdxmesh(pdx_name) or {}).get("animations", {}).get(anim_id)
            rec = world.pick("animation_types", atype) if atype else None
            edge(node, Node("anim_type", atype or "?"), "type", rec is not None, _rec_origin(rec))
        elif node.kind == "anim_type":
            rec = world.pick("animation_types", node.name)
            edge(node, Node("file", rec["file"]), "file", world.exists(rec["file"]), world.origin_of_file(rec["file"]))
        elif node.kind == "pdxmesh":
            pdx = world.pdxmesh(node.name)
            if pdx and pdx.get("file"):
                edge(node, Node("file", pdx["file"]), "file", world.exists(pdx["file"]), world.origin_of_file(pdx["file"]))
                for ms in pdx.get("meshsettings") or []:
                    for key in ("texture_diffuse", "texture_normal", "texture_specular"):
                        if ms.get(key):
                            rel = meshsettings_texture_path(pdx["file"], ms[key])
                            edge(node, Node("file", rel), f"meshsettings:{key}", world.exists(rel),
                                 world.origin_of_file(rel))
        elif node.kind == "file" and node.name.lower().endswith(".mesh"):
            summary = world.mesh_summary(node.name)
            folder = PurePosixPath(node.name).parent
            for shape in (summary or {}).get("shapes", []):
                for key in ("diffuse", "normal", "specular"):
                    tex = shape.get(key)
                    if tex:
                        rel = str(folder / tex)
                        edge(node, Node("file", rel), key, world.exists(rel), world.origin_of_file(rel))
    return g
