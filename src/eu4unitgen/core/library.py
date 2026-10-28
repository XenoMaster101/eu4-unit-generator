"""The part library (SPEC Track L): parts made once from a picture and reused; units put together from them.

  parts   horses, soldiers, guns: each a finished run of ``ref3d.flow`` (kind ``horse``/``soldier``/``gun``), or a
          starter part that comes with the app (``library/starter/<name>/part.json``: our geometry and textures only,
          never a game file). Each carries tags: a horse's style (``STYLES``) and regions (free text).
  units   an assembly: a type (infantry = a soldier; cavalry = a horse + a soldier riding; artillery = a gun + a
          soldier as crew) and per tier (1-4) the parts it uses; a tier left empty is the game's own look. Stored as
          a run of kind ``unit`` (``request.json``: type, tiers, country, gfx). Assembling needs no AI.

Suggestions: a country's graphical culture and technology group (read from the install: read only) rank the parts
(``SUGGEST``: the app's own mapping, not EU4 syntax); a part whose regions name the country's culture or culture
group comes first. Any part can still be picked.

Refs: ``"run:<name>"`` (a part in the projects folder) or ``"starter:<name>"``.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

KINDS = ("horse", "soldier", "gun")
UNIT_TYPES = {"infantry": ("soldier",), "cavalry": ("horse", "soldier"), "artillery": ("gun", "soldier")}
# what each slot is called in a unit of that type
SLOT_TEXT = {("infantry", "soldier"): "Soldier", ("cavalry", "horse"): "Horse", ("cavalry", "soldier"): "Rider",
             ("artillery", "gun"): "Gun", ("artillery", "soldier"): "Crew"}
TYPE_TEXT = {"infantry": "Infantry", "cavalry": "Cavalry", "artillery": "Artillery"}
STYLES = ("light", "barded", "steppe", "arab")
STYLE_TEXT = {"light": "light (unarmoured, saddle cloth)", "barded": "barded (armoured, caparison)",
              "steppe": "steppe (small, hardy)", "arab": "Arab (fine, high tail)"}
# graphical culture -> horse styles in order of fit (the app's own judgement)
SUGGEST = {"westerngfx": ("barded", "light"), "easterngfx": ("light", "barded"), "muslimgfx": ("arab", "light"),
           "indiangfx": ("arab", "light"), "asiangfx": ("steppe", "light"), "africangfx": ("light", "arab"),
           "southeastasiangfx": ("light", "steppe")}
NOMAD_GROUPS = ("nomad_group",)  # a technology group whose horses are steppe horses first
TIER_TEXT = {1: "Tier 1 · from 1444", 2: "Tier 2 · military tech 12", 3: "Tier 3 · military tech 21",
             4: "Tier 4 · military tech 26"}


@dataclass
class Part:
    ref: str
    name: str
    kind: str
    folder: Path
    starter: bool = False
    tags: dict = field(default_factory=dict)  # {"style": "barded", "regions": ["Byzantine", "Greek"]}
    image: str | None = None
    thumb: str | None = None
    rights: str = ""
    done: bool = False
    state: str = "new"

    @property
    def title(self) -> str:
        return self.name.replace("_", " ")

    def tag_text(self) -> str:
        bits = []
        if self.tags.get("style"):
            bits.append(self.tags["style"])
        bits += list(self.tags.get("regions") or [])
        return ", ".join(bits)


def starter_root() -> Path:
    """The starter parts that come with the app."""
    return Path(__file__).resolve().parents[1] / "library" / "starter"


def _json(p: Path) -> dict:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def regions(text: str) -> list[str]:
    """"Byzantine, Greek" -> ["Byzantine", "Greek"]."""
    return [r.strip() for r in re.split(r"[,;]", text or "") if r.strip()]


def run_thumb(r: dict) -> str | None:
    """A run's picture: its turntable's three-quarter frame, its close render, else its reference."""
    from eu4unitgen.ref3d import flow
    f = Path(r["folder"])
    frames = flow.turntable_frames(f)
    if frames:
        return str(frames[min(1, len(frames) - 1)])
    g = f / "preview" / "gallery" / f"close_three_quarter_{r.get('tier', 1)}.png"
    if g.is_file():
        return str(g)
    return r.get("image")


def starter_parts(root: Path | None = None) -> list[Part]:
    root = Path(root) if root else starter_root()
    out = []
    for p in sorted(root.glob("*/part.json")):
        d = _json(p)
        if d.get("kind") not in KINDS:
            continue
        f = p.parent
        out.append(Part(f"starter:{f.name}", d.get("name", f.name), d["kind"], f, True, d.get("tags") or {},
                        str(f / d["image"]) if d.get("image") else None, str(f / d["thumb"]) if d.get("thumb") else None,
                        d.get("rights", ""), True, "done"))
    return out


def parts(kind: str | None = None, starter: Path | None = None) -> list[Part]:
    """Every part (the user's runs first, newest first, then the starter parts), optionally of one kind."""
    from eu4unitgen.ref3d import flow
    out = []
    for r in flow.runs():
        if r.get("kind") not in KINDS:
            continue
        st = r.get("status") or {}
        out.append(Part(f"run:{r['name']}", r["name"], r["kind"], Path(r["folder"]), False, r.get("tags") or {},
                        r.get("image"), run_thumb(r), r.get("rights", ""), bool(st.get("done")), r.get("state", "new")))
    mine = {p.name for p in out}
    out += [p for p in starter_parts(starter) if p.name not in mine]  # your own run of a starter part replaces it
    return [p for p in out if kind is None or p.kind == kind]


def part(ref: str | None, starter: Path | None = None) -> Part | None:
    if not ref:
        return None
    return next((p for p in parts(starter=starter) if p.ref == ref), None)


def units() -> list[dict]:
    """Every unit: the assemblies (kind ``unit``) and the older cavalry runs, newest first."""
    from eu4unitgen.ref3d import flow
    return [r for r in flow.runs() if r.get("kind") in ("unit", "cavalry", "rider")]


def ai_ready() -> tuple[bool, str]:
    """Whether new parts can be made here (BiRefNet, SAM 2, TRELLIS.2), and what to tell the user if not."""
    from eu4unitgen.core import checks
    c = {x.key: x for x in checks.run_all(quick=True)}
    ok = c["ai_env"].ok and c.get("weights", c["ai_env"]).ok
    return ok, ("" if ok else "Making a new part from a picture needs the AI setup (Setup, about 60 GB). You can "
                              "build units from the parts already in your library without it.")


# ------------------------------------------------------------------ the country and the suggestions
@dataclass
class Country:
    tag: str
    name: str = ""
    gfx: str | None = None
    tech_group: str | None = None
    culture: str | None = None
    culture_group: str | None = None


_COUNTRIES: dict[tuple[str, str], Country | None] = {}


def country(tag: str, eu4: Path) -> Country | None:
    """A country's graphical culture, technology group and culture at 1444, read from the install (read only).
    None if the tag isn't in the game."""
    tag = (tag or "").strip().upper()
    key = (tag, str(eu4))
    if key in _COUNTRIES:
        return _COUNTRIES[key]
    from eu4unitgen.library import tags as tg
    from eu4unitgen.library.scanner import GameFiles
    from eu4unitgen.pdxscript.parser import parse_bytes
    out = None
    files = GameFiles(Path(eu4), include_dlc=True)
    try:
        cfile = tg.country_files(files).get(tag)
        if cfile:
            f = tg.TagFacts(tag, cfile)
            raw = files.read(cfile)
            if raw:
                f.graphical_culture = parse_bytes(raw).get_str("graphical_culture")
            for path, _, data in files.iter_files("history/countries", (".txt",)):
                if path.rsplit("/", 1)[-1][:3].upper() == tag:
                    tg.apply_history(f, data, tg.culture_groups(files))
                    break
            name = Path(cfile).stem
            out = Country(tag, name, f.graphical_culture, f.technology_group, f.primary_culture, f.culture_group)
    finally:
        files.close()
    _COUNTRIES[key] = out
    return out


def graphical_cultures(eu4: Path) -> list[str]:
    """The install's graphical cultures (common/graphicalculturetype.txt), read at run time."""
    p = Path(eu4) / "common" / "graphicalculturetype.txt"
    try:
        return re.findall(r"(?m)^\s*([a-z0-9_]+gfx)\b", p.read_text(encoding="latin-1"))
    except OSError:
        return []


def _norm(s: str | None) -> str:
    return re.sub(r"[^a-z]", "", (s or "").lower())


def suggest(kind: str, gfx: str | None = None, tech_group: str | None = None, culture: str | None = None,
            culture_group: str | None = None, pool: list[Part] | None = None) -> list[tuple[Part, str]]:
    """The parts of ``kind`` ranked for a country, each with why ("" when nothing speaks for it). Finished parts
    only."""
    pool = [p for p in (pool if pool is not None else parts(kind)) if p.kind == kind and p.done]
    styles = list(SUGGEST.get(gfx or "", ("light",)))
    if tech_group in NOMAD_GROUPS:
        styles = ["steppe"] + [s for s in styles if s != "steppe"]
    words = {w for w in (_norm(culture), _norm(culture_group), _norm((culture_group or "").replace("_group", ""))) if w}
    ranked = []
    for i, p in enumerate(pool):
        score, why = 0.0, ""
        regs = [_norm(r) for r in p.tags.get("regions") or []]
        hit = next((r for r in regs if r and any(r in w or w in r for w in words)), None)
        if hit:
            score += 10
            why = f"its regions match ({p.tag_text()})"
        st = p.tags.get("style")
        if kind == "horse" and st in styles:
            score += 5 - styles.index(st)
            why = why or f"a {st} horse suits {gfx or 'this country'}"
        ranked.append((-score, i, p, why))
    ranked.sort(key=lambda t: (t[0], t[1]))
    return [(p, why) for _, _, p, why in ranked]


# ------------------------------------------------------------------ units
def new_unit(name: str, utype: str, tiers: dict, country_tag: str = "", gfx: str = "") -> Path:
    """A unit (kind ``unit``): ``tiers`` {1: {"horse": ref, "soldier": ref, "gun": ref}}; empty tiers are the game's
    own. Refused if the name is taken, the type is unknown, or no tier has its parts."""
    from eu4unitgen.ref3d import flow
    if utype not in UNIT_TYPES:
        raise ValueError(f"the type is one of {tuple(UNIT_TYPES)}")
    clean = {}
    for t, slots in tiers.items():
        s = {k: v for k, v in (slots or {}).items() if k in UNIT_TYPES[utype] and v}
        if s:
            missing = [k for k in UNIT_TYPES[utype] if k not in s]
            if missing:
                raise ValueError(f"tier {t} needs a {' and a '.join(missing)} too")
            clean[str(int(t))] = s
    if not clean:
        raise ValueError("pick the parts for at least one tier")
    f = flow.folder_of(name)
    if f.exists():
        raise FileExistsError(f"a unit or part called {name} exists already")
    f.mkdir(parents=True)
    rights = "; ".join(sorted({f"{ref.split(':', 1)[1]}: {(part(ref) or Part(ref, '', '', f)).rights}"
                               for s in clean.values() for ref in s.values()}))
    first = part(next(iter(clean[min(clean, key=int)].values())))
    req = {"schema": 1, "name": name, "kind": "unit", "type": utype, "tiers": clean, "country": country_tag.upper(),
           "gfx": gfx, "rights": rights, "image": first.image if first else None, "tier": int(min(clean, key=int)),
           "created": time.time()}
    (f / "request.json").write_text(json.dumps(req, indent=1), encoding="utf-8")
    return f


def uses(ref: str) -> list[str]:
    """The units that use a part."""
    return [r["name"] for r in units() if r.get("kind") == "unit"
            and any(ref in s.values() for s in (r.get("tiers") or {}).values())]


# ------------------------------------------------------------------ tags
def set_tags(folder: Path, style: str | None, regions_text: str) -> dict:
    """A run's tags (its request.json)."""
    p = Path(folder) / "request.json"
    req = json.loads(p.read_text(encoding="utf-8"))
    tags = {"regions": regions(regions_text)}
    if req.get("kind") == "horse" and style:
        tags["style"] = style
    req["tags"] = tags
    p.write_text(json.dumps(req, indent=1), encoding="utf-8")
    return tags


# ------------------------------------------------------------------ starter parts
def pack(folder: Path, out: Path, tags: dict | None = None) -> Path:
    """A finished part run packed for the starter library: ``out/part.json`` with relative paths, its geometry and
    textures (PNG), the parts' geometry and textures, its picture and the game version's render. Never a .mesh, .dds
    or anything read from the install."""
    from eu4unitgen.ref3d import flow
    folder, out = Path(folder), Path(out)
    req = flow.request(folder)
    fin = folder / "finished"
    m = _json(fin / "manifest.json")
    if not m:
        raise FileNotFoundError(f"{req['name']} isn't finished")
    if out.exists():
        shutil.rmtree(out)
    (out / "files").mkdir(parents=True)
    taken: dict[str, str] = {}

    def keep(src) -> str:
        src = Path(src)
        if src.suffix.lower() in (".mesh", ".dds", ".anim", ".asset", ".gfx"):
            raise ValueError(f"{src.name}: a game format isn't packed")
        if str(src) in taken:
            return taken[str(src)]
        rel = f"files/{src.parent.name}_{src.name}" if (out / "files" / src.name).exists() else f"files/{src.name}"
        shutil.copyfile(src, out / rel)
        taken[str(src)] = rel
        return rel
    geometry = keep(m["geometry"])
    textures = {k: keep(v) for k, v in (m.get("textures") or {}).items() if v}
    parts_ = {}
    for key, p in (m.get("parts") or {}).items():
        q = json.loads(json.dumps(p))
        mesh = q.get("mesh") or {}
        q["mesh"] = {k: keep(v) for k, v in mesh.items() if v}
        parts_[key] = q
    image = keep(req["image"]) if req.get("image") and Path(req["image"]).is_file() else None
    view = "side" if req["kind"] == "horse" else "front"
    render = ((m.get("compare") or {}).get("images") or {}).get("game", {}).get(view)
    thumb = keep(render) if render and Path(render).is_file() else image
    q = _json(folder / "preview" / "quality.json")
    d = {"schema": 1, "name": req["name"], "kind": req["kind"], "tags": tags if tags is not None else (req.get("tags") or {}),
         "rights": req.get("rights", ""), "image": image, "thumb": thumb, "geometry": geometry, "textures": textures,
         "parts": parts_, "on_foot": m.get("on_foot"), "colour": {k: v for k, v in (m.get("colour") or {}).items()
                                                                  if k in ("default_primary",)},
         "quality": {"summary": q.get("summary"), "measures": {k: {"verdict": v.get("verdict"), "text": v.get("text")}
                                                                for k, v in (q.get("measures") or {}).items()}},
         "made_by": "eu4unitgen (TRELLIS.2 from the picture; fitted, weighted and baked by the app)"}
    (out / "part.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
    return out
