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
# slots that may stay "the game's own" while the others are yours: artillery with the game's gun and your crew
OPTIONAL = {"artillery": ("gun",)}


def needed(utype: str, on_foot: bool = False) -> tuple[str, ...]:
    if on_foot and utype == "cavalry":  # cavalry on foot (Track T): the soldier is the cavalry, no horse
        return ("soldier",)
    return tuple(k for k in UNIT_TYPES[utype] if k not in OPTIONAL.get(utype, ()))


# the base game's hand weapons a soldier on foot can carry (referenced by name, never copied; none from DLC; read
# from 1.37): (entity, hand node, animation set, what the app calls it)
WEAPONS = (
    ("", None, "base", "none (its own animations)"),
    ("asian_spear_01", "Right_hand_node", "polearm_twohanded", "yari (Japanese spear)"),
    ("asian_rifle_02", "Left_hand_node", "rifleman", "matchlock (Japanese)"),
    ("asian_rifle_03", "Left_hand_node", "rifleman", "later matchlock (Japanese)"),
    ("muslim_spear", "Right_hand_node", "polearm_twohanded", "spear"),
    ("western_halbard", "Right_hand_node", "polearm_twohanded", "halberd"),
    ("cavalry_saber", "Right_hand_node", "polearm_onehanded", "sabre"),
    ("western_musket", "Left_hand_node", "musketeer", "musket"),
    ("western_musket_2", "Left_hand_node", "musketeer", "later musket"),
)


def weapon(entity: str | None) -> tuple:
    """A ``WEAPONS`` row by its entity (none: the first)."""
    return next((w for w in WEAPONS if w[0] == (entity or "")), WEAPONS[0])
# what each slot is called in a unit of that type
SLOT_TEXT = {("infantry", "soldier"): "Soldier", ("cavalry", "horse"): "Horse", ("cavalry", "soldier"): "Rider",
             ("artillery", "gun"): "Gun", ("artillery", "soldier"): "Crew", ("cavalry", "weapon"): "Weapon"}
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
        return display(self.name)

    def tag_text(self) -> str:
        bits = []
        if self.tags.get("style"):
            bits.append(self.tags["style"])
        bits += list(self.tags.get("regions") or [])
        return ", ".join(bits)


def display(name: str) -> str:
    """A folder name as people read it: "bay_horse" -> "Bay Horse" (folders keep their own names)."""
    return " ".join(w[:1].upper() + w[1:] for w in (name or "").replace("-", "_").split("_") if w)


def rights_text(rights: str, starter: bool = False) -> str:
    """A rights record in a few plain words for the main view ("Made by you with ChatGPT · bundling allowed"); the
    full record stays behind Details."""
    r = (rights or "").lower()
    if not r:
        return "No rights given"
    who = "the app's author" if starter else "you" if any(w in r for w in ("the user owns", "i own", "user's original",
                                                                          "by the user", "mine", "drawn by me")) else ""
    how = "with ChatGPT" if "chatgpt" in r else "from a photo" if "photo" in r else ""
    bits = [" ".join(x for x in ("Made", f"by {who}" if who else "", how) if x)] if (who or how) else [rights.split(";")[0]]
    if "cc by 4.0" in r or "cc-by 4.0" in r:
        bits.append("CC BY 4.0")
    if "bundl" in r and "never bundled" not in r:
        bits.append("bundling allowed")
    return " · ".join(bits)


VERDICT_TEXT = {"good": ("ok", "Good: ready to use"), "ok": ("warn", "Usable: a few things to look at in game"),
                "check": ("error", "Look at it in game first")}


def summary(report: dict | None) -> tuple[str, str]:
    """A quality report as one dot and one line: ("ok"|"warn"|"error"|"none", words)."""
    if not report:
        return "none", "Not checked yet"
    kind, words = VERDICT_TEXT.get(report.get("verdict"), ("none", report.get("summary") or "Checked"))
    return kind, words


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


def part_render(r: dict) -> str | None:
    """The picture of a part run alone (never with the game's rider or anything else of the game's): a horse's game
    version from three-quarter front, a soldier on foot from its preview, else the game version from the front."""
    f = Path(r["folder"])
    if r.get("kind") == "soldier":
        g = f / "preview" / "gallery" / f"close_three_quarter_{r.get('tier', 1)}.png"
        if g.is_file():
            return str(g)
    versions = sorted((f / "trellis" / "game").glob("v[0-9][0-9][0-9]/manifest.json"))
    m = _json(f / "finished" / "manifest.json") or (_json(versions[-1]) if versions else {})
    imgs = ((m.get("compare") or {}).get("images") or {}).get("game") or {}
    for view in ("front34", "side", "front"):
        if imgs.get(view) and Path(imgs[view]).is_file():
            return imgs[view]
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
                        r.get("image"), part_render(r), r.get("rights", ""), bool(st.get("done")), r.get("state", "new")))
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
def new_unit(name: str, utype: str, tiers: dict, country_tag: str = "", gfx: str = "",
             country_colours: bool = True, on_foot: bool = False) -> Path:
    """A unit (kind ``unit``): ``tiers`` {1: {"horse": ref, "soldier": ref, "gun": ref}}; empty tiers are the game's
    own. Refused if the name is taken, the type is unknown, or no tier has its parts."""
    from eu4unitgen.ref3d import flow
    if utype not in UNIT_TYPES:
        raise ValueError(f"the type is one of {tuple(UNIT_TYPES)}")
    clean = {}
    foot = bool(on_foot) and utype == "cavalry"
    allowed = ("soldier", "weapon") if foot else UNIT_TYPES[utype]
    for t, slots in tiers.items():
        s = {k: v for k, v in (slots or {}).items() if k in allowed and v}
        if s:
            missing = [k for k in needed(utype, foot) if k not in s]
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
                               for s in clean.values() for k, ref in s.items() if k != "weapon"}))
    for s in clean.values():
        for k, ref in list(s.items()):
            if k != "weapon" and ref.startswith("starter:"):
                fit(ref.split(":", 1)[1])
                s[k] = "run:" + ref.split(":", 1)[1]
    first = part(next(v for k, v in clean[min(clean, key=int)].items() if k != "weapon"))
    req = {"schema": 1, "name": name, "kind": "unit", "type": utype, "tiers": clean, "country": country_tag.upper(),
           "gfx": gfx, "rights": rights, "image": first.image if first else None, "tier": int(min(clean, key=int)),
           "country_colours": bool(country_colours), **({"on_foot": True} if foot else {}),
           "created": time.time()}
    (f / "request.json").write_text(json.dumps(req, indent=1), encoding="utf-8")
    return f


def unit_preview(utype: str, slots: dict[str, str]) -> tuple[str | None, bool, str | None]:
    """A picture of the whole unit at a tier (horse with rider, gun with crew): (path, True, None) when one has been
    drawn for these parts (an assembled unit's preview, or an older cavalry run of the same horse and soldier), else
    the picture of its main part that is picked, False, and that part's slot. (None, False, None) for the game's
    own."""
    from eu4unitgen.ref3d import flow
    if not slots:
        return None, False, None
    for r in flow.runs():
        if not (r.get("status") or {}).get("done"):
            continue
        tiers = {}
        if r.get("kind") == "unit" and r.get("type") == utype:
            tiers = {int(t): s for t, s in (r.get("tiers") or {}).items()}
        elif r.get("kind") == "cavalry" and utype == "cavalry":
            tiers = {int(r.get("tier", 1)): {"horse": f"run:{r.get('horse')}", "soldier": f"run:{r.get('soldier')}"}}
        for t, s in tiers.items():
            if s == slots:
                g = Path(r["folder"]) / "preview" / "gallery" / f"close_three_quarter_{t}.png"
                if g.is_file():
                    return str(g), True, None
    main = {"infantry": "soldier", "cavalry": "horse", "artillery": "gun"}[utype]
    shown = main if slots.get(main) else next(iter(slots))
    p = part(slots[shown])
    return ((p.thumb or p.image) if p else None), False, shown


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
STARTER_RIGHTS = ("Starter part that comes with the app: made by the app's author with ChatGPT from their own picture, "
                  "TRELLIS.2 from it; CC BY 4.0 (credit: the EU4 Unit Generator's author)")
# the starter library (the user, 2026-10-28): run name -> tags. Packed from the author's finished runs when the
# installer is built (``pack_starters``; the folder isn't in git), never from anything of the game's
STARTER = {"bay_horse": {"style": "light", "regions": ["Western", "European"]},
           "english_horse": {"style": "light", "regions": ["English", "Western", "generic"]},
           "cataphract_horse": {"style": "barded", "regions": ["Byzantine", "Greek"]},
           # "english_soldier": out until he's fixed (the user, 2026-10-28; rerun from the arms-out picture)
           "cataphract": {"regions": ["Byzantine", "Greek"]},
           "bombard": {"regions": ["European", "generic"]},
           # the app's example of a stylised (exaggerated) imported model (the user, 2026-10-29: Mambo, a chibi; kept as
           # an example of an overexaggerated model; the private mod's Tracen Academy cavalry)
           "mambo_import_test": {"regions": ["stylised example", "Tracen Academy"], "example": "stylised"}}


def pack_starters(dest: Path | None = None) -> dict[str, str]:
    """Every ``STARTER`` part packed into ``dest`` (default: the app's starter folder). -> {name: "packed" or why
    not}."""
    from eu4unitgen.ref3d import flow
    dest = Path(dest or starter_root())
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, tags in STARTER.items():
        f = flow.folder_of(name)
        if not (flow.status(f) or {}).get("done"):
            out[name] = "not finished: not packed"
            continue
        pack(f, dest / name, tags=tags)
        out[name] = "packed"
    return out


def pack(folder: Path, out: Path, tags: dict | None = None, rights: str | None = None) -> Path:
    """A finished part run packed for the starter library (the user, 2026-10-28: the raw TRELLIS.2 model, its tags and
    rights; fitted and weighted on each PC on first use): ``out/part.json`` with relative paths, ``raw.glb``, the
    cut-out picture (``input/``), the part crops cut from the picture (``crops_cache/``: no SAM 2 needed to fit it)
    and TRELLIS.2's models of them (``parts_cache/``), the picture, and a render of the raw model. Nothing fitted to
    the game: no geometry on the game's skeleton, no weights, never a .mesh or .dds."""
    from eu4unitgen.ref3d import flow
    folder, out = Path(folder), Path(out)
    req = flow.request(folder)
    tr = folder / "trellis"
    fin = folder / "finished"
    m = _json(fin / "manifest.json")
    if not m or not (tr / "raw.glb").is_file():
        raise FileNotFoundError(f"{req['name']} isn't finished")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copyfile(tr / "raw.glb", out / "raw.glb")
    shutil.copytree(tr / "input", out / "input")
    crops = Path(m.get("edited_from") or fin) / "parts" / "crops"
    if not (crops / "crops.json").is_file():
        v = sorted((tr / "game").glob("v[0-9][0-9][0-9]"))
        crops = (v[-1] / "parts" / "crops") if v else crops
    if (crops / "crops.json").is_file():
        shutil.copytree(crops, out / "crops_cache", ignore=shutil.ignore_patterns("*_hint.png"))
    if (tr / "parts_cache").is_dir():
        shutil.copytree(tr / "parts_cache", out / "parts_cache", ignore=shutil.ignore_patterns("*.log", "logs"))
    if (tr / "import.json").is_file():  # an imported model: its rig's joints and regions (the fit's 1:1 path)
        shutil.copyfile(tr / "import.json", out / "import.json")
    for bad in out.rglob("*"):
        if bad.suffix.lower() in (".mesh", ".dds", ".anim", ".asset", ".gfx"):
            raise ValueError(f"{bad.name}: a game format isn't packed")
    image = None
    if req.get("image") and Path(req["image"]).is_file():
        image = "picture" + Path(req["image"]).suffix.lower()
        shutil.copyfile(req["image"], out / image)
    raw = ((m.get("compare") or {}).get("images") or {}).get("raw") or {}
    view = next((v for v in ("front34", "side", "front") if raw.get(v) and Path(raw[v]).is_file()), None)
    thumb = None
    if view:
        thumb = "render.png"
        shutil.copyfile(raw[view], out / thumb)
    imported = req.get("source")
    rights = rights or (req.get("rights") if imported else STARTER_RIGHTS)
    d = {"schema": 2, "name": req["name"], "kind": req["kind"],
         "tags": tags if tags is not None else (req.get("tags") or {}), "rights": rights, "image": image,
         **({"source": imported} if imported else {}),
         "thumb": thumb or image, "options": req.get("options") or {},
         "edits": {k: v for k, v in (req.get("edits") or {}).items() if k != "candidate"},
         "made_by": ("an imported 3D model" if imported else "TRELLIS.2 from the picture (the raw model)")
                    + "; fitted to the game on the PC that uses it"}
    (out / "part.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
    return out


def fit(name: str, starter: Path | None = None) -> Path:
    """A starter part fitted on this PC: a run of its own in the projects folder (its raw model in place of the AI
    steps), started if it isn't finished. A run of that name already there (your own) is used as it is."""
    from eu4unitgen.ref3d import flow
    f = flow.folder_of(name)
    if (f / "request.json").is_file():
        if not (flow.status(f) or {}).get("done") and (flow.status(f) or {}).get("state") != "running":
            flow.start(f)
        return f
    src = Path(starter or starter_root()) / name
    d = _json(src / "part.json")
    if d.get("kind") not in KINDS:
        raise FileNotFoundError(f"no starter part called {name}")
    image = src / (d.get("image") or "picture.png")
    f = flow.new_request(name, image, d.get("rights", STARTER_RIGHTS), d["kind"], 1, options=d.get("options") or {})
    tr = f / "trellis"
    shutil.copyfile(src / "raw.glb", tr / "raw.glb")
    shutil.copytree(src / "input", tr / "input", dirs_exist_ok=True)
    for sub in ("crops_cache", "parts_cache"):
        if (src / sub).is_dir():
            shutil.copytree(src / sub, tr / sub, dirs_exist_ok=True)
    if (src / "import.json").is_file():
        shutil.copyfile(src / "import.json", tr / "import.json")
    (tr / "candidates.json").write_text(json.dumps({"candidates": [{"seed": 0, "glb": str((tr / "raw.glb").resolve()),
                                                                     "note": "the starter part's model"}], "best": 0}),
                                        encoding="utf-8")
    req = flow.request(f)
    req.update(tags=d.get("tags") or {}, edits=d.get("edits") or {}, starter=True,
               **({"source": d["source"]} if d.get("source") else {}))
    (f / "request.json").write_text(json.dumps(req, indent=1), encoding="utf-8")
    now = time.time()
    flow._write(f, {"name": name, "kind": d["kind"], "state": "stopped", "done": False, "started": now,
                    "stages": {s: {"state": "done", "seconds": 0, "from": "the starter part"}
                               for s in ("import",) if d.get("source")} | {s: {"state": "done", "seconds": 0, "from":
                               "the starter part"} for s in ("cutout", "trellis") if not d.get("source")}})
    flow.start(f)
    return f


def fitted(name: str) -> tuple[Path, dict]:
    """A starter part's fitted, finished folder and subject (``fit`` first; ``ref3d.assemble`` waits for it)."""
    from eu4unitgen.ref3d import flow
    f = flow.folder_of(name)
    if not (f / "finished" / "manifest.json").is_file():
        fit(name)
        raise FileNotFoundError(f"the starter part {display(name)} is being fitted to the game on this PC first")
    return f / "finished", json.loads((f / "subject.json").read_text(encoding="utf-8"))
