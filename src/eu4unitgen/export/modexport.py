"""Add to mod (Track P2.6-7): a finished unit written into the user's EU4 mod folder.

Three kinds of mod:
  normal            a mod of its own, ``<subject>_units`` (launcher name "<Subject> units"): the unit set for a tag
                    (its graphical culture, sprite pack, unit files with stats, the MIL tech lines, country colours,
                    localisation with the subject's name in the unit names), built by the test-mod writer without its
                    test-only tech remap, validated, with a manifest (rule 5: nothing the tool didn't write is ever
                    overwritten). Changes the game's checksum: no achievements.
  achievement_safe  graphics only: the chosen graphical culture's cavalry at one tier re-dressed. The culture's own
                    entities (``<gfx>_CAVALRY_<tier>`` and its rider) are written again with a higher ``version``
                    (how DLC116 replaces base-game entities, SPEC §3), with our meshes and textures; nothing in
                    ``common/`` or any other folder the checksum covers. UNVERIFIED in game: that a mod's entity with
                    a higher version replaces a base-game graphical culture's (DLC116 is the precedent for tag
                    entities).

The checksum (achievements, ironman): the folders and extensions listed in the install's ``checksum_manifest.txt``
(read at run time: rule 6); ``checksum_files`` lists the files of a mod that would change it.

When Windows won't let the app write into the mod folder (Controlled Folder Access), the mod is staged under the
workspace instead and ``copy_steps`` says exactly what to copy where.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath


@dataclass
class AddResult:
    mode: str
    mod_name: str
    written_to: Path | None  # the mod folder it was written into, or None when staged
    staged_at: Path | None = None
    files: list[str] = field(default_factory=list)
    checksum_files: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    readme: str = ""  # the export's breakdown (``breakdown``), also written as <mod>/README.txt


def checksum_rules(eu4: Path) -> list[tuple[str, bool, str]]:
    """(folder, with its subfolders, extension) from the install's checksum_manifest.txt."""
    text = (Path(eu4) / "checksum_manifest.txt").read_text(encoding="utf-8", errors="ignore")
    out = []
    for block in re.findall(r"directory\s*=\s*\{(.*?)\}", text, re.S):
        name = re.search(r'name\s*=\s*"([^"]+)"', block)
        sub = re.search(r"sub_directories\s*=\s*(\w+)", block)
        ext = re.search(r'file_extension\s*=\s*"([^"]+)"', block)
        if name and ext:
            out.append((name.group(1).strip("/").lower(), bool(sub and sub.group(1) == "yes"), ext.group(1).lower()))
    return out


def checksum_files(paths: list[str], rules: list[tuple[str, bool, str]]) -> list[str]:
    """The mod-relative ``paths`` the game's checksum covers."""
    out = []
    for p in paths:
        pp = PurePosixPath(p.replace("\\", "/").lower())
        parent = str(pp.parent)
        for folder, sub, ext in rules:
            inside = parent == folder or (sub and parent.startswith(folder + "/"))
            if inside and pp.suffix == ext:
                out.append(p)
                break
    return out


DROPPED_RULES = {"sprite_packs", "pack_order", "units", "unit_files", "technologies", "tech", "countries", "country",
                 "country_colours", "colours", "localisation", "graphical_culture", "graphicalculturetype",
                 "version_collisions", "unit_names"}


def _rule_of(error: str) -> str:
    """The validator rule an error line names ("ERROR   sprite_packs: ...")."""
    m = re.match(r"\s*\w+\s+([\w.]+):", error)
    return m.group(1).lower() if m else ""


def culture_pack(files: list, name: str, gfx: str, tag: str, tiers: list[int], set_entities: str, gfx_entities: str,
                 set_pack: str, foot: set | None = None) -> tuple[list, list[str]]:
    """Track T: a graphics pack for a new graphical culture (the user's "Tracen Academy"), no tag: of a set built for a
    working tag keep the graphics (``gfx/``, ``interface/``), the culture's list (``common/graphicalculturetype.txt``),
    the culture's own entities and sprite pack, the descriptors and manifest; the tag's dressed units (infantry,
    cavalry and its riders, artillery and its crews, cannons, their parts) replace the culture's, every reference in
    them renamed to it. Dropped: the tag's entities, sprite pack and order, unit stats, technology, countries,
    colours, localisation. Any nation with ``graphical_culture = <gfx>`` shows it. -> (the files kept, notes)."""
    keep, notes = [], []
    tag_file = next((f for f in files if f.path.lower().endswith(set_entities.lower())), None)
    tag_blocks = dict(_entity_blocks(tag_file.data.decode("utf-8-sig"))) if tag_file else {}
    for f in files:
        low = f.path.replace("\\", "/").lower()
        inner = low[len(name) + 1:] if low.startswith(name + "/") else low
        if low in (f"{name}.mod", f"{name}/descriptor.mod") or inner.startswith(".eu4unitgen"):
            keep.append(f)
        elif inner == set_entities.lower() or inner == set_pack.lower() or inner.startswith("gfx/sprite_packs_order"):
            continue
        elif inner == gfx_entities.lower():
            # the tag's dressed set under the culture's names (every reference in it renamed: a horse's rider, an
            # artillery's crew, a cannon); the culture's own blocks the tag has no version of are kept
            ours = {}
            for bname, block in tag_blocks.items():
                if bname.startswith(f"{tag}_"):
                    gname = f"{gfx}_" + bname[len(tag) + 1:]
                    ours[gname] = block.replace(f'"{tag}_', f'"{gfx}_')
                    notes.append(f"{gname}: the unit as built (from {bname})")
            out, seen = [], set()
            for bname, block in _entity_blocks(f.data.decode("utf-8-sig")):
                out.append(ours.get(bname, block))
                seen.add(bname)
            out += [b for n_, b in ours.items() if n_ not in seen]
            out += [b for n_, b in tag_blocks.items() if not n_.startswith(f"{tag}_")]  # its parts (no tag in names)
            f.data = ("# Generated by eu4unitgen (culture pack). Regenerate rather than editing by hand.\n\n"
                      + "\n\n".join(out) + "\n").encode("utf-8")
            keep.append(f)
        elif inner.startswith("gfx/") or inner.startswith("interface/") or inner == "common/graphicalculturetype.txt":
            keep.append(f)
    return keep, notes


# what each part of an exported mod is for (the user, 2026-10-29: "a list of the export folder and a breakdown of it,
# so we know what to take out and how to import it"); the first pattern that matches a path wins
PARTS = [
    ("{name}.mod", "the launcher's entry for the mod (its name and where the folder is); goes beside the folder"),
    ("{name}/descriptor.mod", "the same entry inside the folder (the launcher needs both)"),
    ("{name}/README.txt", "this breakdown (the game ignores it)"),
    ("{name}/eu4unitgen_credits.txt", "the credits of the 3D models the units were made from"),
    ("{name}/.eu4unitgen/", "the app's record of what it wrote, so it never overwrites your own edits (the game "
                            "ignores it)"),
    ("{name}/common/graphicalculturetype.txt", "the game's list of graphical cultures with the new one added"),
    ("{name}/common/units/", "the unit types and their stats (one file per unit per tier)"),
    ("{name}/common/technologies/", "the military technology lines that unlock the units"),
    ("{name}/common/countries/", "country files (their graphical culture)"),
    ("{name}/common/country_colors/", "the countries' unit colours"),
    ("{name}/common/", "other game rules"),
    ("{name}/gfx/entities/", "the unit definitions: which model each unit and tier wears, its animations and what "
                             "it holds"),
    ("{name}/gfx/sprite_packs_order/", "the order the sprite packs are tried in"),
    ("{name}/gfx/sprite_packs/", "which countries or graphical cultures use these units"),
    ("{name}/gfx/models/", "the models (.mesh) and their textures (.dds: colour, normal, gloss)"),
    ("{name}/gfx/", "other graphics"),
    ("{name}/interface/", "the model definitions: which .mesh each model is and the animations it plays"),
    ("{name}/localisation/", "the units' names in game"),
]


def breakdown(files: list[str], name: str, mode: str, mod_dir: Path, tag: str = "", gfx: str = "",
              display: str = "") -> str:
    """The export's folder, file by file group, with what each is for and how to use it (written into the mod as
    README.txt and shown in the app)."""
    groups: dict[str, list[str]] = {}
    for p in sorted(files):
        low = p.replace("\\", "/")
        for pat, what in PARTS:
            pat = pat.format(name=name)
            if low == pat or (pat.endswith("/") and low.startswith(pat)):
                groups.setdefault(pat, []).append(low)
                break
        else:
            groups.setdefault(low, []).append(low)
    lines = [f"{display or name}: what is in this mod and how to use it", "",
             f"Folder: {name}  (and {name}.mod beside it)", ""]
    for pat, what in PARTS:
        pat = pat.format(name=name)
        if pat in groups:
            got = groups.pop(pat)
            lines.append(f"{pat}  ({len(got)} file{'s' if len(got) != 1 else ''})")
            lines.append(f"    {what}")
            lines += [f"      {g[len(pat):] if pat.endswith('/') else ''}" for g in got if pat.endswith("/")][:40]
    for rest, got in groups.items():
        lines.append(f"{rest}")
    lines += ["", "How to use it"]
    lines.append(f"  1. Copy the folder {name} and the file {name}.mod into your EU4 mod folder ({mod_dir}). If an "
                 f"older {name} is there, delete it first (copying a folder over another merges them).")
    lines.append(f"  2. In the EU4 launcher, enable \"{display or name}\".")
    if mode == "culture_pack":
        lines += [f"  3. Give your nation the culture: in its country file (common/countries/<Nation>.txt in your "
                  f"nation mod) write  graphical_culture = {gfx}  . Every country with that line shows these units; "
                  "no other country changes.",
                  "  4. If your nation mod also changes common/graphicalculturetype.txt, load it after this mod and add "
                  f"{gfx} to its list (the last mod to load wins that file)."]
    elif mode == "achievement_safe":
        lines.append(f"  3. Every country with the graphical culture {gfx} shows the new units. Nothing the checksum "
                     "covers changes: achievements stay.")
    else:
        lines.append(f"  3. {tag} gets the new units (their stats, tech and colours). This changes the game's "
                     "checksum: no achievements while it's on.")
    return "\n".join(lines) + "\n"


def mod_name(subject: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", subject.lower()).strip("_") + "_units"


def copy_steps(staged: Path, mod_dir: Path, name: str) -> list[str]:
    return [f"Windows didn't let the app write into {mod_dir} (Controlled Folder Access).",
            f"Open {staged} in Explorer.",
            f"In {mod_dir}, delete the old folder {name} and the file {name}.mod if they are there (copying a folder "
            "over another merges them, and files the new version no longer has would stay and still load).",
            f"Copy the folder {name} and the file {name}.mod into {mod_dir}.",
            f"In the EU4 launcher, enable \"{name}\"."]


def keep_graphics(paths: list[str], name: str, gfx: str, tier: int, rules) -> tuple[set[str], dict[str, bytes], list[str]]:
    """Achievement-safe: which of a mod's planned files stay (paths under ``<name>/``, plus the launcher file):
    graphics and interface only, never a checksummed file nor a sprite pack (a pack triggers on a tag; the
    culture's own entities are replaced instead). -> (the paths kept, rewritten entity files, notes)."""
    keep, rewrite, notes = set(), {}, []
    for path in paths:
        low = path.replace("\\", "/").lower()
        if low == f"{name}.mod" or low == f"{name}/descriptor.mod" or low.startswith(f"{name}/.eu4unitgen"):
            keep.add(path)
            continue
        inner = low[len(name) + 1:] if low.startswith(name + "/") else low
        if checksum_files([inner], rules) or not (inner.startswith("gfx/") or inner.startswith("interface/")):
            continue
        if inner.startswith("gfx/sprite_packs"):
            continue
        keep.add(path)
    return keep, rewrite, notes


def entities_only(data: bytes, gfx: str, tier: int | list[int], tag: str, utype: str = "cavalry") -> tuple[bytes | None, int, int]:
    """Entity definitions (the set's .asset files joined) cut down to the tag's unit of type ``utype`` (cavalry or
    infantry) at ``tier``, a cavalry's rider and every entity they attach (the parts), the unit (and rider) renamed to
    the culture's own (``<gfx>_CAVALRY_<tier>`` and its ``_rider``; ``<gfx>_INFANTRY_<tier>``, which the culture's
    guns also take as their crew), which the culture's countries without a pack of their own show. -> (the text or
    None, kept, of how many)."""
    from eu4unitgen.core.naming import Naming
    n = Naming()
    blocks = dict(_entity_blocks(data.decode("utf-8-sig")))
    tiers = [tier] if isinstance(tier, int) else list(tier)  # a unit from the library: each tier with parts
    roots = [n.unit_entity(tag, utype, t) for t in tiers]
    if not all(r in blocks for r in roots):
        return None, 0, len(blocks)
    rename = {}
    for t in tiers:
        rename[n.unit_entity(tag, utype, t)] = n.unit_entity(gfx, utype, t)
        if utype == "cavalry":
            rename[n.rider_entity(tag, t)] = n.rider_entity(gfx, t)
    keep, todo = [], list(roots)
    while todo:
        n = todo.pop()
        if n in keep or n not in blocks:
            continue
        keep.append(n)
        for target in re.findall(r'\b\w+\s*=\s*"([^"]+)"', " ".join(re.findall(r"attach\s*=\s*\{[^}]*\}", blocks[n]))):
            todo.append(target)
    out = []
    for n in keep:
        b = blocks[n]
        for old, new_name in rename.items():
            b = b.replace(f'"{old}"', f'"{new_name}"')
        out.append(b)
    return ("\n".join(out) + "\n").encode("utf-8"), len(keep), len(blocks)


def _entity_blocks(text: str) -> list[tuple[str, str]]:
    """Each top-level ``entity = { ... }`` block of an .asset file with its name."""
    out, i = [], 0
    while True:
        m = re.search(r"(?m)^\s*entity\s*=\s*\{", text[i:])
        if not m:
            return out
        start = i + m.start()
        depth, j = 0, i + m.end() - 1
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        block = text[start:j + 1]
        nm = re.search(r'name\s*=\s*"([^"]+)"', block)
        out.append((nm.group(1) if nm else "", block.strip()))
        i = j + 1


def add_to_mod(preview_spec: Path, subject: str, mode: str, tag: str, gfx: str, tier: int | list[int], mod_dir: Path,
               eu4: Path, stage_root: Path, write: bool = True, utype: str = "cavalry",
               credits: tuple[list[str], bool] | None = None, folder: str | None = None,
               display: str | None = None) -> AddResult:
    """Write the finished unit (its preview spec: the loadout and set) into ``mod_dir`` as mod ``<subject>_units``.
    ``mode``: "normal" (for ``tag``, graphical culture ``gfx``) or "achievement_safe" (``gfx``'s ``utype`` at
    ``tier``, cavalry or infantry: ``gfx`` an existing culture such as westerngfx). ``write`` False: the plan
    only. ``credits``: ``importer.credits_for``'s (lines, any CC BY): written as ``<mod>/eu4unitgen_credits.txt``
    (rule 8; in both modes: a text file in the mod's root is outside the checksum); a CC BY credit that can't be
    written stops the export."""
    import json

    from eu4unitgen.core.character import CharacterSpec
    from eu4unitgen.export import testmod
    from eu4unitgen.export.pipeline import Session, Settings
    from eu4unitgen.export.writer import ExportError, apply_export, check_writable, stage_plan
    from eu4unitgen.ref3d.unitstage import unit_names
    if mode not in ("normal", "achievement_safe", "culture_pack"):
        raise ValueError("mode is normal, achievement_safe or culture_pack")
    name = folder or mod_name(subject)  # ``folder``: the mod's own folder name (Track T: "tracen_academy")
    d = json.loads(Path(preview_spec).read_text(encoding="utf-8"))
    colours = next(iter(d["set"]["unit_colours"].values()))
    d["set"].update({"tags": [tag], "gfx": gfx, "set_name": name.removesuffix("_units"), "unit_colours": {tag: colours},
                     "unit_names": unit_names(subject)})
    # the unit at the chosen tier only (the preview's vanilla tier is gone with the test block)
    d.pop("test", None)
    work = Path(stage_root) / name
    work.mkdir(parents=True, exist_ok=True)
    sp = work / f"{name}.json"
    sp.write_text(json.dumps(d, indent=1), encoding="utf-8")
    s = Session(Settings(eu4_path=Path(eu4), owned_dlc=[], prefix="hr_", stage_dir=work / "stage", mod_dir=Path(mod_dir)))
    spec = CharacterSpec.load(sp).set_spec()
    game, idx = s.game()
    nation = s.nations(spec)[0]
    project = s.project(mod_root=Path(mod_dir) / name)
    tm = testmod.build_test_mod(project, nation, game, idx, Path(mod_dir), name=name, test_sprite_techs=None,
                                display=display, **s._parts_args(spec))
    plan = tm.plan
    rules = checksum_rules(eu4)
    out = AddResult(mode, name, None)
    if mode == "achievement_safe":
        keep, _, _ = keep_graphics([f.path for f in plan.files], name, gfx, tier, rules)
        files, assets = [], []
        for f in plan.files:
            if f.path not in keep:
                continue
            if f.path.lower().endswith(".asset") and "/gfx/entities/" in f.path.lower():
                assets.append(f)
                continue
            files.append(f)
        if assets:  # every entity file joined, cut down to the unit, written as one file
            joined = b"\n".join(a.data for a in assets)
            data, k, n = entities_only(joined, gfx, tier, tag, utype)
            if data is None:
                out.notes.append(f"validation: no {utype} entity at tier {tier} was generated for {tag}")
                return out
            assets[0].data = data
            assets[0].path = f"{name}/gfx/entities/{name}_{gfx}.asset"
            files.append(assets[0])
            out.notes.append(f"{k} of {n} entities kept: the unit as {gfx}'s {utype} at tier {tier}, with its parts")
        plan.files = files
        plan.manifest["files"] = {p: e for p, e in plan.manifest.get("files", {}).items() if p in {f.path for f in files}}
        # the validation ran on the whole set: its errors about what this mode leaves out don't apply (the sprite
        # packs, unit files, tech, countries, colours, localisation); nor does reusing the culture's own entity names
        # with a higher version, which is the point here
        plan.validation_errors = [e for e in plan.validation_errors if _rule_of(e) not in DROPPED_RULES]
    if mode == "culture_pack":
        nm = project.naming
        set_name = d["set"]["set_name"]
        foot = {int(t) for t in ((d.get("loadout") or {}).get("foot_cavalry") or {})}
        kept, notes = culture_pack(plan.files, name, gfx, tag, [1, 2, 3, 4], nm.set_entities_file(set_name),
                                   nm.gfx_entities_file(gfx), nm.set_pack_file(set_name), foot)
        plan.files = kept
        plan.manifest["files"] = {p: e for p, e in plan.manifest.get("files", {}).items() if p in {f.path for f in kept}}
        for f in kept:  # the rewritten culture file's hash
            if f.path in plan.manifest["files"]:
                from eu4unitgen.export.writer import sha256
                plan.manifest["files"][f.path]["sha256"] = sha256(f.data)
        out.notes += notes
        # the whole set was validated; what this mode leaves out (the tag's stats, tech, countries, pack) doesn't apply
        plan.validation_errors = [e for e in plan.validation_errors if _rule_of(e) not in DROPPED_RULES]
    from eu4unitgen.core import config as _cfg
    try:  # the user's EU4 mod folder (the export itself may go to a staging folder)
        install = _cfg.load().mods()
    except Exception:  # noqa: BLE001 - no settings: where it was written
        install = Path(mod_dir)
    out.readme = breakdown([f.path for f in plan.files] + [f"{name}/README.txt"], name, mode, install, tag, gfx,
                           display or "")
    from eu4unitgen.export.writer import plan_extra
    plan_extra(plan, f"{name}/README.txt", out.readme.encode("utf-8"))
    if credits and credits[0]:
        from eu4unitgen.export.writer import BLOCKED, plan_extra
        from eu4unitgen.ref3d import importer
        pf = plan_extra(plan, f"{name}/{importer.CREDITS_FILE}", importer.credits_text(credits[0]))
        if pf.action == BLOCKED and credits[1]:
            plan.validation_errors.append(f"ERROR   credits: {pf.path} {pf.reason}: a CC BY model's credit must ship "
                                          "with the mod")
    out.files = sorted(f.path for f in plan.files)
    out.checksum_files = checksum_files([p.split("/", 1)[1] if "/" in p else p for p in out.files], rules)
    if plan.validation_errors:
        out.notes += [f"validation: {e}" for e in plan.validation_errors]
        return out
    if not write:
        return out
    try:
        check_writable(Path(mod_dir))
        apply_export(plan)
        out.written_to = Path(mod_dir) / name
    except (OSError, ExportError):  # Controlled Folder Access: staged, with the copy steps
        staged = stage_plan(plan, work / "stage", require_empty=False)
        apply_export(staged)
        out.staged_at = work / "stage"
        out.steps = copy_steps(work / "stage", Path(mod_dir), name)
    return out
