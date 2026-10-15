"""Command line: generate and validate unit sets, build the throwaway test mod.

    python -m eu4unitgen.cli generate --tag MAC --gfx hellenicgfx [--out output] [--dry-run]
    python -m eu4unitgen.cli generate --tag MAC EPI --set argead --gfx hellenicgfx   (one set, two tags)
    python -m eu4unitgen.cli pick-test-tag
    python -m eu4unitgen.cli test-mod --tag HDA --gfx hellenicgfx --set macedonian_test [--stage-dir DIR]
    python -m eu4unitgen.cli test-mod --textures --tag HDA --gfx hellenicgfx --set macedonian_test --test-sprite-techs 2 3 4
    python -m eu4unitgen.cli test-mod --suite --tag HDA --gfx hellenicgfx --gfx-set-base muslimgfx [--stage-dir DIR]
    python -m eu4unitgen.cli test-mod --tag HDA --gfx hellenicgfx --set macedonian_test --loadout loadout.json --stage-dir DIR
    python -m eu4unitgen.cli parts-preview --tag HDA --gfx hellenicgfx --set macedonian_test --loadout loadout.json --tier 1 --role infantry

generate and test-mod run through export/pipeline.py, the same code path as the GUI.
generate writes a drop-in package to <out>/<set>/ (new files + INTEGRATION.md) and never
writes into a mod folder. test-mod is a dry run unless --apply (into the EU4 mod
directory) or --stage-dir (into an empty folder, for copying by hand).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eu4unitgen.core.generator import GenerationError
from eu4unitgen.core.naming import TEST_MOD_NAME
from eu4unitgen.core.project import DEFAULT_ENTITY_VERSION, DEFAULT_TAG_PACK_ORDER, Project
from eu4unitgen.export.pipeline import Session, SetSpec, Settings, SpecError
from eu4unitgen.export.writer import ExportError, ExportPlan
from eu4unitgen.library.scanner import GameFiles, build_index
from eu4unitgen.library.tags import coverage, tag_facts
from eu4unitgen.validate import Report

NATIVE_TECH = {"north_american", "mesoamerican", "south_american", "andean", "high_american",
               "aboriginal_tech", "polynesian_tech"}


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--eu4", type=Path, help="EU4 install (default: $EU4_PATH)")
    p.add_argument("--owned-dlc", nargs="*", default=[], help="dlc folder names you own")


def _set_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tag", dest="tags", nargs="+", default=[], type=str.upper,
                   help="one tag, or several sharing one set (one sprite pack, OR over the tags; --set required)")
    p.add_argument("--culture", dest="cultures", nargs="+", default=[], type=str.lower,
                   help="primary cultures the set's sprite pack also matches (OR with the tags), so the look "
                        "survives tag changes; with no --tag, --set is required")
    p.add_argument("--gfx", required=True, help="the nation's graphical culture, e.g. hellenicgfx")
    p.add_argument("--set", dest="set_name", default="", help="unit set name (default: the tag in lowercase)")
    p.add_argument("--base-gfx", default="westerngfx", help="vanilla culture whose set is copied")
    p.add_argument("--name", default="")
    p.add_argument("--adjective", default="")
    p.add_argument("--colour", type=int, nargs=3, default=None, metavar=("R", "G", "B"))
    p.add_argument("--prefix", default="hr_")
    p.add_argument("--version", type=int, default=DEFAULT_ENTITY_VERSION, help="entity version")
    p.add_argument("--pack-order", type=int, default=DEFAULT_TAG_PACK_ORDER)
    p.add_argument("--diff-out", type=Path, help="also write the diff to this file")
    p.add_argument("--unit-colours", type=int, nargs=9, metavar="V",
                   help="unit colours color1 color2 color3 as 9 numbers (R G B R G B R G B) for every tag "
                        "(common/country_colors)")
    p.add_argument("--tag-unit-colours", nargs=10, action="append", metavar="X",
                   help="unit colours for one tag: TAG R G B R G B R G B (repeat per tag; overrides --unit-colours)")
    p.add_argument("--texture-recipe", type=Path, help="Phase 2: texture recipe JSON (retextures the TAG set)")
    p.add_argument("--loadout", type=Path, help="Phase 3: parts loadout JSON (procedural parts; runs Blender)")
    p.add_argument("--blender", type=Path, help="Blender executable (default: the pinned 4.2 LTS path)")


def _triple(v: list[int]) -> tuple:
    return tuple(v[0:3]), tuple(v[3:6]), tuple(v[6:9])


def _unit_colours(args: argparse.Namespace):
    """--unit-colours (every tag) and --tag-unit-colours (one tag each) -> {tag: triple} or None."""
    out = {t: _triple(args.unit_colours) for t in args.tags} if args.unit_colours else {}
    for entry in args.tag_unit_colours or []:
        tag, nums = entry[0].upper(), entry[1:]
        if not all(n.isdigit() for n in nums):
            raise SpecError(f"--tag-unit-colours {tag}: nine numbers 0-255 expected, got {' '.join(nums)}")
        out[tag] = _triple([int(n) for n in nums])
    return out or None


def _settings(args: argparse.Namespace, **kw) -> Settings:
    if getattr(args, "blender", None):
        kw.setdefault("blender_path", args.blender)
    return Settings(eu4_path=args.eu4, owned_dlc=list(args.owned_dlc), prefix=args.prefix,
                    entity_version=args.version, tag_pack_order=args.pack_order,
                    mod_dir=getattr(args, "mod_dir", None), **kw)


def _spec(args: argparse.Namespace) -> SetSpec:
    return SetSpec(tags=list(args.tags), cultures=list(args.cultures), gfx=args.gfx, set_name=args.set_name, base_gfx=args.base_gfx, name=args.name,
                   adjective=args.adjective, colour=tuple(args.colour) if args.colour else None,
                   tech_group=args.tech_group or "", gfx_set_base=getattr(args, "gfx_set_base", None),
                   unit_colours=_unit_colours(args), texture_recipe=args.texture_recipe, loadout=args.loadout)


def _print_plan(label: str | None, report: Report, plan: ExportPlan, args: argparse.Namespace, suffix: str = "") -> None:
    if label:
        print(f"=== {label}")
    print(f"validator: {report.summary()}")
    for issue in report.issues:
        print(f"  {issue}")
    text = plan.diff_text()
    if args.diff_out:
        out = args.diff_out.with_name(f"{args.diff_out.stem}{suffix}{args.diff_out.suffix}") if suffix else args.diff_out
        out.write_text(text, encoding="utf-8")
        print(f"diff written to {out}")
    print(text)


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate + validate, then write the drop-in package to <out>/<set>/ (export/pipeline.py).
    Never writes to a mod: --mod is only read, and only with --check-mod."""
    session = Session(_settings(args, output_dir=args.out, mod_folder=args.mod if args.check_mod else None))
    try:
        results = session.generate(_spec(args), write=not args.dry_run)
    except ExportError as exc:
        print(f"REFUSED: {exc}")
        return 1
    finally:
        session.close()
    code = 0
    for r in results:
        _print_plan(None, r.report, r.plan, args)
        if not r.plan.can_apply:
            code = 1
        elif args.dry_run:
            print("dry run: nothing written")
        else:
            print(f"wrote {len(r.written)} files and the manifest under {r.plan.mod_root}")
            print(f"package: {r.package_dir} (copy its contents into your mod's root; merge INTEGRATION.md by hand)")
    return code


def cmd_pick_test_tag(args: argparse.Namespace) -> int:
    game = GameFiles(args.eu4, include_dlc=True)
    try:
        idx = build_index(game)
        facts = tag_facts(game)
    finally:
        game.close()
    clean, near = [], []
    for tag, f in facts.items():
        cov = coverage(idx, f)
        if cov.clean:
            clean.append(f)
        elif not cov.entities and all(dlc and dlc not in args.owned_dlc for _, dlc, _ in cov.packs):
            near.append((f, cov))
    print(f"{len(facts)} tags own land at 1444.11.11")
    print(f"Not matched by any vanilla or DLC sprite pack and no {{TAG}}_INFANTRY/CAVALRY/ARTILLERY entities: {len(clean)}")
    for f in sorted(clean, key=lambda f: (f.technology_group in NATIVE_TECH, -f.provinces)):
        print(f"  {f.tag}  {f.provinces:3} prov  {f.primary_culture} ({f.culture_group})  {f.technology_group}  "
              f"{f.graphical_culture}  {f.country_file}")
    if args.owned_dlc:
        print(f"Matched only by DLC packs you don't own ({', '.join(args.owned_dlc)} owned): {len(near)}")
        for f, cov in sorted(near, key=lambda t: -t[0].provinces)[:args.limit]:
            print(f"  {f.tag}  {f.provinces:3} prov  {f.primary_culture}  {f.technology_group}  <- "
                  + ", ".join(sorted({dlc for _, dlc, _ in cov.packs})))
    return 0 if clean else 1


def cmd_test_mod(args: argparse.Namespace) -> int:
    """The test mod, the in-game test suite (--suite) or the texture test (--textures), through
    export/pipeline.py. A dry run unless --apply (into the EU4 mod directory) or --stage-dir."""
    write = "apply" if args.apply else "stage" if args.stage_dir else "dry"
    session = Session(_settings(args, stage_dir=args.stage_dir or Path("build") / "test_mod"))
    techs = dict(zip((2, 3, 4), args.test_sprite_techs)) if args.test_sprite_techs else None
    try:
        res = session.build_test_mod(_spec(args), suite=args.suite, textures=args.textures, sprite_techs=techs,
                                     base_mod=args.base_mod, give_cores=args.give_cores, base_name=args.base_name,
                                     rival_gfx=args.rival_gfx, write=write,
                                     fallback_tags=dict(zip("abc", args.fallback_tags)) if args.fallback_tags else None)
    except ExportError as exc:
        print(f"EXPORT FAILED, nothing written: {exc}")
        return 1
    finally:
        session.close()
    print(res.info)
    for w in res.warnings:
        print(f"WARNING: {w}")
    several = len(res.plans) > 1
    for i, ((name, report), plan) in enumerate(zip(res.reports, res.plans)):
        _print_plan(name if several else None, report, plan, args, f"_v{i + 1}" if several else "")
    code = 0 if all(p.can_apply for p in res.plans) else 1
    if code == 0 and write == "dry":
        print("dry run: nothing written")
    elif res.written:
        print(f"wrote {len(res.written)} files and the manifest(s) under {res.stage_dir}")
        if write == "stage":
            print(f"staged: copy everything inside {res.stage_dir} into {session.settings.user_mod_dir()}")
    if res.checklist:
        print()
        print(res.checklist)
    return code


def cmd_parts_preview(args: argparse.Namespace) -> int:
    """Render one loadout slot (a body with its parts, front and side, and each part alone) in
    background Blender, to <out>/part_previews/<set>/<tier>_<role>.png."""
    from eu4unitgen.export.pipeline import load_loadout

    session = Session(_settings(args, output_dir=args.out))
    try:
        spec = _spec(args)
        loadout = load_loadout(spec)
        if loadout is None:
            raise SpecError("pass --loadout")
        pv = session.parts_preview(spec, args.tier, args.role, loadout)
    finally:
        session.close()
    print(f"body {pv.body}; preview written to {pv.path}")
    for part in pv.parts:
        fitted = ", ".join(f"{k} {v}" for k, v in sorted(part["fitted"].items()))
        print(f"  {part['entity']} on {part['bone']} ({part['shader']})" + (f"; fitted: {fitted}" if fitted else ""))
    for n in pv.notes:
        print(f"NOTE: {n}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eu4unitgen", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate + validate + dry-run export for one unit set")
    _set_args(g)
    _common(g)
    g.add_argument("--tech-group", default=None, help="unit_type for the units (default: the tag's 1444 tech group, else western)")
    g.add_argument("--out", type=Path, default=Path("output"), help="output root; the package goes to <out>/<set>/")
    g.add_argument("--dry-run", action="store_true", help="show the plan and diff, write nothing")
    g.add_argument("--mod", type=Path, help="mod folder to READ with --check-mod (default: <EU4 mod dir>/hellenistic_revolution)")
    g.add_argument("--check-mod", action="store_true",
                   help="read the mod folder's existing files for collisions and shared-file copies (off by default)")
    g.set_defaults(func=cmd_generate)

    p = sub.add_parser("pick-test-tag", help="list 1444 tags no vanilla/DLC sprite pack touches")
    _common(p)
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_pick_test_tag)

    t = sub.add_parser("test-mod", help=f"build the throwaway '{TEST_MOD_NAME}' mod around one unit set")
    _set_args(t)
    _common(t)
    t.add_argument("--tech-group", default=None, help="unit_type for the units (default: the tag's 1444 tech group)")
    t.add_argument("--mod-dir", type=Path, help="EU4 user mod directory (default: the one under Documents)")
    t.add_argument("--stage-dir", type=Path,
                   help="write the finished mod here instead (empty folder) and copy it into the mod directory by hand")
    t.add_argument("--apply", action="store_true", help="write into the EU4 mod directory (default is a dry run)")
    t.add_argument("--test-sprite-techs", type=int, nargs=3, metavar=("T2", "T3", "T4"),
                   help="TEST ONLY: MIL techs for sprite levels 2-4 in the test mod's mil.txt, e.g. 2 3 4")
    t.add_argument("--suite", action="store_true",
                   help="build the in-game test suite: two variant mods (pack priority swapped) with fallback "
                        "cases a/b/c on extra tags, and print the checklist")
    t.add_argument("--textures", action="store_true",
                   help="build the texture test instead (eu4unitgen_test_textures): tier 1 retextured through "
                        "pdxmesh meshsettings, and print the checklist")
    t.add_argument("--base-mod", type=Path,
                   help="build the test mod as a copy of this mod folder (only read) with the set integrated: "
                        "enable it instead of that mod")
    t.add_argument("--base-name", help="launcher name of the base-mod copy (default: '<its name> - unitgen test')")
    t.add_argument("--give-cores", action="store_true",
                   help="TEST ONLY, with --base-mod: the tag owns its cored provinces and capital at the start")
    t.add_argument("--fallback-tags", nargs=3, metavar=("A", "B", "C"),
                   help="suite: tags for fallback cases a, b, c (default: picked automatically)")
    t.add_argument("--rival-gfx", default="asiangfx", help="suite: vanilla set the rival priority pack lists")
    t.add_argument("--gfx-set-base", default=None,
                   help="vanilla culture the --gfx default set copies (default: --base-gfx). The suite "
                        "uses a different look here so case c can be told apart from a westerngfx fallback")
    t.set_defaults(func=cmd_test_mod)

    v = sub.add_parser("parts-preview", help="render one loadout slot (body + parts) in background Blender")
    _set_args(v)
    _common(v)
    v.add_argument("--tech-group", default=None, help=argparse.SUPPRESS)
    v.add_argument("--tier", type=int, choices=(1, 2, 3, 4), required=True)
    v.add_argument("--role", choices=("infantry", "rider", "crew"), default="infantry")
    v.add_argument("--out", type=Path, default=Path("output"), help="output root; previews go to <out>/part_previews/")
    v.set_defaults(func=cmd_parts_preview)

    args = ap.parse_args(argv)
    if getattr(args, "eu4", None) is None:
        project = Project(mod_root=Path("."))
        if project.eu4_path is None:
            ap.error("pass --eu4 or set EU4_PATH")
        args.eu4 = project.eu4_path
    from eu4unitgen.core.loadout import LoadoutError
    from eu4unitgen.core.recipe import RecipeError

    try:
        return args.func(args)
    except (GenerationError, RecipeError, LoadoutError, SpecError) as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
