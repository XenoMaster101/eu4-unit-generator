"""Command line: generate and validate unit sets, build the throwaway test mod.

    python -m eu4unitgen.cli generate --tag MAC --gfx hellenicgfx [--out output] [--dry-run]
    python -m eu4unitgen.cli pick-test-tag
    python -m eu4unitgen.cli test-mod --tag HDA --gfx hellenicgfx --set macedonian_test [--stage-dir DIR]
    python -m eu4unitgen.cli test-mod --suite --tag HDA --gfx hellenicgfx --gfx-set-base muslimgfx [--stage-dir DIR]

generate writes a drop-in package to <out>/<set>/ (new files + INTEGRATION.md) and never
writes into a mod folder. test-mod is a dry run unless --apply (into the EU4 mod
directory) or --stage-dir (into an empty folder, for copying by hand).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eu4unitgen.core.generator import generate
from eu4unitgen.core.naming import TEST_MOD_NAME
from eu4unitgen.core.nation import Nation
from eu4unitgen.core.project import DEFAULT_ENTITY_VERSION, DEFAULT_TAG_PACK_ORDER, Project, default_mod_dir
from eu4unitgen.export.testmod import build_suite, build_test_mod, suite_checklist
from eu4unitgen.export.writer import ExportError, ExportPlan, apply_export, plan_package, stage_plan
from eu4unitgen.library.scanner import GameFiles, build_index
from eu4unitgen.library.tags import coverage, culture_groups, start_level, tag_facts
from eu4unitgen.validate import Report, validate

NATIVE_TECH = {"north_american", "mesoamerican", "south_american", "andean", "high_american",
               "aboriginal_tech", "polynesian_tech"}


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--eu4", type=Path, help="EU4 install (default: $EU4_PATH)")
    p.add_argument("--owned-dlc", nargs="*", default=[], help="dlc folder names you own")


def _set_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tag", required=True)
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


def _print_result(report: Report, plan: ExportPlan, args: argparse.Namespace, write: bool) -> int:
    print(f"validator: {report.summary()}")
    for issue in report.issues:
        print(f"  {issue}")
    text = plan.diff_text()
    if args.diff_out:
        args.diff_out.write_text(text, encoding="utf-8")
        print(f"diff written to {args.diff_out}")
    print(text)
    stage = getattr(args, "stage_dir", None)
    if write or stage:
        target = stage_plan(plan, stage) if stage else plan
        try:
            written = apply_export(target)
        except ExportError as exc:
            print(f"EXPORT FAILED, nothing written: {exc}")
            return 1
        print(f"wrote {len(written)} files and the manifest under {target.mod_root}")
        if stage:
            print(f"staged: copy everything inside {target.mod_root} into {plan.mod_root}")
    else:
        print("dry run: nothing written")
    return 0 if plan.can_apply else 1


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate + validate, then write the drop-in package to <out>/<set>/. Never writes to a mod:
    --mod is only read, and only with --check-mod."""
    project = Project(mod_root=args.mod or default_mod_dir() / "hellenistic_revolution", eu4_path=args.eu4,
                      prefix=args.prefix, entity_version=args.version, tag_pack_order=args.pack_order,
                      owned_dlc=args.owned_dlc, check_mod_collisions=args.check_mod)
    nation = Nation(args.tag, args.gfx, base_gfx=args.base_gfx, name=args.name, adjective=args.adjective,
                    colour=tuple(args.colour) if args.colour else None, tech_group=args.tech_group,
                    set_name=args.set_name)
    game = GameFiles(project.eu4_path, include_dlc=project.include_dlc)
    mod = project.mod_files()
    try:
        idx = build_index(game)
        generated = generate(project, nation, game, idx, mod)
        report = validate(project, nation, generated, game, idx, mod)
        try:
            plan = plan_package(args.out, generated, game, [str(i) for i in report.errors],
                                mod_dirs=[default_mod_dir(), project.mod_root])
        except ExportError as exc:
            print(f"REFUSED: {exc}")
            return 1
    finally:
        game.close()
    code = _print_result(report, plan, args, write=not args.dry_run)
    if not args.dry_run and code == 0:
        print(f"package: {plan.mod_root / nation.set_name} (copy its contents into your mod's root; "
              "merge INTEGRATION.md by hand)")
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


def cmd_test_suite(args: argparse.Namespace, project: Project, nation: Nation, game: GameFiles, idx, facts,
                   sprite_techs: dict[int, int] | None, mod_dir: Path) -> int:
    fallback = dict(zip("abc", args.fallback_tags)) if args.fallback_tags else None
    suite = build_suite(project, nation, game, idx, mod_dir, facts, culture_groups(game), rival_gfx=args.rival_gfx,
                        test_sprite_techs=sprite_techs, fallback_tags=fallback)
    code = 0
    for i, tm in enumerate(suite.variants):
        print(f"=== variant {i + 1}: {tm.plan.manifest_rel.split('/', 1)[0]}")
        print(f"validator: {tm.report.summary()}")
        for issue in tm.report.issues:
            print(f"  {issue}")
        if args.diff_out:
            out = args.diff_out.with_name(f"{args.diff_out.stem}_v{i + 1}{args.diff_out.suffix}")
            out.write_text(tm.plan.diff_text(), encoding="utf-8")
            print(f"diff written to {out}")
        print(tm.plan.summary())
        if not tm.plan.can_apply:
            print(tm.plan.diff_text())
            code = 1
    if code == 0 and (args.apply or args.stage_dir):
        targets = []
        try:
            for i, tm in enumerate(suite.variants):
                targets.append(stage_plan(tm.plan, args.stage_dir, require_empty=i == 0) if args.stage_dir else tm.plan)
            for target in targets:
                written = apply_export(target)
                print(f"wrote {len(written)} files and the manifest under {target.mod_root / target.manifest_rel.split('/')[0]}")
        except ExportError as exc:
            print(f"EXPORT FAILED: {exc}")
            return 1
        if args.stage_dir:
            print(f"staged: copy everything inside {args.stage_dir} into {mod_dir}")
    elif code == 0:
        print("dry run: nothing written")
    print()
    print(suite_checklist(suite, sprite_techs))
    return code


def cmd_test_mod(args: argparse.Namespace) -> int:
    mod_dir = args.mod_dir or default_mod_dir()
    project = Project(mod_root=mod_dir / TEST_MOD_NAME, eu4_path=args.eu4, prefix=args.prefix,
                      entity_version=args.version, tag_pack_order=args.pack_order, owned_dlc=args.owned_dlc)
    game = GameFiles(project.eu4_path, include_dlc=project.include_dlc)
    try:
        idx = build_index(game)
        facts = tag_facts(game).get(args.tag)
        if facts is None:
            print(f"{args.tag} owns no land at 1444.11.11; pick another tag (see pick-test-tag)")
            return 1
        cov = coverage(idx, facts)
        if not cov.clean:
            print(f"WARNING: {args.tag} is matched by {[p for p, _, _ in cov.packs]} / has {cov.entities[:4]}; "
                  "the map may not show the generated set")
        nation = Nation(args.tag, args.gfx, base_gfx=args.base_gfx, name=args.name, adjective=args.adjective,
                        colour=tuple(args.colour) if args.colour else None,
                        tech_group=args.tech_group or facts.technology_group, set_name=args.set_name,
                        gfx_set_base=args.gfx_set_base)
        sprite_techs = None
        if args.test_sprite_techs:
            sprite_techs = dict(zip((2, 3, 4), args.test_sprite_techs))
            start = start_level(game, facts.technology_group)
            if start is not None and min(sprite_techs.values()) <= start:
                print(f"WARNING: {args.tag} starts at MIL {start}; tiers at MIL <= {start} show from day one")
        if args.suite:
            return cmd_test_suite(args, project, nation, game, idx, tag_facts(game), sprite_techs, mod_dir)
        tm = build_test_mod(project, nation, game, idx, mod_dir, test_sprite_techs=sprite_techs)
    finally:
        game.close()
    print(f"test mod for {args.tag} ({facts.primary_culture}, {facts.technology_group}, was {facts.graphical_culture})")
    return _print_result(tm.report, tm.plan, args, write=args.apply)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eu4unitgen", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate + validate + dry-run export for one unit set")
    _set_args(g)
    _common(g)
    g.add_argument("--tech-group", default="western")
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
    t.add_argument("--fallback-tags", nargs=3, metavar=("A", "B", "C"),
                   help="suite: tags for fallback cases a, b, c (default: picked automatically)")
    t.add_argument("--rival-gfx", default="asiangfx", help="suite: vanilla set the rival priority pack lists")
    t.add_argument("--gfx-set-base", default=None,
                   help="vanilla culture the --gfx default set copies (default: --base-gfx). The suite "
                        "uses a different look here so case c can be told apart from a westerngfx fallback")
    t.set_defaults(func=cmd_test_mod)

    args = ap.parse_args(argv)
    if getattr(args, "eu4", None) is None:
        project = Project(mod_root=Path("."))
        if project.eu4_path is None:
            ap.error("pass --eu4 or set EU4_PATH")
        args.eu4 = project.eu4_path
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
