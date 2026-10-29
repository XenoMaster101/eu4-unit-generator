# EU4 Unit Generator (eu4unitgen)

Desktop GUI app that generates complete Europa Universalis IV **land unit** sets (infantry, cavalry, artillery × 4 sprite tiers) for a nation: all script files, dependencies, and eventually textures and meshes. The first target mod is the user's total conversion "Hellenistic Revolution". Ships, portraits and city models come later, so keep the architecture extensible.

**Read `docs/SPEC.md` before starting any task.** It has the full EU4 file chain, naming rules, example templates, architecture and roadmap.

## Stack (fixed; don't swap without asking)
- Python 3.11+, PySide6 for the GUI
- Pillow ≥ 11.2.1 for DDS read/write; texconv is an optional Windows-only encoder
- Jinja2 for script templates
- Blender (pinned version, 4.2 LTS) + ross-g/io_pdx_mesh addon, **only ever run as a subprocess** (`blender --background --python job.py -- args.json`). Never import bpy into the app process.
- Project license: GPL-3.0 (compatible with io_pdx_mesh)
- Cross-platform: must run on Windows and Linux. Use `pathlib`. No hard-coded path separators or Windows-only calls without a Linux fallback.

## Hard rules
1. **Never copy, bundle or commit Paradox game assets** (meshes, textures, anims, vanilla script files) into this repo or the template library. Read them from the user's EU4 install path at runtime. DLC assets must never be redistributed.
2. Reference vanilla names and paths; never copy vanilla files into the mod unchanged. **Derived textures and meshes** (recoloured, edited or extended from base-game files) are allowed in generated mod output only: never in the repo or the app, and never derived from DLC files. Derived body meshes (a base-game body edited or extended, e.g. with skinned tail, hair or cloth) are allowed on the same terms as derived textures (decided 2026-10-16, SPEC Track R). Rigid parts stay meshes made from scratch and attached to bones (`head` attach verified, in-game test 7). Generated files never `clone =` a vanilla entity: clones resolve in load order and vanilla's files load after ours (SPEC §3). Write full entity definitions that reference vanilla pdxmesh, animation and attach names instead.
3. All generated entity, sprite pack and pdxmesh names come from `core/naming.py`. No string-building of names anywhere else.
4. Every generator must have a matching validator rule. Generated output must pass `validate/` before export.
5. Export is non-destructive: dry-run diff first, write a manifest of generated files, and never overwrite files the tool didn't generate.
6. Don't invent EU4 syntax. If something isn't in SPEC.md or verified against the user's vanilla files, stop and ask, or add a `# UNVERIFIED` comment and a TODO.
7. Build in roadmap phase order. Don't start mesh or Blender work before Phases 0–1 are done and working.
8. All supplied reference images and generated outputs must be original, appropriately licensed, or otherwise authorized for the intended use. Do not reproduce protected character designs, franchise-specific uniforms, names, or logos without appropriate authorization. Reference images must be actively analyzed and incorporated into the planning process rather than ignored.

## Conventions
- Code in `src/eu4unitgen/`, tests in `tests/` (pytest). Test template output against golden files.
- EU4 text files: write with UTF-8; localisation `.yml` files need a UTF-8 BOM.
- Mod-specific prefixes are configurable (default `hr_`).
- Keep the GUI thin: logic lives in `core/`, `pdxscript/`, `validate/`, and the GUI only calls them.

## The user
- Comfortable on Windows and Linux. Prefers direct, blunt communication.
- When you run or suggest terminal commands, explain in plain language what each one does, and mention the GUI equivalent when one exists.

## Current phase
Phase 3 done; Track P (the app built from the lessons, then packaged) in progress: P1 (rules, parts from the reference, regression set, quality report), P2 (the one-flow unit screen, edits, checks, Add to mod normal or achievement-safe) and P3 (archive, data cleanup, settings file, docs, licences) done; P4 packaging and P5 done: everything in one folder chosen at install (default D:\EU4 Unit Generator: app, tools, ai, wsl, work, settings; `core/layout.py`), the zip ("Install EU4 Unit Generator", README, LICENSE, files\) and an unsigned setup.exe (Inno Setup); the author's PC migrated into it (regression: nothing worse); the old scattered copies and test output are deleted (docs/CLEANUP.md; only the source, the installed folder, the reference images and the 95 MB regression baseline remain); Windows Sandbox is switched off, so the clean-Windows test is dropped for now. Track U (the UI and flow update, 0.2.0) done: the top bar, the guided New unit flow, My units, Help (example, tour, picture guide, updates), soldier (on foot: infantry and gun crew; its hand parts in the base game's convention for its stance) and cavalry kinds, the candidate pick on outline, colour and worst stretch; the old builder archived. The soldier on foot is not yet checked in game. Track L (the part library; SPEC Track L): L1 SPEC, L2 the screens (the design pass: Admiralty default, the Library tile grid, plain part pages), L3 assembly (run kind `unit`), L4 the starter library (raw TRELLIS.2 models, fitted on each PC on first use, packed at installer build) done; L5 Track A built (the gun kind, `fresh_gun`, the tier's own cannon entity) and in-game test 37 staged (the bombard crewed by the cataphract); the generic bases: the English horse done, the English soldier needs work (a web under the raised arm), the bombard done. AI steps wait for free RAM and name what to close. A dev checkout uses the installed root via `.eu4unitgen-root` (tests never do). Track V (soldiers that move and read like the game's; SPEC Track V) built after T38: the joint fit, the animation sets and the game's weapon grip (`core.stances`), the animation review (`ref3d.review`), the painted pass, the triangle budget and proportions; review sheets in `build/review_t39`; T39 not staged. Track W (SPEC: a base body of our own, the outfit pushed onto it, loose cloth as sheets, rigid parts) planned after the T39 review sheets; W0 (weapons straight and rigid, checked) done; W1 (the base body) waits for the user's go and A-pose picture; soldier baselines wait for in-game approval. The cataphract (test 35, tier 1) is the reference result; the regression set (`ref3d.regress`) runs after every pipeline change. Track F (the nation wizard) is archived; its core modules and the command line stay. Update this line as phases complete.
