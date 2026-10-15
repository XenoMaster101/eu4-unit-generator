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
2. Reference vanilla names and paths; never copy vanilla files into the mod unchanged. **Derived textures** (recoloured or edited from base-game files) are allowed in generated mod output only: never in the repo or the app, and never derived from DLC files. Generated files never `clone =` a vanilla entity: clones resolve in load order and vanilla's files load after ours (SPEC §3). Write full entity definitions that reference vanilla pdxmesh, animation and attach names instead.
3. All generated entity, sprite pack and pdxmesh names come from `core/naming.py`. No string-building of names anywhere else.
4. Every generator must have a matching validator rule. Generated output must pass `validate/` before export.
5. Export is non-destructive: dry-run diff first, write a manifest of generated files, and never overwrite files the tool didn't generate.
6. Don't invent EU4 syntax. If something isn't in SPEC.md or verified against the user's vanilla files, stop and ask, or add a `# UNVERIFIED` comment and a TODO.
7. Build in roadmap phase order. Don't start mesh or Blender work before Phases 0–1 are done and working.

## Conventions
- Code in `src/eu4unitgen/`, tests in `tests/` (pytest). Test template output against golden files.
- EU4 text files: write with UTF-8; localisation `.yml` files need a UTF-8 BOM.
- Mod-specific prefixes are configurable (default `hr_`).
- Keep the GUI thin: logic lives in `core/`, `pdxscript/`, `validate/`, and the GUI only calls them.

## The user
- Comfortable on Windows and Linux. Prefers direct, blunt communication.
- When you run or suggest terminal commands, explain in plain language what each one does, and mention the GUI equivalent when one exists.

## Current phase
Phase 3 (asset generation: Blender mesh jobs). Phases 0, 1 and 2 are done; see docs/SPEC.md §10. Update this line as phases complete.
