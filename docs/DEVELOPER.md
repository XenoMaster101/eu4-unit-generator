# eu4unitgen for developers

The specification and full history are in [SPEC.md](SPEC.md); this is the short map.

## Set up a checkout

```
python -m venv .venv
.venv\Scripts\activate            # Linux: source .venv/bin/activate
pip install -e .[dev]
python -m eu4unitgen.gui          # the app
pytest                            # the tests (set EU4_PATH to run the checks against real game files)
```

Paths (EU4, the mod folder, the workspace, the AI environment, WSL, Blender) live in one settings file, `core/config.py`: `%APPDATA%\eu4unitgen\settings.json` (Linux: `~/.config/eu4unitgen/settings.json`). The app's Setup screen edits it. Environment variables still win (`EU4_PATH`, `BLENDER_PATH`, `EU4UNITGEN_AI_PYTHON`, `EU4UNITGEN_WORKSPACE`); `EU4UNITGEN_CONFIG_DIR` points at another settings folder (fresh-user tests). From a checkout, the workspace (`assets_in/`, `build/`) is the checkout itself.

## The pipeline

```
image ──cut-out──> RGBA ──TRELLIS.2 (WSL)──> raw.glb ──build──> game version vNNN ──finish──> finished/ ──preview──> sheet, quality report ──> Add to mod
        BiRefNet                               horse: trellis.process           edits                 unitstage + gallery
        (AI env)                               rider: trellis.process_rider
```

| Step | Code | Runs where |
|---|---|---|
| the app's run (stages, resume, waits while EU4 runs) | `ref3d/flow.py`, `ref3d/runstate.py` | a detached process |
| cut-out | `ref3d/trellis.py: cutout` → `ai/jobs/ref3d_mask.py` | the Windows AI environment |
| 3D model | `ref3d/trellis.py: generate` → `ai/jobs/trellis2_generate.py` (+ `_uv_raster.py`) | WSL (`setup_scripts/trellis2_setup.sh`) |
| horse build | `trellis.process` → Blender `fresh_horse` (direct path), `render_model`, `bake_colour`, `bake_horse`, `render_compare`, `check_mesh_deform` | Blender 4.2, as a subprocess |
| rider build | `trellis.process_rider` → `rider_frames`, `fresh_rider`, the bakes, parts (`ref3d/part_gen.py`, `core/gear.py`) | Blender + the AI environment (SAM 2) + WSL (TRELLIS.2 on part crops) |
| edits | `ref3d/edits.py` (build options vs finish on a copy) | the app's run |
| preview, gallery | `ref3d/unitstage.py`, `export/gallery.py` + Blender `render_unit` | |
| quality report | `ref3d/quality.py` (pure Python) | |
| Add to mod | `export/modexport.py` (normal or achievement-safe; the checksum from the install's `checksum_manifest.txt`) on `export/testmod.py`, `export/writer.py` | |
| checks, guide, errors, setup | `core/checks.py`, `core/guide.py`, `core/errors.py`, `core/setup.py` | |
| GUI | `gui/unit_page.py` (the flow), `gui/setup_page.py` (Setup, About), `gui/app.py`; the nation wizard behind Home | |

The app's own process has no numpy: everything it runs directly is pure Python + Pillow. numpy code lives in Blender jobs (`blender/jobs/`) and AI jobs (`ai/jobs/`). Blender is never imported into the app (CLAUDE.md).

## Where each rule lives

Every lesson from the horse and rider tracks is a general rule with a test, listed in `core/known_fixes.py` (the rule, the case that taught it, the code, the test). The table is in SPEC ("Known fixes"); `python -m eu4unitgen.core.known_fixes` prints it, and `tests/test_known_fixes.py` checks that every entry's code and test exist.

Rules that live in Blender jobs are tested inside Blender: `tests/blender_rules_check.py` runs the jobs' numpy helpers (`_colour.py`, `_part_view.py`, `_reins.py`) in Blender's Python, `tests/test_blender_rules.py` reads the results.

## The regression set

Three reference models, rebuilt from their raw TRELLIS.2 models after every pipeline change: horse 2, the cataphract horse and the cataphract rider (test 35, approved in game).

```
python -m eu4unitgen.ref3d.regress run                 # about 8 minutes; exit code 1 if anything got worse
python -m eu4unitgen.ref3d.regress baseline --run build/regression/<time>             # accept a run as the new baseline
python -m eu4unitgen.ref3d.regress baseline --run build/regression/<time> --add-new   # only add new measures
```

Inputs: `assets_in/regression/<name>/` (raw.glb, input/, subject.json; the rider's part crops are cached in `parts_cache/`). Baseline: `regression/baseline.json` (numbers only). Report: `build/regression/<time>/report.md`. Measures and tolerances: `ref3d/regress.py` (`METRICS`, `RULES`).

## Licences

`core/licences.py` lists every model and tool (shown on the About page and in SPEC §9). No Paradox file is ever copied into the repo or the package (rule 1).

## The nation wizard and the command line

The older Track F/Phase 1-3 tools are behind **Home** → **Advanced editor** (Settings, Unit set, Textures, Parts, Validate, Generate, Test mod) and on the command line: `python -m eu4unitgen.cli --help` (`generate`, `test-mod`, `pick-test-tag`, `parts-preview`). SPEC §10 describes them.

## Archive

`archive/` holds the abandoned approaches (the hand-built master, cloth panels, precision fitting, the procedural and self-reading horse pipelines, TripoSG), not packaged or tested. See `archive/README.md`.

## Packaging (Windows)

`python packaging/windows/build_package.py` builds `dist/eu4unitgen-<version>-windows.zip`: the app's wheel, `Install.cmd` and `install.ps1` (Python found or installed, a venv in `%LOCALAPPDATA%\eu4unitgen\app`, shortcuts). The AI side is set up from the app's Setup screen (`core/setup.py`).
