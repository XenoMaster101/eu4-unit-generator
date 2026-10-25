# Cleanup report (2026-10-25)

Nothing has been deleted. Each line says whether it's safe to delete; tell me which to remove.

## C:\Users\antho\Projects\eu4unitgen (2.8 GB)

| Folder | Size | What it is | Safe to delete? |
|---|---|---|---|
| `.venv` | 710 MB | the dev Python environment (PySide6, pytest) | **no** (needed to run and test from source; rebuildable with `pip install -e .[dev]`) |
| `assets_in\regression` | 1.6 GB | the regression set: 3 raw models and inputs (~90 MB) plus every rebuild under `runs\` | the inputs: **no**. `runs\` except the newest version of each model: **yes**, about 1.3 GB |
| `assets_in\cataphract_budget` | 227 MB | test 35's horse v001 and riders v004/v005: the approved reference result and the baseline's source | **no** |
| `assets_in\horse2_flow` | 111 MB | horse 2 run through the new app flow (in the app's list) | yes, if you don't want it in the list |
| `assets_in\new_refs` | 9 MB | your reference images | **no** |
| `build\test35` | 30 MB | test 35's staged mod and gallery | **no** (the approved result) |
| `build\side_try`, `build\side_unit` | 102 MB | the shield-side test | **yes** |
| `build\shield_cmp` | 17 MB | the kite vs TRELLIS.2 shield comparison sheet | yes once you've seen it |
| `build\hem_try` | 13 MB | the hem-length test | **yes** |
| `build\lib`, `build\*.png`, `build\*.bak`, `build\LICENSE.gpl3.txt`, `build\regression` | 3 MB | build leftovers, screenshots, regression reports | **yes** (the reports are small; keep them if you like) |
| `dist` | 1.6 MB | the built package | yes (rebuilt by `build_package.py`) |
| `output` | 0.4 MB | old wizard output | yes |
| `.git`, `src`, `tests`, `docs`, `archive`, `packaging` | 37 MB | the project | **no** |

Safe here: about 1.4 GB (plus `horse2_flow` if you don't want it).

## D:\eu4unitgen_archive\2026-10-24 (14 GB)

Everything moved out of the project on 2026-10-24: old tests, bake-offs and scratch. The project no longer reads any of it; the regression set has its own copies of the raw models it needs.

| Folder | Size | What it is | Safe to delete? |
|---|---|---|---|
| `assets_in\cataphract_horse` | 4.3 GB | the M17-M20 hand-built master runs | **yes** |
| `assets_in\bakeoff_m19` | 2.3 GB | the 3D-tool bake-off (TRELLIS.2's first cataphract and horse 2 raw models, also copied into the regression set) | **yes** |
| `assets_in\horse_ref` | 2.0 GB | Track C/M horse reference runs | **yes** |
| `assets_in\cataphract_rider` | 1.9 GB | the rider's development runs (its raw model is in the regression set) | **yes** |
| `build\macedon` | 736 MB | the Phase 1 vertical slice (Macedon) | **yes** |
| `assets_in\cataphract_budget` | 537 MB | the detail-experiment tiers 2-3 and earlier rider versions (test 33) | yes (keep if you might revisit test 33's budgets) |
| `assets_in\horse2_light_cavalry`, `horse2_auto`, `horse2_t2`, `cataphract_auto`, `horse2_flip_test`, `proof_t24` | 1.0 GB | horse 2 and cataphract runs of the older pipelines (M18-M22) | **yes** |
| `build\regress_try` | 414 MB | the first regression try | **yes** |
| `build\test11`-`test34`, `look33`, `q0`-`q5`, `c0`, `hoplite`, `plan11`, `spike_r0`, `tmp_*` | ~420 MB | old in-game test mods and spikes | **yes** (keep any test you might want to reload in game) |

Safe here: all 14 GB.

## The AI side (see the dependency inventory in SPEC or the app's About page)

Safe to remove: about 27 GB in all (the Windows TRELLIS.2 environment, TripoSG, SDXL inpainting, SigLIP 2, caches and the WSL build tree). The required pieces (TRELLIS.2, its weights, DINOv3, SAM 2.1, BiRefNet, the Windows AI environment, the WSL environment, CUDA) stay.
