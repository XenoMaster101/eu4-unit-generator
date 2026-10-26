# Cleanup report (2026-10-25)

Nothing has been deleted or uninstalled. Tell me which lines to remove. After the move into `D:\EU4 Unit Generator`
(SPEC Track P5), everything the app needs is in that one folder; the rest below are old copies, caches and test output.

## The whole PC, outside the project folder

| # | Path | Size | What it is | Keep or delete |
|---|---|---|---|---|
| 1 | `D:\EU4 Unit Generator` | 63.9 GB | **the app's one folder now**: app, tools (Blender), ai, wsl (the Linux disk), work, settings | **keep** |
| 2 | `D:\eu4unitgen_ai` (what's left after the move) | 37.1 GB | the unused AI pieces: the Windows TRELLIS.2 environment (6.6 GB), uv's cache (11.8 GB), TripoSG weights and code (7.6 GB), SDXL inpainting (6.6 GB), SigLIP 2 (4.4 GB), CPython 3.11, uv, wheels, old logs | **delete** (the whole folder) |
| 3 | `D:\eu4unitgen_archive` | 13.6 GB | old tests, bake-offs and scratch moved out of the project on 2026-10-24 (details below) | **delete** (keep any test you might reload in game) |
| 4 | WSL distribution **Ubuntu** (`%LOCALAPPDATA%\wsl\{5e282d44-...}\ext4.vhdx`) | 37.4 GB | the original Linux disk; `EU4UnitGen` in the app's folder is its copy and works | **delete** (`wsl --unregister Ubuntu`) |
| 5 | Inside the new Linux disk: `/usr/local/cuda-12.8`, `~/eu4unitgen_ai/uv_cache`, `~/eu4unitgen_ai/ext` | 15.6 GB | an unused CUDA toolkit (everything links 12.9), a download cache, the extension build tree | **delete**, then let the disk shrink (`wsl --manage EU4UnitGen --set-sparse true`) to give the space back to D: |
| 6 | `C:\Program Files\Blender Foundation\Blender 4.2` (installed with its MSI) | 0.9 GB | the old Blender install; the app now uses its own copy in `tools\blender` | **uninstall** (Settings > Apps > Blender 4.2.23) |
| 7 | `%APPDATA%\Blender Foundation\Blender\4.2` | 2 MB | that Blender's settings and io_pdx_mesh (copied into the portable Blender) | **delete** |
| 8 | `C:\Program Files\Blender Foundation\Blender 5.2` | 0.9 GB | Blender 5.2.2, installed 2026-10-14; the app doesn't use it | **yours?** Keep if you installed it for yourself; otherwise uninstall |
| 9 | `%LOCALAPPDATA%\Programs\eu4unitgen` | 0.7 GB | the previous install (this morning's); its shortcuts and Apps entry now point at the new folder | **delete** |
| 10 | `%LOCALAPPDATA%\eu4unitgen` | 0.8 GB | the first installer's venv (`app\`), the fresh-user test's staged mod (`work\`), an old log | **delete** |
| 11 | `%APPDATA%\eu4unitgen` | 1 KB | the old settings (copied into the new folder) | **delete** |
| 12 | Registry `HKCU\Software\eu4unitgen` | tiny | the old window state (now `settings\gui.ini`) | **delete** |
| 13 | `%USERPROFILE%\.cache\huggingface` | 0.2 MB | **a saved Hugging Face token** (`token`, `stored_tokens`) and download logs | **delete**, and revoke that token on huggingface.co if you don't use it elsewhere |
| 14 | `%LOCALAPPDATA%\pip\cache` | 289 MB | pip's download cache | **delete** (pip rebuilds it if you use Python for other things) |
| 15 | `%TEMP%\eu4unitgen_fresh`, `%TEMP%\eu4unitgen_fresh2` | 1.4 GB | the fresh-user test installs | **delete** |
| 16 | `%TEMP%\blender_*`, `pdxzip`, `eu4unitgen_ai___*`, `eu4unitgen_arrow_*.png`, `eu4unitgen_install_step.log*`, `blender.crash.txt` | < 1 MB | leftovers from runs before the app kept its temp files in its folder | **delete** |
| 17 | `%TEMP%\claude` | 95 MB | Claude Code's own scratch folder (this session uses it) | delete **after** we're done |
| 18 | EU4 mod folder: `eu4unitgen_test\`, `eu4unitgen_test.mod` | 15 MB | the old test mod | **delete** (the other mods there, `Victorias_Secret` and the `ugc_*.mod` files, aren't ours) |
| 19 | `%USERPROFILE%\.wslconfig` | 1 KB | WSL's memory setting (20 GB), written for this project; WSL reads it only there | **keep** (the app's uninstaller removes it) |
| 20 | Windows feature: WSL | - | needed by the app | **keep** |
| 21 | Windows feature: Windows Sandbox | - | enabled for the clean-Windows installer test (not yet run: it needs a restart) | keep until that test, then turn off (Windows Features) |
| 22 | `C:\Python314` | 163 MB | Python for working on the app from source | **keep** while you develop (the installed app has its own) |
| 23 | Shortcuts and the Apps entry | - | now the new folder's | **keep** |

Nothing else was found: no scheduled tasks, no other registry keys, no `Documents\eu4unitgen`, no uv folder in the
profile, no Ubuntu Store package.

**Deleting everything marked delete frees about 108 GB** (about 39 on C: from the old Linux disk, temp files and
caches; about 69 on D:), plus 0.9 GB if Blender 5.2 isn't yours.

## C:\Users\antho\Projects\eu4unitgen (2.8 GB)

| Folder | Size | What it is | Safe to delete? |
|---|---|---|---|
| `.venv` | 710 MB | the dev Python environment (PySide6, pytest) | **no** (needed to run and test from source; rebuildable with `pip install -e .[dev]`) |
| `assets_in\regression` | 1.6 GB | the regression set: 3 raw models and inputs (~90 MB) plus every rebuild under `runs\` | the inputs: **no**. `runs\` except the newest version of each model: **yes**, about 1.3 GB |
| `assets_in\cataphract_budget` | 227 MB | test 35's horse v001 and riders v004/v005: the approved reference result and the baseline's source | **no** |
| `assets_in\horse2_flow` | 111 MB | horse 2 run through the new app flow (in the app's list) | yes, if you don't want it in the list |
| `assets_in\horse2_cands` | 165 MB | horse 2 with two TRELLIS.2 candidates (2026-10-25; both raw models kept) | yes, once you've seen its sheet |
| `assets_in\new_refs` | 9 MB | your reference images | **no** |
| `build\test35` | 30 MB | test 35's staged mod and gallery | **no** (the approved result) |
| `build\side_try`, `build\side_unit` | 102 MB | the shield-side test | **yes** |
| `build\shield_cmp` | 17 MB | the kite vs TRELLIS.2 shield comparison sheet | yes once you've seen it |
| `build\hem_try` | 13 MB | the hem-length test | **yes** |
| `build\lib`, `build\*.png`, `build\*.bak`, `build\LICENSE.gpl3.txt`, `build\regression` | 3 MB | build leftovers, screenshots, regression reports | **yes** (the reports are small; keep them if you like) |
| `dist` | 44 MB | the built package, the cached signed Python and the Sandbox results folder | yes after the Sandbox test (rebuilt by `build_package.py`) |
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
