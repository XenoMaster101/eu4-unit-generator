# Archive

The abandoned approaches, kept for reference only (not packaged, not tested, not imported by the app):

- `src/ref3d/`: the hand-built master pipeline (M15-M18: E-A precision fitting, E-B plates, E-C cloth panels, E-D 2D labels, the master assembly), the self-reading pipeline (`auto`, `parts`, `s1`), the cloak pipeline's game builder (`game`) and its app job (`job`).
- `src/ai/`: TripoSG (`triposg.py`, MIT; optional) and the AI jobs only those pipelines used (SigLIP labels, SAM 2 segments, SDXL inpaint, the fit, the measures).
- `src/blender/jobs/`: the master, plate, panel and repose jobs; the Phase 3 export round-trip spike.
- `src/gui/horse_page.py`: the M19 horse screen (replaced by the unit screen).
- `src/gui_nation_wizard/`: the keyword/vanilla-parts builder (Track F: the "describe your soldiers" Home, the Result
  screen, the Advanced editor's Settings, Unit set, Textures, Parts, Validate, Generate and Test mod pages, its main
  window) and the Track P unit screen, replaced on 2026-10-28 by the guided picture flow (`gui/home.py`,
  `gui/new_unit.py`, `gui/my_units.py`). The `core` modules they used (prompt, look, army) and the command line stay.
  Their tests: `tests/test_gui_nation_wizard.py`.
- `tests/`, `scripts/`: their tests and research scripts.

Git history has every version. The live pipeline is in `src/eu4unitgen` (see docs/DEVELOPER.md).
