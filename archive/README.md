# Archive

The abandoned approaches, kept for reference only (not packaged, not tested, not imported by the app):

- `src/ref3d/`: the hand-built master pipeline (M15-M18: E-A precision fitting, E-B plates, E-C cloth panels, E-D 2D labels, the master assembly), the self-reading pipeline (`auto`, `parts`, `s1`), the cloak pipeline's game builder (`game`) and its app job (`job`).
- `src/ai/`: TripoSG (`triposg.py`, MIT; optional) and the AI jobs only those pipelines used (SigLIP labels, SAM 2 segments, SDXL inpaint, the fit, the measures).
- `src/blender/jobs/`: the master, plate, panel and repose jobs; the Phase 3 export round-trip spike.
- `src/gui/horse_page.py`: the M19 horse screen (replaced by the unit screen).
- `tests/`, `scripts/`: their tests and research scripts.

Git history has every version. The live pipeline is in `src/eu4unitgen` (see docs/DEVELOPER.md).
