"""The app's one flow (Track P2): image -> name and rights -> horse or rider, unit slot -> Generate -> stages ->
the result sheet and quality report -> edits -> Add to mod.

A run lives in ``<workspace>/assets_in/<name>/``: ``request.json`` (what was asked), ``status.json`` (each stage's
state and time, a heartbeat), ``trellis/`` (the cut-out, TRELLIS.2's model, the built versions), ``finished/`` (the
built version with the finish edits), ``preview/`` (staged with vanilla next, the gallery, the sheet, the quality
report). It runs detached (``start``): closing the app doesn't stop it, and a stopped or failed run resumes from
its first unfinished stage (every stage checks what is already on disk). GPU stages (TRELLIS.2, and the rider's
build, which makes its shield with TRELLIS.2) wait while EU4 is open.

  stages   cutout (BiRefNet) -> trellis (TRELLIS.2 in WSL) -> build (processing, parts, bakes, checks) ->
           finish (the finish edits on a copy) -> preview (stage, gallery, quality report, sheet)

usage: python -m eu4unitgen.ref3d.flow go <run folder>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

STAGES = ("cutout", "trellis", "build", "finish", "preview")
LABELS = {"cutout": "Cut the subject out of the image", "trellis": "Make two 3D models (TRELLIS.2) and pick the better",
          "build": "Build the game model (shape, parts, textures, checks)", "finish": "Apply your edits",
          "preview": "Stage it and draw the result sheet"}
GPU = {"trellis", "build"}
# horse 2's and the cataphract rider's runs (2026-10-23/24) until this machine has its own
DEFAULT_SECONDS = {"cutout": 45, "trellis": 820, "build_horse": 95, "build_rider": 380, "finish": 5, "preview": 150}
KINDS = ("horse", "rider")  # the foot soldier is on the roadmap (SPEC Track P)
EU4_POLL = 20.0


def root() -> Path:
    from eu4unitgen.core import config
    return config.load().work() / "assets_in"


def folder_of(name: str) -> Path:
    return root() / name


# ------------------------------------------------------------------ requests
def new_request(name: str, image: Path, rights: str, kind: str, tier: int = 1, horse: str | None = None,
                options: dict | None = None) -> Path:
    """A new run's folder and request (refused if the name is taken)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    f = folder_of(name)
    if f.exists():
        raise FileExistsError(f"a run called {name} exists already")
    (f / "trellis").mkdir(parents=True)
    req = {"schema": 1, "name": name, "image": str(Path(image).resolve()), "rights": rights, "kind": kind, "tier": int(tier),
           "horse": horse, "options": options or {}, "edits": {}, "created": time.time()}
    (f / "request.json").write_text(json.dumps(req, indent=1), encoding="utf-8")
    subject = {"schema": 1, "name": name, "template": kind, "rights": rights, "original": str(Path(image).resolve()),
               "pipeline": "trellis", "written_by": "ref3d.flow (the app)"}
    (f / "subject.json").write_text(json.dumps(subject, indent=1), encoding="utf-8")
    return f


def request(folder: Path) -> dict:
    return json.loads((Path(folder) / "request.json").read_text(encoding="utf-8"))


def set_edits(folder: Path, edits: dict) -> str | None:
    """New edits for a run: the stage it goes back to ("build", "finish" or None), marked waiting from there on."""
    from eu4unitgen.ref3d import edits as ed
    folder = Path(folder)
    req = request(folder)
    back = ed.stage_for(req.get("edits"), edits)
    req["edits"] = ed.merged(edits)
    (folder / "request.json").write_text(json.dumps(req, indent=1), encoding="utf-8")
    if back:
        st = read_status(folder) or {"stages": {}}
        for s in STAGES[STAGES.index(back):]:
            (st["stages"].setdefault(s, {}))["state"] = "waiting"
        st["done"] = False
        _write(folder, st)
    return back


# ------------------------------------------------------------------ status
def read_status(folder: Path) -> dict | None:
    from eu4unitgen.ref3d import runstate
    return runstate.read_status(folder)


def status(folder: Path) -> dict | None:
    from eu4unitgen.ref3d import runstate
    return runstate.status(folder)


def _write(folder: Path, st: dict) -> None:
    p = Path(folder) / "status.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def runs() -> list[dict]:
    """Every run in the workspace (a request.json), newest first, with its status."""
    out = []
    for p in root().glob("*/request.json"):
        try:
            req = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        st = status(p.parent) or {"state": "new", "stages": {}}
        out.append({**req, "folder": str(p.parent), "state": st.get("state", "new"), "status": st})
    return sorted(out, key=lambda r: -r.get("created", 0))


def estimates(kind: str) -> dict[str, float]:
    """Each stage's expected seconds: the median of this machine's finished runs, else the defaults."""
    seen: dict[str, list] = {s: [] for s in STAGES}
    for r in runs():
        if r.get("kind") != kind:
            continue
        for s, v in (r["status"].get("stages") or {}).items():
            if s in seen and v.get("state") == "done" and v.get("seconds"):
                seen[s].append(v["seconds"])
    return {s: statistics.median(v) if v else DEFAULT_SECONDS.get(f"{s}_{kind}", DEFAULT_SECONDS.get(s, 60))
            for s, v in seen.items()}


# ------------------------------------------------------------------ the stages
def _latest(d: Path) -> Path | None:
    vs = sorted(p for p in Path(d).glob("v[0-9][0-9][0-9]") if (p / "manifest.json").is_file())
    return vs[-1] if vs else None


def stage_cutout(folder: Path, req: dict, cfg) -> dict:
    from eu4unitgen.ref3d import trellis
    inp = folder / "trellis" / "input"
    if (inp / "input_rgba.png").is_file() and (inp / "input_fg.png").is_file():
        return {"kept": True}
    trellis.cutout(Path(req["image"]), inp, cfg.ai())
    return {}


def stage_trellis(folder: Path, req: dict, cfg) -> dict:
    """Two TRELLIS.2 candidates (seeds 42 and 43, one model load), each scored by its outline against the reference;
    the better one becomes ``raw.glb`` (an edit can pick the other: ``candidate``)."""
    from eu4unitgen.blender.runner import BlenderRunner
    from eu4unitgen.ref3d import trellis
    tr = folder / "trellis"
    cj = tr / "candidates.json"
    if (tr / "raw.glb").is_file() and cj.is_file():
        return {"kept": True}
    cands = trellis.generate_candidates((tr / "input" / "input_rgba.png").resolve(), tr, cfg.ai_root_path())
    br = BlenderRunner(cfg.blender(), log_dir=tr / "logs", timeout=1200)
    for c in cands:
        c.update(trellis.score_candidate(br, Path(c["glb"]), tr / "input" / "input_fg.png", tr / f"cand_s{c['seed']}"))
    best = max(range(len(cands)), key=lambda i: cands[i]["iou"])
    cj.write_text(json.dumps({"candidates": cands, "best": best}, indent=1), encoding="utf-8")
    use_candidate(folder, best)
    # TRELLIS.2's own timings under their own name: the stage's "seconds" is its wall time
    return {"candidates": [{"seed": c["seed"], "iou": c["iou"]} for c in cands], "chosen": best + 1}


def candidates(folder: Path) -> dict | None:
    p = Path(folder) / "trellis" / "candidates.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def use_candidate(folder: Path, index: int) -> None:
    """Candidate ``index`` (0-based) copied to ``raw.glb``, the model every later stage builds from."""
    c = candidates(folder)
    if c is None:
        return
    shutil.copyfile(c["candidates"][index]["glb"], Path(folder) / "trellis" / "raw.glb")


def horse_textures(name: str | None) -> dict | None:
    """The horse a rider rides (another run's built version): its textures, primary and geometry (palette, legs)."""
    if not name:
        return None
    f = folder_of(name)
    v = f / "finished" if (f / "finished" / "manifest.json").is_file() else _latest(f / "trellis" / "game")
    if v is None:
        return None
    m = json.loads((v / "manifest.json").read_text(encoding="utf-8"))
    return {"diffuse": m["textures"]["diffuse"], "specular": m["textures"]["specular"],
            "primary": (m.get("colour") or {}).get("default_primary"), "geometry": m["geometry"], "dir": str(v)}


def stage_build(folder: Path, req: dict, cfg) -> dict:
    from eu4unitgen.ref3d import edits as ed
    from eu4unitgen.ref3d import trellis
    c = candidates(folder)
    if c is not None:  # the candidate asked for (an edit), else the better-scored one
        pick = (req.get("edits") or {}).get("candidate")
        use_candidate(folder, (int(pick) - 1) if pick else c["best"])
    opts = ed.build_options(req.get("edits"), req["kind"])
    glb = folder / "trellis" / "raw.glb"
    if req["kind"] == "horse":
        o = dict(req.get("options") or {})
        emblem = ed.emblem_file(opts.pop("emblem", None) or o.get("emblem"), folder)
        m = trellis.process(glb, folder / "trellis" / "game", emblem=emblem,
                            country_mask=opts.get("country_mask", bool(o.get("country_mask"))), cfg=opts.get("cfg"))
    else:
        m = trellis.process_rider(glb, folder / "trellis" / "game", palette_from=horse_textures(req.get("horse")),
                                  part_cache=folder / "trellis" / "parts_cache", **opts)
    return {"version": m["version"]}


def stage_finish(folder: Path, req: dict, cfg) -> dict:
    from eu4unitgen.ref3d import edits as ed
    v = _latest(folder / "trellis" / "game")
    if v is None:
        raise RuntimeError("there is no built version to finish")
    ed.finish(v, req.get("edits"), req["kind"], folder / "finished")
    return {"from": v.name}


def stage_preview(folder: Path, req: dict, cfg) -> dict:
    """Staged with vanilla at the next tier, the gallery (close, near, map zoom), the seated clearance, the quality
    report and the result sheet."""
    from eu4unitgen.ref3d import quality, unitstage
    fin = folder / "finished"
    m = json.loads((fin / "manifest.json").read_text(encoding="utf-8"))
    subj = json.loads((folder / "subject.json").read_text(encoding="utf-8"))
    tier = int(req.get("tier", 1))
    horse = rider = None
    if req["kind"] == "horse":
        horse = unitstage.horse_entry(fin, subj)
    else:
        rider = unitstage.rider_entry(fin, subj)
        h = horse_textures(req.get("horse"))
        if h:
            hs = json.loads((folder_of(req["horse"]) / "subject.json").read_text(encoding="utf-8"))
            horse = unitstage.horse_entry(Path(h["dir"]), hs)
    pv = folder / "preview"
    if pv.exists():
        shutil.rmtree(pv)
    names = unitstage.unit_names(req["name"])
    sp = unitstage.spec(f"{req['name']}: preview (tier {tier}; vanilla at tier {tier + 1})", tier, horse, rider, names,
                        unitstage.TEST)
    st = unitstage.stage(pv, sp, "preview")
    if st["exit"] != 0:
        raise RuntimeError(f"staging the preview failed; see {pv / 'log.txt'}")
    g = unitstage.gallery(pv, "preview", [tier, tier + 1], {tier: req["name"], tier + 1: "vanilla"}, cfg.eu4())
    legs = None
    if req["kind"] == "rider":
        from eu4unitgen.core.character import CharacterSpec
        from eu4unitgen.export.gallery import GAME_CLOSE
        from eu4unitgen.export.pipeline import Session, Settings
        s = Session(Settings(eu4_path=cfg.eu4(), owned_dlc=[], prefix="hr_", stage_dir=pv / "gallery_stage"))
        s.gallery(CharacterSpec.load(pv / "preview.json").set_spec(), units=[("cavalry", tier), ("cavalry", tier + 1)],
                  camera=GAME_CLOSE, dump_dir=pv / "posed")
        legs = {"ours": quality.legs_clear(pv / "posed" / f"cavalry_{tier}_posed.json"),
                "vanilla": quality.legs_clear(pv / "posed" / f"cavalry_{tier + 1}_posed.json")}
    view = "side" if req["kind"] == "horse" else "front"
    imgs = (m.get("compare") or {}).get("images") or {}
    rep = quality.report(m, req["kind"], folder / "trellis" / "input" / "input_fg.png",
                         Path(imgs["game"][view]) if imgs.get("game", {}).get(view) else None,
                         Path(g["images"].get(f"map_three_quarter_{tier}", "")) or None,
                         Path(g["images"].get(f"map_three_quarter_{tier + 1}", "")) or None,
                         Path(imgs["raw"][view]) if imgs.get("raw", {}).get(view) else None, legs,
                         exclude=[Path(m.get("edited_from", fin)) / "parts" / "crops" / f"{k}_mask.png"
                                  for k in ("lance", "shield", "crest")] if req["kind"] == "rider" else None)
    quality.save(rep, pv)
    sheet = result_sheet(folder, req, g, rep)
    return {"sheet": str(sheet), "quality": rep["summary"], "verdict": rep["verdict"]}


def result_sheet(folder: Path, req: dict, gallery: dict, rep: dict) -> Path:
    """The reference next to the result (close three-quarter and side, map zoom against vanilla) and the quality
    report's lines."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font, big = ImageFont.truetype("arial.ttf", 18), ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font = big = ImageFont.load_default()
    tier = int(req.get("tier", 1))
    cell = 420
    tiles = [("the reference", Image.open(req["image"]).convert("RGB"))]
    for key, label in ((f"close_three_quarter_{tier}", "in game, close (three-quarter)"), (f"close_side_{tier}", "in game, close (side)"),
                       (f"map_three_quarter_{tier}", "map zoom"), (f"map_three_quarter_{tier + 1}", "map zoom: vanilla")):
        p = gallery["images"].get(key)
        if p:
            tiles.append((label, Image.open(p).convert("RGB")))
    lines = [rep["summary"]] + [f"{quality_title(k)} ({x['verdict']}): {x['text']}" for k, x in rep["measures"].items()]
    text_h = 30 + 24 * sum(1 + len(ln) // 150 for ln in lines)
    sheet = Image.new("RGB", (len(tiles) * cell, cell + 40 + text_h), (232, 232, 236))
    d = ImageDraw.Draw(sheet)
    for i, (label, im) in enumerate(tiles):
        im = im.copy()
        im.thumbnail((cell - 10, cell - 10))
        sheet.paste(im, (i * cell + (cell - im.width) // 2, 36 + (cell - im.height) // 2))
        d.text((i * cell + 8, 8), label, fill=(20, 20, 24), font=font)
    y = cell + 50
    d.text((8, y), lines[0], fill=(20, 20, 24), font=big)
    y += 34
    for ln in lines[1:]:
        while ln:
            d.text((8, y), ln[:150], fill=(30, 30, 34), font=font)
            ln, y = ln[150:], y + 24
    out = Path(folder) / "preview" / "result_sheet.png"
    sheet.save(out)
    return out


def quality_title(key: str) -> str:
    from eu4unitgen.ref3d.quality import TITLES
    return TITLES.get(key, key)


RUN = {"cutout": stage_cutout, "trellis": stage_trellis, "build": stage_build, "finish": stage_finish,
       "preview": stage_preview}


# ------------------------------------------------------------------ the run (a detached process)
def go(folder: Path) -> dict:
    from eu4unitgen.core import checks, config
    from eu4unitgen.ref3d.runstate import Status as _Status
    folder = Path(folder)
    req = request(folder)
    cfg = config.load()
    old = status(folder)
    if old and old.get("alive") and old.get("pid") != os.getpid() and old.get("state") == "running":
        raise SystemExit(f"{req['name']} is already running (process {old.get('pid')})")
    stages = (old or {}).get("stages", {})
    for s in stages.values():  # a stage cut off by the stop runs again
        if s.get("state") in ("running", "waiting_eu4"):
            s["state"] = "waiting"
        if not isinstance(s.get("seconds", 0), (int, float)):  # a time recorded wrongly (0.1.0's trellis stage)
            s.pop("seconds")
    st = _Status(folder, {"name": req["name"], "kind": req["kind"], "pid": os.getpid(), "started": (old or {}).get("started", time.time()),
                          "resumed": bool(old), "stages": stages, "error": None, "trace": None, "done": False,
                          "sheet": (old or {}).get("sheet")})
    try:
        for s in STAGES:
            if (st.d["stages"].get(s) or {}).get("state") == "done":
                continue
            if s in GPU:
                while checks.eu4_running():  # never while EU4 is open
                    st.stage(s, "waiting_eu4")
                    time.sleep(EU4_POLL)
            st.stage(s, "running")
            res = RUN[s](folder, request(folder), cfg)
            st.stage(s, "done", **{k: v for k, v in res.items() if k != "sheet"})
            if res.get("sheet"):
                st.write(sheet=res["sheet"], quality=res.get("quality"), verdict=res.get("verdict"))
        st.write(done=True)
    except BaseException as exc:  # noqa: BLE001 - recorded for the app, then raised
        st.write(error="".join(traceback.format_exception_only(type(exc), exc)).strip(), trace=traceback.format_exc())
        raise
    finally:
        st.close()
    return st.d


def start(folder: Path) -> subprocess.Popen:
    """``go`` in its own process, detached (closing the app doesn't stop it); output to ``<folder>/run.log``."""
    folder = Path(folder)
    cmd = [sys.executable, "-m", "eu4unitgen.ref3d.flow", "go", str(folder)]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}
    log = open(folder / "run.log", "a", encoding="utf-8")  # noqa: SIM115 - handed to the child
    kw: dict = {"env": env, "stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL}
    from eu4unitgen.core import config
    kw["cwd"] = str(config.checkout_root() or config.load().work())
    if sys.platform == "win32":
        kw["creationflags"] = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW)
    else:
        kw["start_new_session"] = True
    try:
        return subprocess.Popen(cmd, **kw)
    finally:
        log.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("go")
    g.add_argument("folder", type=Path)
    a = p.parse_args(argv)
    go(a.folder)


if __name__ == "__main__":
    main()
