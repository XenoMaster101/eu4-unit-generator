"""The app's one flow (Track P2): image -> name and rights -> what it is, unit slot -> Generate -> stages ->
the result sheet and quality report -> edits -> Add to mod.

Kinds: ``horse`` (from a side view); ``soldier`` (one person from a front view, on the infantry skeleton: on foot it
is the infantry and the gun crew, and it can ride); ``cavalry`` (a finished horse and a finished soldier put
together: no picture of its own, so only build, finish and preview: the soldier is built again for that horse, its
legs clear of the horse's width and its parts on the horse's palette); ``rider`` (older runs: a rider from its own
picture for a chosen horse).

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
LABELS = {"import": "Prepare your 3D model", "cutout": "Cut the subject out of the image", "trellis": "Make two 3D models (TRELLIS.2) and pick the better",
          "build": "Build the game model (shape, parts, textures, checks)", "finish": "Apply your edits",
          "preview": "Stage it and draw the result sheet", "assemble": "Put the parts together and check them"}
# what each stage does and why, in plain words (the Make it step and the Library show these)
EXPLAIN = {"import": "Your model is joined into one, stood up and turned to face the front, its colours put on one "
                     "texture, and drawn from the front as its reference picture. It takes the place of the AI's "
                     "3D model.",
           "cutout": "The background is removed so only the subject becomes part of the model.",
           "trellis": "An AI (TRELLIS.2) guesses the 3D shape and colours from your one picture. It makes two "
                      "versions; the app keeps the one whose outline, colours and movement come out best.",
           "build": "The model is cut down to the game's size, fitted onto the game's own skeleton so it moves with "
                    "the game's animations, and painted. Shields, plumes and weapons are made as separate parts.",
           "finish": "Your edits (colour, brightness, plume and so on) are applied to a copy.",
           "preview": "The unit is put into a test copy of the game's files and drawn the way the game draws it, next to "
                      "the game's own unit, and checked.",
           "assemble": "Your parts are put together for each tier (the soldier seated on the horse, or on foot) and "
                       "checked against the game's rules. No AI is needed."}
GPU = {"trellis", "build"}
# free RAM each AI step needs before it starts (GB): TRELLIS.2 in WSL takes up to 20 GB (killed at 3.2 GB free,
# exit 137, 2026-10-28); the build runs SAM 2 and may make a part with TRELLIS.2
NEED_RAM = {"cutout": 4.0, "trellis": 14.0, "build": 8.0}
RAM_POLL = 30.0
KIND_STAGES = {"cavalry": ("build", "finish", "preview"),  # the rest run every stage
               "unit": ("assemble", "preview")}  # a unit from library parts (ref3d.assemble): no AI
IMPORTED = ("import", "build", "finish", "preview")  # a part from the user's own 3D model (ref3d.importer)
# horse 2's and the cataphract rider's runs (2026-10-23/24) until this machine has its own
DEFAULT_SECONDS = {"import": 60, "cutout": 45, "trellis": 910, "build_horse": 95, "build_rider": 380, "finish": 5, "preview": 150,
                   "assemble": 40}
KINDS = ("horse", "soldier", "cavalry", "rider", "gun")
FOOT_TYPE = {"horse": "cavalry", "soldier": "infantry", "cavalry": "cavalry", "rider": "cavalry",
             "gun": "artillery"}  # the unit slot's type


def stages_of(kind: str, req: dict | None = None) -> tuple[str, ...]:
    """A run's stages: by its kind, or for a part made from an imported model (``req``'s ``source``) ``IMPORTED``."""
    if req is not None and (req.get("source") or {}).get("model"):
        return IMPORTED
    return KIND_STAGES.get(kind, STAGES)


def body_kind(kind: str) -> str:
    """How the build, edits and quality report treat it: a horse, a gun (rigid, Track A), or a person on the
    infantry skeleton."""
    return kind if kind in ("horse", "gun") else "rider"
EU4_POLL = 20.0


def root() -> Path:
    """The projects folder (one folder per unit)."""
    from eu4unitgen.core import config
    return config.load().projects()


def folder_of(name: str) -> Path:
    return root() / name


# ------------------------------------------------------------------ requests
def new_request(name: str, image: Path | None, rights: str, kind: str, tier: int = 1, horse: str | None = None,
                options: dict | None = None, soldier: str | None = None, model: dict | None = None) -> Path:
    """A new run's folder and request (refused if the name is taken). A cavalry run names its ``horse`` and
    ``soldier`` (finished runs) instead of an image: the soldier's picture and both rights are recorded. A part
    from an imported 3D model has ``model`` ({"path", "licence", "credit"}: ``ref3d.importer``) instead of an image;
    its rights are its licence."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    source = None
    if model is not None:
        from eu4unitgen.ref3d import importer
        if kind not in ("soldier", "horse", "gun"):
            raise ValueError("only a soldier, horse or gun can be made from a 3D model")
        why = importer.problem(Path(model["path"]), model.get("licence"), model.get("credit", ""))
        if why:
            raise ValueError(why)
        source = importer.source_record(Path(model["path"]), model["licence"], model.get("credit", ""))
        rights = importer.rights_text(model["licence"], model.get("credit", ""))
    f = folder_of(name)
    if f.exists():
        raise FileExistsError(f"a run called {name} exists already")
    edits = {}
    if kind == "cavalry":
        if not (horse and soldier):
            raise ValueError("a cavalry unit needs a finished horse and a finished soldier")
        h, sd = request(folder_of(horse)), request(folder_of(soldier))
        image = Path(sd["image"]) if sd.get("image") else None  # an imported soldier's: its render, once made
        rights = rights or f"horse: {h['rights']}; soldier: {sd['rights']}"
        edits = {k: v for k, v in (sd.get("edits") or {}).items() if k != "candidate"}  # the soldier's look
    (f / "trellis").mkdir(parents=True)
    image = keep_reference(Path(image)) if source is None and image else None  # an imported model's: its render
    req = {"schema": 1, "name": name, "image": str(Path(image).resolve()) if image else None, "rights": rights, "kind": kind,
           "tier": int(tier), "horse": horse, "soldier": soldier, "options": options or {}, "edits": edits,
           "created": time.time(), **({"source": source} if source else {})}
    (f / "request.json").write_text(json.dumps(req, indent=1), encoding="utf-8")
    subject = {"schema": 1, "name": name, "template": body_kind(kind), "rights": rights,
               "original": source["model"] if source else str(Path(image).resolve()) if image else None,
               "pipeline": "import" if source else "trellis", **({"source": source} if source else {}),
               "written_by": "ref3d.flow (the app)"}
    (f / "subject.json").write_text(json.dumps(subject, indent=1), encoding="utf-8")
    return f


def keep_reference(image: Path) -> Path:
    """The reference copied into the references folder (unless it's already there), so a project never depends on
    a file that may move; the same image twice is kept once. -> the copy."""
    import hashlib

    from eu4unitgen.core import config
    refs = config.load().references()
    image = Path(image).resolve()
    if refs.resolve() in image.parents:
        return image
    refs.mkdir(parents=True, exist_ok=True)
    data = image.read_bytes()
    q = refs / image.name
    n = 1
    while q.exists() and hashlib.sha256(q.read_bytes()).digest() != hashlib.sha256(data).digest():
        q = refs / f"{image.stem}_{n}{image.suffix}"
        n += 1
    if not q.exists():
        q.write_bytes(data)
    return q


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


def estimates(kind: str, req: dict | None = None) -> dict[str, float]:
    """Each stage's expected seconds: the median of this machine's finished runs, else the defaults."""
    seen: dict[str, list] = {s: [] for s in stages_of(kind, req)}
    for r in runs():
        if r.get("kind") != kind:
            continue
        for s, v in (r["status"].get("stages") or {}).items():
            if s in seen and v.get("state") == "done" and v.get("seconds"):
                seen[s].append(v["seconds"])
    return {s: statistics.median(v) if v else DEFAULT_SECONDS.get(f"{s}_{body_kind(kind)}", DEFAULT_SECONDS.get(s, 60))
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
    """Two TRELLIS.2 candidates (seeds 42 and 43, one model load), each scored on its outline and colour against
    the reference and on its worst stretch once fitted to the game skeleton (``trellis.pick_score``); the better
    one becomes ``raw.glb`` (an edit can pick the other: ``candidate``)."""
    from eu4unitgen.blender.runner import BlenderRunner
    from eu4unitgen.ref3d import trellis
    tr = folder / "trellis"
    cj = tr / "candidates.json"
    if (tr / "raw.glb").is_file() and cj.is_file():
        return {"kept": True}
    cands = trellis.generate_candidates((tr / "input" / "input_rgba.png").resolve(), tr, cfg.ai_root_path())
    br = BlenderRunner(cfg.blender(), log_dir=tr / "logs", timeout=1200)
    build_kind = body_kind(req["kind"])
    for c in cands:
        out = tr / f"cand_s{c['seed']}"
        c.update(trellis.score_candidate(br, Path(c["glb"]), tr / "input" / "input_fg.png", out))
        c["colour_de"] = trellis.score_colour(br, Path(c["glb"]), tr / "input" / "input_rgba.png", out, c["azimuth"])
        try:
            if build_kind != "gun":  # a gun is rigid: nothing stretches
                c.update(trellis.trial_deform(Path(c["glb"]), build_kind, out / "trial", cfg.blender()))
        except Exception as exc:  # noqa: BLE001 - scored on outline and colour alone; recorded
            c["trial_error"] = f"{type(exc).__name__}: {exc}"
        c["score"] = trellis.pick_score(c)
    best = max(range(len(cands)), key=lambda i: cands[i]["score"])
    cj.write_text(json.dumps({"candidates": cands, "best": best}, indent=1), encoding="utf-8")
    use_candidate(folder, best)
    # TRELLIS.2's own timings under their own name: the stage's "seconds" is its wall time
    return {"candidates": [{k: c.get(k) for k in ("seed", "iou", "colour_de", "worst_stretch", "score")} for c in cands],
            "chosen": best + 1}


def candidate_text(c: dict) -> str:
    """One candidate's scores in plain words (the edits' 3D model list, the Example, the Library)."""
    bits = [f"outline {c['iou']:.0%}"] if c.get("iou") is not None else []
    if c.get("colour_de") is not None:
        bits.append(f"colour distance {c['colour_de']:.0f}")
    if c.get("worst_stretch") is not None:
        bits.append(f"worst stretch {c['worst_stretch']:.2f}")
    return ", ".join(bits) or c.get("note") or "not scored"


def candidates(folder: Path) -> dict | None:
    p = Path(folder) / "trellis" / "candidates.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def use_candidate(folder: Path, index: int) -> None:
    """Candidate ``index`` (0-based) copied to ``raw.glb``, the model every later stage builds from."""
    c = candidates(folder)
    if c is None:
        return
    shutil.copyfile(c["candidates"][index]["glb"], Path(folder) / "trellis" / "raw.glb")


def soldier_glb(req: dict) -> Path:
    """The 3D model a person is built from: its own (``raw.glb``, the candidate picked), or for a cavalry unit its
    soldier's."""
    if req["kind"] == "cavalry":
        return folder_of(req["soldier"]) / "trellis" / "raw.glb"
    return folder_of(req["name"]) / "trellis" / "raw.glb"


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
    opts = ed.build_options(req.get("edits"), body_kind(req["kind"]))
    glb = soldier_glb(req) if req["kind"] not in ("horse", "gun") else folder / "trellis" / "raw.glb"
    if req["kind"] == "gun":  # Track A: rigid, on the base game's gun skeleton
        return {"version": trellis.process_gun(glb, folder / "trellis" / "game", cfg.blender())["version"]}
    if req["kind"] == "horse":
        o = dict(req.get("options") or {})
        emblem = ed.emblem_file(opts.pop("emblem", None) or o.get("emblem"), folder)
        m = trellis.process(glb, folder / "trellis" / "game", emblem=emblem,
                            country_mask=opts.get("country_mask", bool(o.get("country_mask"))), cfg=opts.get("cfg"))
    else:  # a soldier alone has no horse: its own palette, its legs as built
        cache = glb.parent / "parts_cache"  # a cavalry unit shares its soldier's part models (cached by crop)
        soldier = req["kind"] == "soldier"
        from eu4unitgen.ref3d import base_body
        body = base_body.current() if soldier else None  # Track W: the outfit over the base body, once it passed
        m = trellis.process_rider(glb, folder / "trellis" / "game", palette_from=horse_textures(req.get("horse")),
                                  part_cache=cache, fit="standing" if soldier else "seated", body=body, **opts)
        if soldier:  # two fits (the user, T37): standing for infantry and the crew, its hand parts strapped for its
            # stance; seated for riding (ref3d.assemble seats it on a horse from this one)
            trellis.to_foot(folder / "trellis" / "game" / m["version"], cfg.blender())
            seat = trellis.process_rider(glb, folder / "trellis" / "game_seated", palette_from=None, part_cache=cache,
                                         fit="seated", **opts)
            return {"version": m["version"], "seated": seat["version"]}
    return {"version": m["version"]}


def stage_finish(folder: Path, req: dict, cfg) -> dict:
    from eu4unitgen.ref3d import edits as ed
    v = _latest(folder / "trellis" / "game")
    if v is None:
        raise RuntimeError("there is no built version to finish")
    ed.finish(v, req.get("edits"), body_kind(req["kind"]), folder / "finished")
    seated = _latest(folder / "trellis" / "game_seated") if req["kind"] == "soldier" else None
    if seated is not None:  # the seated fit, with the same edits, for riding
        ed.finish(seated, req.get("edits"), body_kind(req["kind"]), folder / "finished_seated")
    return {"from": v.name, **({"seated": seated.name} if seated else {})}


def gun_report(m: dict) -> dict:
    """A gun's quality report (Track A) in plain words: its outline against the raw model's, where its muzzle is
    against the base game's flash, its wheels, its size against the budget."""
    fr = m.get("fresh") or {}
    iou = ((m.get("compare") or {}).get("iou") or {})
    best = max(iou.values()) if iou else None
    mz = fr.get("muzzle") or {}
    tris = (fr.get("final") or {}).get("tris", 0)
    wheels = (fr.get("shares") or {}).get("wheels", 0)
    meas = {
        "silhouette": {"verdict": "good" if (best or 0) > 0.8 else "ok" if (best or 0) > 0.65 else "check",
                       "text": f"The game version's outline overlaps TRELLIS.2's model by {best:.0%}." if best is not None
                       else "Not measured."},
        "muzzle": ({"verdict": "good", "text": "The flash fires from its own muzzle (the game's cannon_muzzle point "
                                               "moved onto it; to be confirmed in game)."}
                   if mz.get("locator") == "moved to ours" else
                   {"verdict": "good" if mz.get("off", 9) < 0.6 else "ok" if mz.get("off", 9) < 1.2 else "check",
                    "text": f"Its muzzle is {mz.get('off', '?')} from where the game fires the flash (the base game's "
                            "cannon_muzzle point, kept as it is)."}),
        "wheels": {"verdict": "good" if 0.08 < wheels < 0.6 else "check",
                   "text": f"{wheels:.0%} of it turns with the wheels; the rest recoils with the barrel."},
        "budget": {"verdict": "good" if tris <= m.get("budget", 1500) else "ok",
                   "text": f"{tris} triangles (the base game's gun has 682; budget {m.get('budget', 1500)})."}}
    worst = max((x["verdict"] for x in meas.values()), key=["good", "ok", "check"].index)
    return {"verdict": worst, "summary": {"good": "Good.", "ok": "Usable; see the notes.",
                                          "check": "Look at it in game first."}[worst], "measures": meas}


def stage_preview_gun(folder: Path, req: dict, cfg) -> dict:
    """A gun at its tier with the game's own crew, the game's gun at the next tier; gallery, turntable, report."""
    from eu4unitgen.ref3d import unitstage
    fin = folder / "finished"
    m = json.loads((fin / "manifest.json").read_text(encoding="utf-8"))
    subj = json.loads((folder / "subject.json").read_text(encoding="utf-8"))
    tier = int(req.get("tier", 1))
    pv = folder / "preview"
    if pv.exists():
        shutil.rmtree(pv)
    sp = unitstage.spec(f"{req['name']}: preview (tier {tier}; vanilla at tier {tier + 1})", tier, None, None,
                        unitstage.unit_names(req["name"]), unitstage.TEST, gun=unitstage.gun_entry(fin, subj))
    st = unitstage.stage(pv, sp, "preview")
    if st["exit"] != 0:
        raise RuntimeError(f"staging the preview failed; see {pv / 'log.txt'}")
    g = unitstage.gallery(pv, "preview", [tier, tier + 1], {tier: req["name"], tier + 1: "vanilla"}, cfg.eu4(),
                          utype="artillery")
    rep = gun_report(m)
    (pv / "quality.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    sheet = result_sheet(folder, req, g, rep)
    try:
        frames = len(unitstage.turntable(pv, "preview", tier, cfg.eu4(), "artillery"))
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        frames = f"failed: {type(exc).__name__}: {exc}"
    return {"sheet": str(sheet), "quality": rep["summary"], "verdict": rep["verdict"], "turntable": frames}


def stage_preview(folder: Path, req: dict, cfg) -> dict:
    """Staged with vanilla at the next tier, the gallery (close, near, map zoom), the seated clearance, the quality
    report and the result sheet."""
    from eu4unitgen.ref3d import quality, unitstage
    if req["kind"] == "gun":
        return stage_preview_gun(folder, req, cfg)
    fin = folder / "finished"
    m = json.loads((fin / "manifest.json").read_text(encoding="utf-8"))
    subj = json.loads((folder / "subject.json").read_text(encoding="utf-8"))
    tier = int(req.get("tier", 1))
    horse = rider = soldier = None
    utype = FOOT_TYPE[req["kind"]]
    if req["kind"] == "horse":
        horse = unitstage.horse_entry(fin, subj)
    elif req["kind"] == "soldier":
        soldier = unitstage.rider_entry(fin, subj)
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
                        unitstage.TEST, soldier=soldier)
    st = unitstage.stage(pv, sp, "preview")
    if st["exit"] != 0:
        raise RuntimeError(f"staging the preview failed; see {pv / 'log.txt'}")
    g = unitstage.gallery(pv, "preview", [tier, tier + 1], {tier: req["name"], tier + 1: "vanilla"}, cfg.eu4(), utype=utype)
    legs = rev = None
    if req["kind"] == "soldier":  # on foot: its stance's own animations, not the rider's
        from eu4unitgen.ref3d import review, trellis
        m = {**m, "deform": trellis.foot_deform(fin, pv / "foot", cfg.blender())}
        rev = review.review(pv, tier, trellis.foot_stance(m), cfg.eu4())  # next to the game's unit of that set
    if req["kind"] in ("rider", "cavalry"):
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
    fg = (soldier_glb(req).parent if req["kind"] == "cavalry" else folder / "trellis") / "input" / "input_fg.png"
    rep = quality.report(m, body_kind(req["kind"]), fg,
                         Path(imgs["game"][view]) if imgs.get("game", {}).get(view) else None,
                         Path(g["images"].get(f"map_three_quarter_{tier}", "")) or None,
                         Path(g["images"].get(f"map_three_quarter_{tier + 1}", "")) or None,
                         Path(imgs["raw"][view]) if imgs.get("raw", {}).get(view) else None, legs,
                         exclude=[Path(m.get("edited_from", fin)) / "parts" / "crops" / f"{k}_mask.png"
                                  for k in ("lance", "shield", "crest")] if req["kind"] != "horse" else None,
                         on_foot=req["kind"] == "soldier", review=rev)
    quality.save(rep, pv)
    sheet = result_sheet(folder, req, g, rep)
    try:  # the Library's rotating preview; the unit is usable without it
        frames = len(unitstage.turntable(pv, "preview", tier, cfg.eu4(), utype))
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        frames = f"failed: {type(exc).__name__}: {exc}"
    return {"sheet": str(sheet), "quality": rep["summary"], "verdict": rep["verdict"], "turntable": frames}


def turntable_frames(folder: Path) -> list[Path]:
    return sorted((Path(folder) / "preview" / "turntable").glob("frame_*.png"))


def gpu_busy(folder: Path) -> str | None:
    """Another live run in a graphics-card stage (two at once run out of video memory): its name, else None."""
    for r in runs():
        if Path(r["folder"]).resolve() == Path(folder).resolve() or r["state"] != "running":
            continue
        if any(v.get("state") == "running" for k, v in ((r["status"].get("stages") or {}).items()) if k in GPU):
            return r["name"]
    return None


def waiting_for(req: dict) -> list[str]:
    """The part runs a unit needs that aren't finished yet (its first stage waits for them): a cavalry run's horse
    and soldier; a library unit's parts (a starter part is fitted into a run of its own on first use)."""
    if req["kind"] == "cavalry":
        names = [req.get("horse"), req.get("soldier")]
    elif req["kind"] == "unit":
        names = sorted({ref.split(":", 1)[1] for s in (req.get("tiers") or {}).values() for ref in s.values()})
    else:
        return []
    return [n for n in names if n and not (status(folder_of(n)) or {}).get("done")]


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
    ref = req.get("image") or (request(folder_of(req["soldier"])).get("image") if req.get("soldier") else None)
    imported = bool((req.get("source") or {}).get("model"))  # its picture is its render
    tiles = [("your 3D model" if imported else "the reference", Image.open(ref).convert("RGB"))] if ref else []
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
    from eu4unitgen.core import config  # and a copy among the result sheets
    sheets = config.load().sheets()
    sheets.mkdir(parents=True, exist_ok=True)
    sheet.save(sheets / f"{req['name']}.png")
    return out


def quality_title(key: str) -> str:
    from eu4unitgen.ref3d.quality import TITLES
    return TITLES.get(key, key)


def stage_import(folder: Path, req: dict, cfg) -> dict:
    from eu4unitgen.ref3d import importer
    return importer.stage_import(folder, req, cfg)


RUN = {"import": stage_import, "cutout": stage_cutout, "trellis": stage_trellis, "build": stage_build, "finish": stage_finish,
       "preview": stage_preview}


# ------------------------------------------------------------------ the run (a detached process)
def ram_short(stage: str, free: float | None = None, top=None) -> dict | None:
    """None when stage ``stage`` has the RAM it needs (``NEED_RAM``), else what to tell the user: {"free", "need",
    "close": [(program, GB), ...]} (the largest programs now)."""
    need = NEED_RAM.get(stage)
    if need is None:
        return None
    if free is None:
        from eu4unitgen.ref3d import memory
        free = memory.ram_gb()[1]
    if free >= need:
        return None
    if top is None:
        from eu4unitgen.core import checks
        top = checks.top_programs(4)
    mine = ("python", "pythonw", "blender", "vmmem", "vmmemwsl", "wsl", "wslhost", "eu4unitgen")
    return {"free": round(free, 1), "need": need,
            "close": [(n, round(gb, 1)) for n, gb in top if n.lower().removesuffix(".exe") not in mine][:3]}


def go(folder: Path) -> dict:
    from eu4unitgen.core import checks, config
    from eu4unitgen.ref3d.runstate import Status as _Status
    folder = Path(folder).resolve()  # every path a stage hands a job is absolute
    req = request(folder)
    cfg = config.load()
    old = status(folder)
    if old and old.get("alive") and old.get("pid") != os.getpid() and old.get("state") == "running":
        raise SystemExit(f"{req['name']} is already running (process {old.get('pid')})")
    stages = (old or {}).get("stages", {})
    plan = stages_of(req["kind"], req)
    for s in stages.values():  # a stage cut off by the stop runs again
        if s.get("state") in ("running", "waiting_eu4", "waiting_units", "waiting_gpu", "waiting_ram"):
            s["state"] = "waiting"
        if not isinstance(s.get("seconds", 0), (int, float)):  # a time recorded wrongly (0.1.0's trellis stage)
            s.pop("seconds")
    st = _Status(folder, {"name": req["name"], "kind": req["kind"], "pid": os.getpid(), "started": (old or {}).get("started", time.time()),
                          "resumed": bool(old), "stages": stages, "error": None, "trace": None, "done": False,
                          "sheet": (old or {}).get("sheet")})
    try:
        for s in plan:
            if (st.d["stages"].get(s) or {}).get("state") == "done":
                continue
            while waiting_for(request(folder)):  # a cavalry unit: its horse and soldier first
                broken = [n for n in waiting_for(request(folder)) if (status(folder_of(n)) or {}).get("state") == "failed"]
                if broken:
                    raise RuntimeError(f"{' and '.join(broken)} stopped with an error: open it in the Library and resume it first")
                st.stage(s, "waiting_units")
                time.sleep(EU4_POLL)
            if s in GPU:
                while gpu_busy(folder):  # one unit on the graphics card at a time; the next waits its turn
                    st.stage(s, "waiting_gpu")
                    time.sleep(EU4_POLL)
                while checks.eu4_running():  # never while EU4 is open
                    st.stage(s, "waiting_eu4")
                    time.sleep(EU4_POLL)
            while (short := ram_short(s)):  # an AI step waits for the RAM it needs, naming what to close
                st.stage(s, "waiting_ram", **short)
                time.sleep(RAM_POLL)
            st.stage(s, "running")
            if req["kind"] == "unit":
                from eu4unitgen.ref3d import assemble
                res = assemble.RUN[s](folder, request(folder), cfg)
            else:
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


def redo(folder: Path, stage: str) -> None:
    """Mark ``stage`` and every stage after it to run again (a pipeline fix: rebuilt from the models already made)."""
    folder = Path(folder)
    st = read_status(folder) or {}
    plan = stages_of(request(folder)["kind"], request(folder))
    if stage not in plan:
        raise ValueError(f"{stage} isn't one of {plan}")
    for s in plan[plan.index(stage):]:
        (st.get("stages") or {}).pop(s, None)
    st.update(done=False, state="stopped", error=None, heartbeat=0, pid=None)  # a run stopped just now isn't "alive"
    _write(folder, st)


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
    from eu4unitgen.core import layout
    layout.use_root_temp()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("go")
    g.add_argument("folder", type=Path)
    a = p.parse_args(argv)
    go(a.folder)


if __name__ == "__main__":
    main()
