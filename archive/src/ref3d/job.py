"""A horse from one image as an app job (SPEC M19 step 1): the GUI's Generate button, a run that outlives
the window, its progress, the final sheet and "Add to mod". Standard library and Pillow only: this module
runs in the app's process (no numpy there; the heavy work is ref3d.auto in a detached process).

  precheck   warnings before a run: not a side view, a busy background, low free RAM (and what holds it)
  start      ``python -m eu4unitgen.ref3d.job go ...`` detached (closing the window doesn't stop it)
  go         auto new (unless done), auto run, the final sheet; ``status.json`` in the subject folder holds
             the stage, each stage's state and time, a heartbeat and any error. Run again, it resumes:
             finished steps are kept (ref3d.auto), a half-read subject is read again
  status     the status file and whether its process still beats
  estimates  each stage's seconds: the median of earlier runs under the same folder, else horse 2's
  stage      "Add to mod": the game version staged in a new ``build/<name>_NN`` test mod (game.stage_test)

usage: python -m eu4unitgen.ref3d.job go <image> --name <subject> --rights "<declaration>" --ai-python <env python>
           [--eu4 <install>] [--blender <exe>] [--root assets_in]
       python -m eu4unitgen.ref3d.job check <image>       (the warnings the app shows before a run)
       python -m eu4unitgen.ref3d.job stage <subject folder>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from PIL import Image, ImageStat

STAGES = ("read", "shells", "fit", "refine", "master", "game")
LABELS = {"read": "Read the image (mask, parts, landmarks)", "shells": "3D shells (TripoSG, 8 seeds)",
          "fit": "Fit the body", "refine": "Refine the fit", "master": "Build the master", "game": "Game version"}
# horse 2's untouched run (M18 step 5): the estimates until this machine has its own runs
DEFAULT_SECONDS = {"read": 112, "shells": 503, "fit": 7018, "refine": 694, "master": 107, "game": 97}
DEFAULT_FIT_WORKERS = 4
BEAT = 15.0  # seconds between heartbeats; a status older than 4 beats is a stopped run
STATUS = "status.json"


# ------------------------------------------------------------------ before a run
def precheck(image: Path) -> list[str]:
    """What may go wrong with this image: each a sentence for the user (empty: looks fine)."""
    out = []
    im = Image.open(image)
    small = im.convert("RGBA")
    small.thumbnail((256, 256))
    w, h = small.size
    alpha = small.getchannel("A")
    if alpha.getextrema()[0] < 128:  # a cut-out: the background is already gone
        fg = alpha.point(lambda a: 255 if a >= 128 else 0)
    else:
        rgb = small.convert("RGB")
        px = rgb.load()
        b = max(2, round(0.03 * min(w, h)))
        # the background from the border: each row's left and right strips, each column's top and bottom ones
        # (a soft gradient or vignette stays background), blended across the image
        med = lambda pts: tuple(statistics.median(px[q][k] for q in pts) for k in range(3))  # noqa: E731
        left = [med([(x, y) for x in range(b)]) for y in range(h)]
        right = [med([(x, y) for x in range(w - b, w)]) for y in range(h)]
        top = [med([(x, y) for y in range(b)]) for x in range(w)]
        bottom = [med([(x, y) for y in range(h - b, h)]) for x in range(w)]
        # clean: the border changes little from pixel to pixel (no props, floor lines or scenery)
        ring = [(x, y) for y in range(h) for x in range(w) if min(x, y, w - 1 - x, h - 1 - y) < b]
        calm = sum(1 for x, y in ring if x + 1 < w and y + 1 < h and
                   sum(abs(px[x, y][k] - px[x + 1, y][k]) + abs(px[x, y][k] - px[x, y + 1][k]) for k in range(3)) < 24)
        calm /= len(ring)
        if calm < 0.75:  # clean AI renders: 89%; a reference sheet with text and props: 31%
            out.append(f"The background doesn't look clean: only {calm:.0%} of the image's border is plain. "
                       "A plain, even background reads best; the horse may be cut out badly.")
        fg = Image.new("L", (w, h))
        fgp = fg.load()
        for y in range(h):
            for x in range(w):
                u, v = x / (w - 1), y / (h - 1)
                bg = [((1 - u) * left[y][k] + u * right[y][k] + (1 - v) * top[x][k] + v * bottom[x][k]) / 2 for k in range(3)]
                fgp[x, y] = 255 if sum(abs(px[x, y][k] - bg[k]) for k in range(3)) >= 60 else 0
    box = fg.getbbox()
    if box is None:
        return out + ["No horse found: the image looks empty."]
    bw, bh = box[2] - box[0], box[3] - box[1]
    if bw < 1.1 * bh:
        out.append(f"This may not be a side view: the horse is {bw / bh:.2f} times as wide as it is tall (a side "
                   "view is about 1.2 to 1.6). The pipeline reads side views; a front or three-quarter view builds badly.")
    edges = sum((box[0] <= 1, box[1] <= 1, box[2] >= w - 1, box[3] >= h - 1))
    if edges >= 2:
        out.append("The horse touches the image's edges: it may be cropped (missing hooves, ears or tail).")
    return out


def ram_check() -> tuple[str | None, dict]:
    """A warning when free RAM is under what the shells step wants, naming the programs holding the most."""
    from eu4unitgen.ref3d import memory
    total, free = memory.ram_gb()
    need = memory.STEP_MIN_GB["shells"]
    info = {"total_gb": round(total, 1), "free_gb": round(free, 1), "need_gb": need,
            "fit_workers": memory.fit_workers(free, 6)}
    if free >= need:
        return None, info
    top = ", ".join(f"{n} ({gb:.1f} GB)" for n, gb in top_programs(5))
    return (f"Only {free:.1f} of {total:.1f} GB RAM is free; the 3D shells want {need:.0f} GB. Close programs first"
            + (f" (the largest: {top})" if top else "") + ". The run stops at a step without enough RAM and resumes later."), info


def top_programs(n: int) -> list[tuple[str, float]]:
    """The ``n`` programs using the most RAM (summed by name), or [] if the system won't say."""
    use: dict[str, float] = {}
    try:
        if sys.platform == "win32":
            r = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True, timeout=20,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in r.stdout.splitlines():
                f = [c.strip('"') for c in line.split('","')]
                if len(f) >= 5:
                    kb = int("".join(ch for ch in f[4] if ch.isdigit()) or 0)
                    use[f[0]] = use.get(f[0], 0) + kb / 2 ** 20
        else:
            r = subprocess.run(["ps", "-eo", "rss=,comm="], capture_output=True, text=True, timeout=20)
            for line in r.stdout.splitlines():
                kb, _, name = line.strip().partition(" ")
                if kb.isdigit():
                    use[name] = use.get(name, 0) + int(kb) / 2 ** 20
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    return sorted(use.items(), key=lambda kv: -kv[1])[:n]


# ------------------------------------------------------------------ the run's state
def read_status(folder: Path) -> dict | None:
    p = Path(folder) / STATUS
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def status(folder: Path) -> dict | None:
    """The status file plus ``alive`` (its process beat recently) and ``state``: running, done, failed,
    stopped (closed or killed: resume it)."""
    st = read_status(folder)
    if st is None:
        return None
    st["alive"] = time.time() - st.get("heartbeat", 0) < 4 * BEAT
    st["state"] = "done" if st.get("done") else "failed" if st.get("error") else "running" if st["alive"] else "stopped"
    return st


def runs(root: Path) -> list[dict]:
    """Every subject under ``root`` started from the app (a status file), newest first."""
    out = []
    for p in Path(root).glob(f"*/{STATUS}"):
        st = status(p.parent)
        if st:
            out.append({**st, "folder": str(p.parent)})
    return sorted(out, key=lambda s: -s.get("started", 0))


def estimates(root: Path, fit_workers: int | None = None) -> dict[str, float]:
    """Each stage's expected seconds: the median of the finished runs under ``root``, else horse 2's.
    The fit scales with its processes (the restarts share them)."""
    seen: dict[str, list[float]] = {k: [] for k in STAGES}
    for p in Path(root).glob("*/auto_run.json"):
        try:
            rep = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for k, v in rep.get("steps", {}).items():
            if k in seen and v.get("seconds"):
                seen[k].append(v["seconds"] * (rep.get("fit_workers", DEFAULT_FIT_WORKERS) if k == "fit" else 1))
        n = p.parent / "auto_new.json"
        if n.is_file():
            try:
                seen["read"].append(json.loads(n.read_text(encoding="utf-8"))["seconds"])
            except (OSError, ValueError, KeyError):
                pass
    est = {k: statistics.median(v) if v else DEFAULT_SECONDS[k] * (DEFAULT_FIT_WORKERS if k == "fit" else 1)
           for k, v in seen.items()}
    est["fit"] /= max(1, fit_workers or DEFAULT_FIT_WORKERS)  # worker-seconds -> seconds
    return est


class _Status:
    """The status file, rewritten whole on each change and every BEAT seconds (the heartbeat)."""

    def __init__(self, folder: Path, base: dict):
        self.path, self.d = Path(folder) / STATUS, base
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.write()
        self.t = threading.Thread(target=self._beat, daemon=True)
        self.t.start()

    def write(self, **kw):
        with self.lock:
            self.d.update(kw)
            self.d["heartbeat"] = time.time()
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.d, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)

    def stage(self, key: str, state: str, **kw):
        s = self.d["stages"].setdefault(key, {})
        if state == "running":
            s["started"] = time.time()
        if state == "done" and "started" in s:
            s.setdefault("seconds", round(time.time() - s["started"], 1))
        s.update(state=state, **kw)
        self.write(stage=key)

    def _beat(self):
        while not self.stop.wait(BEAT):
            self.write()

    def close(self):
        self.stop.set()
        self.t.join()


# ------------------------------------------------------------------ the run itself (the detached process)
def go(image: Path, name: str, rights: str, ai_python: str, eu4: Path, blender: str | None = None,
       root: Path = Path("assets_in")) -> dict:
    from eu4unitgen.ref3d import auto, memory

    folder = Path(root) / name
    old = status(folder)
    if old and old["alive"] and old.get("pid") != os.getpid():
        raise SystemExit(f"{name} is already running (process {old.get('pid')})")
    if folder.exists() and old is None:
        raise SystemExit(f"{folder} exists and wasn't started from the app: pick a new name")
    folder.mkdir(parents=True, exist_ok=True)
    base = {"name": name, "image": str(Path(image).resolve()), "rights": rights, "pid": os.getpid(),
            "started": (old or {}).get("started", time.time()), "resumed": bool(old), "stages": (old or {}).get("stages", {}),
            "error": None, "done": False, "sheet": None,
            "ram": dict(zip(("total_gb", "free_gb"), (round(x, 1) for x in memory.ram_gb())))}
    for s in base["stages"].values():  # a stage cut off by the stop runs again
        if s.get("state") == "running":
            s["state"] = "waiting"
    st = _Status(folder, base)
    try:
        if not (folder / "auto_new.json").is_file():
            for p in folder.iterdir():  # a half-read subject (stopped during the read): read again
                if p.name not in (STATUS, "job.log"):
                    shutil.rmtree(p) if p.is_dir() else p.unlink()
            st.stage("read", "running")
            rep = auto.new(Path(image), name, rights, ai_python, Path(eu4), root=Path(root))
            st.stage("read", "done", seconds=rep["seconds"], side_view=rep["side_view_window"], parts=rep["parts"])
        else:
            st.stage("read", "done")
        st.write(parts=json.loads((folder / "auto_new.json").read_text(encoding="utf-8")).get("parts"))
        auto.run(folder / "subject.json", ai_python, blender=blender,
                 progress=lambda key, state, **kw: st.stage(key, state, **kw))
        st.write(sheet=str(final_sheet(folder)), done=True)
    except BaseException as exc:  # noqa: BLE001 - recorded for the app, then raised
        st.write(error="".join(traceback.format_exception_only(type(exc), exc)).strip(), trace=traceback.format_exc())
        raise
    finally:
        st.close()
    return st.d


def start(image: Path, name: str, rights: str, ai_python: str, eu4: Path, blender: str | None, root: Path,
          cwd: Path) -> subprocess.Popen:
    """``go`` in its own process, detached from the app (it keeps running when the window closes); its
    output goes to ``<root>/<name>/job.log``."""
    folder = Path(cwd) / root / name
    folder.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "eu4unitgen.ref3d.job", "go", str(image), "--name", name, "--rights", rights,
           "--ai-python", ai_python, "--eu4", str(eu4), "--root", str(root)] + (["--blender", str(blender)] if blender else [])
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}
    log = open(folder / "job.log", "a", encoding="utf-8")  # noqa: SIM115 - handed to the child
    kw: dict = {"cwd": str(cwd), "env": env, "stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL}
    if sys.platform == "win32":
        kw["creationflags"] = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                               | subprocess.CREATE_NO_WINDOW)
    else:
        kw["start_new_session"] = True
    try:
        return subprocess.Popen(cmd, **kw)
    finally:
        log.close()


# ------------------------------------------------------------------ results
def latest(folder: Path, part: str) -> Path | None:
    vs = sorted(p for p in (Path(folder) / part).glob("v[0-9][0-9][0-9]") if (p / "manifest.json").is_file())
    return vs[-1] if vs else None


def final_sheet(folder: Path) -> Path:
    """The reference, what was read from it, the master and the game version in one sheet."""
    from eu4unitgen.ref3d import s1
    folder = Path(folder)
    s = json.loads((folder / "subject.json").read_text(encoding="utf-8"))
    new = json.loads((folder / "auto_new.json").read_text(encoding="utf-8"))
    mv = s.get("main_view", "side")
    run_ = json.loads((folder / "auto_run.json").read_text(encoding="utf-8"))
    tiles = [("the reference", Image.open((folder / s["original"]).resolve()), False)]
    ov = folder / "refs" / "clean" / new["labels"] / f"{mv}_labels_overlay.png"
    if ov.is_file():
        tiles.append(("read from the image: parts,\nzones, landmarks", Image.open(ov), False))
    m, g = latest(folder, "master"), latest(folder, "game")
    if m and (m / "assemble" / f"{mv}_lit.png").is_file():
        tiles.append((f"master {m.name}", Image.open(m / "assemble" / f"{mv}_lit.png"), True))
    if g:
        tris = json.loads((g / "manifest.json").read_text(encoding="utf-8")).get("low", {}).get("final", {}).get("tris")
        for view, cap in (("side", f"game version {g.name}" + (f" ({tris:,} tris)" if tris else "")),
                          ("three_quarter", "game version, three-quarter")):
            p = g / f"horseb_preview_{view}_normal.png"
            if p.is_file():
                tiles.append((cap, Image.open(p), True))
    found = ", ".join(k.replace("_", " ") for k, v in (new.get("parts") or {}).items() if v) or "none"
    times = ", ".join(f"{k} {v['seconds'] / 60:.0f} min" for k, v in run_.get("steps", {}).items())
    low = min((v.get("free_ram_gb", {}).get("lowest", 99) for v in run_.get("steps", {}).values()), default=None)
    notes = [f"Parts found: {found}.", f"Run: read {new['seconds'] / 60:.0f} min, {times}"
             + (f"; lowest free RAM {low} GB." if low and low < 99 else ".")]
    return s1.grid_sheet([(f"{s['name']}\nfrom one image", tiles)], f"{s['name']}: the reference and what was built",
                         folder / "final_sheet.png", tile=(420, 380), notes=notes)


def stage(folder: Path, build: Path = Path("build")) -> dict:
    """"Add to mod": the newest game version staged at tier 1 of a new test mod ``build/<name>_NN``
    (tier 2 the vanilla horse for comparison; game.stage_test)."""
    from eu4unitgen.ref3d import game, s1
    folder = Path(folder)
    s = s1.load_subject(folder / "subject.json")
    g = latest(folder, "game")
    if g is None:
        raise SystemExit(f"{s['name']} has no game version yet")
    m = json.loads((g / "manifest.json").read_text(encoding="utf-8"))
    k = 1
    while (build / f"{s['name']}_{k:02d}").exists():
        k += 1
    horse = game.horse_entry(s, Path(m.get("geometry") or g / "low_geometry.json"), m["textures"])
    return {"game": g.name, **game.stage_test(f"{s['name']}_{k:02d}", [horse], game.GAME_DEFAULTS)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("go")
    a.add_argument("image", type=Path)
    a.add_argument("--name", required=True)
    a.add_argument("--rights", required=True)
    a.add_argument("--ai-python", required=True)
    a.add_argument("--eu4", type=Path, default=os.environ.get("EU4_PATH"))
    a.add_argument("--blender")
    a.add_argument("--root", type=Path, default=Path("assets_in"))
    c = sub.add_parser("check")
    c.add_argument("image", type=Path)
    d = sub.add_parser("stage")
    d.add_argument("folder", type=Path)
    args = p.parse_args(argv)
    if args.cmd == "check":
        print(json.dumps({"image": precheck(args.image), "ram": ram_check()}, indent=1))
        return
    if args.cmd == "stage":
        print(json.dumps(stage(args.folder)))
        return
    if not args.eu4:
        raise SystemExit("give --eu4 or set EU4_PATH (the vanilla horse's skeleton is read from it)")
    go(args.image, args.name, args.rights, args.ai_python, args.eu4, args.blender, args.root)


if __name__ == "__main__":
    main()
