"""The standard horse pipeline (SPEC M21): one image -> TRELLIS.2 -> processing -> the EU4 game version.

  run      the whole chain for one image, into ``assets_in/<name>/trellis/``:
             1. cut-out: the horse masked by BiRefNet (ai job ref3d_mask, the Windows AI environment), RGBA;
             2. TRELLIS.2 in WSL2 (ai job trellis2_generate in ``~/eu4unitgen_ai/trellis2_env``, built by
                setup_scripts/trellis2_setup.sh; the weights read from the AI root through /mnt): ``raw.glb``;
             3. ``process`` below;
           and with ``--test`` the game version staged alone (tier 1, vanilla next).
  process  a TRELLIS.2 .glb into ``<out>/vNNN``:
             the base colour pulled out of the .glb; fresh_horse's ``direct`` source (shape kept: turned to
             face -Z, one scale to vanilla's seat, slimmed about the seat when bulkier than vanilla, lower legs
             under vanilla's heels, the hair (our tail, the mane smoothed), a closed solid, quadric collapse to
             ~2,800, region UVs); the raw model rendered lit from its side (render_model, EEVEE) as the colour
             target; bake_colour (base colour from the raw model, tail paint, head band and marks, optional
             emblem decal and country-colour mask, gloss by colour class, the colour matched to the raw render,
             silver held under the raw's); bake_horse (normal and AO from the raw model); the comparison sheet
             (raw against game, one camera, overlay); the deformation check against vanilla.
  stage    game versions as tiers 1, 2, ... of a new test (vanilla next), with unit names and, when the first
           has a country-colour mask, its cloth's red as the test tag's primary colour.

usage: python -m eu4unitgen.ref3d.trellis run <image> --name <subject> --rights "<declaration>" --ai-python <python>
                                              [--emblem <decal.png>] [--country-mask] [--test testNN]
       python -m eu4unitgen.ref3d.trellis process <glb> --out <dir> [--emblem <png>] [--country-mask]
       python -m eu4unitgen.ref3d.trellis stage --test testNN --names T28 <version dir>@<subject.json> [...]
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import shlex
import struct
import subprocess
import time
from pathlib import Path, PureWindowsPath

# TRELLIS.2 in WSL (SPEC M21): the environment on the Linux disk, the checkout and weights on the AI root via /mnt
WSL = {"python": "~/eu4unitgen_ai/trellis2_env/bin/python", "repo": "TRELLIS.2", "weights": "weights/TRELLIS.2-4B",
       "ss_decoder": "weights/TRELLIS-image-large", "dinov3": "weights/dinov3-vitl16-pretrain-lvd1689m",
       "seed": 42, "pipeline_type": "1024_cascade"}
# a masked horse's secondary and tertiary unit colours (its primary is its cloth's red): gold and off-white
BYZ_GOLD, BYZ_WHITE = (212, 175, 55), (230, 225, 210)
PROCESS = {"target_tris": 2800, "smooth": 12, "influences": 3, "chain_weights": True, "uv_margin": 0.002,
           "hair": {"mane": True, "tail": True}, "size": 1024,
           "gloss": {"silver": 14, "gold": 12, "matte": 5},
           "bake": {"cage": 0.3, "distance": 0.6, "margin": 8},
           "normal": {"cage": 0.1, "distance": 0.35, "high_smooth": 0, "strength": 0.6, "samples": 32, "ao_strength": 0.35}}


def next_version(folder: Path) -> Path:
    """The next ``vNNN`` folder under ``folder`` (made)."""
    n = 1
    while (folder / f"v{n:03d}").exists():
        n += 1
    out = folder / f"v{n:03d}"
    out.mkdir(parents=True)
    return out


def ai_paths(ai_python: str | None = None) -> dict:
    """The model folders under the AI root (the settings', else the AI environment's)."""
    from eu4unitgen.core import config
    from eu4unitgen.ref3d.writeguard import ai_root
    root = config.load().ai_root_path() if ai_python is None else ai_root(ai_python)
    if root is None:
        raise SystemExit("the AI environment isn't set (Settings, or Setup)")
    return {"root": root, "u2net": os.environ.get("U2NET_HOME") or str(root / "rembg_models"),
            "sam2": str(root / "weights" / "sam2.1-hiera-large")}


def wsl_path(p: Path) -> str:
    """A Windows path as WSL sees it (C:\\x -> /mnt/c/x)."""
    w = PureWindowsPath(p)
    if not w.drive:
        w = PureWindowsPath(os.path.abspath(p))
    return "/mnt/" + w.drive[0].lower() + w.as_posix()[2:]


def sh_python(python: str) -> str:
    """The WSL python for a bash command line, quoted (a leading ``~/`` stays the home folder)."""
    return '"$HOME"/' + shlex.quote(python[2:]) if python.startswith("~/") else shlex.quote(python)


def wsl_job_cmd(python: str, job: Path, args_path: Path, log: Path) -> str:
    """The bash command that runs one TRELLIS.2 job in WSL, every path quoted (the root may have spaces in it)."""
    return (f"cd /tmp && {sh_python(python)} {shlex.quote(wsl_path(job))} {shlex.quote(wsl_path(args_path))} "
            f"> {shlex.quote(wsl_path(log))} 2>&1")


def eu4() -> Path:
    from eu4unitgen.core import config
    p = config.load().eu4()
    if p is None:
        raise SystemExit("set the EU4 install in Settings (or EU4_PATH): the vanilla skeletons are read from it")
    return p


def cutout(image: Path, out: Path, ai_python: str) -> Path:
    """The horse cut out (BiRefNet: ai job ref3d_mask) as an RGBA png: TRELLIS.2 takes the alpha as is."""
    from PIL import Image

    from eu4unitgen.ai.env import AIJobRunner
    out.mkdir(parents=True, exist_ok=True)
    rgb = out / "input_rgb.png"
    Image.open(image).convert("RGB").save(rgb)
    r = AIJobRunner(ai_python, env={"U2NET_HOME": ai_paths(ai_python)["u2net"]}).run_job(
        "ref3d_mask", {"images": {"input": str(rgb)}, "out_dir": str(out)}).result
    rgba = Image.open(rgb).convert("RGBA")
    rgba.putalpha(Image.open(r["input"]["path"]).convert("L"))
    path = out / "input_rgba.png"
    rgba.save(path)
    return path


def generate(rgba: Path, glb: Path, ai_root: Path, wsl: dict | None = None) -> dict:
    """TRELLIS.2 on one RGBA image in WSL2 -> ``glb``; its result (seconds, peak VRAM, faces)."""
    from eu4unitgen.core import config
    cfg = config.load()
    w = {**WSL, "python": cfg.wsl_python or WSL["python"], **(wsl or {})}
    job = Path(__file__).resolve().parents[1] / "ai" / "jobs" / "trellis2_generate.py"
    result = glb.with_suffix(".result.json")
    args = {"image": wsl_path(rgba), "output": wsl_path(glb), "result_path": wsl_path(result),
            "repo": wsl_path(ai_root / w["repo"]), "weights": wsl_path(ai_root / w["weights"]),
            "ss_decoder": wsl_path(ai_root / w["ss_decoder"]), "dinov3": wsl_path(ai_root / w["dinov3"]),
            "seed": w["seed"], "pipeline_type": w["pipeline_type"]}
    args_path = glb.with_suffix(".args.json")
    args_path.write_text(json.dumps(args, indent=1), encoding="utf-8")
    log = glb.with_suffix(".log")
    cmd = wsl_job_cmd(w["python"], job, args_path, log)
    rc = subprocess.run(cfg.wsl_base() + ["-e", "bash", "-lc", cmd]).returncode
    if rc != 0 or not result.is_file():
        raise SystemExit(f"TRELLIS.2 failed (exit {rc}); see {log}")
    return json.loads(result.read_text(encoding="utf-8"))


SEEDS = (42, 43)  # two TRELLIS.2 candidates per subject (the user's decision); the better outline is built


def generate_candidates(rgba: Path, out_dir: Path, ai_root: Path, seeds=SEEDS) -> list[dict]:
    """TRELLIS.2 on one image with several seeds in one model load -> [{"seed", "glb", ...}] (``raw_s<seed>.glb``)."""
    from eu4unitgen.core import config
    cfg = config.load()
    w = {**WSL, "python": cfg.wsl_python or WSL["python"]}
    job = Path(__file__).resolve().parents[1] / "ai" / "jobs" / "trellis2_generate.py"
    out_dir = Path(out_dir).resolve()
    result = out_dir / "candidates.result.json"
    items = [{"image": wsl_path(Path(rgba).resolve()), "output": wsl_path(out_dir / f"raw_s{s}.glb"), "seed": s} for s in seeds]
    args = {"batch": items, "result_path": wsl_path(result), "repo": wsl_path(ai_root / w["repo"]),
            "weights": wsl_path(ai_root / w["weights"]), "ss_decoder": wsl_path(ai_root / w["ss_decoder"]),
            "dinov3": wsl_path(ai_root / w["dinov3"]), "pipeline_type": w["pipeline_type"]}
    args_path = out_dir / "candidates.args.json"
    args_path.write_text(json.dumps(args, indent=1), encoding="utf-8")
    log = out_dir / "candidates.log"
    cmd = wsl_job_cmd(w["python"], job, args_path, log)
    rc = subprocess.run(cfg.wsl_base() + ["-e", "bash", "-lc", cmd]).returncode
    if rc != 0 or not result.is_file():
        raise SystemExit(f"TRELLIS.2 failed (exit {rc}); see {log}")
    r = json.loads(result.read_text(encoding="utf-8"))
    return [{"seed": s, "glb": str(out_dir / f"raw_s{s}.glb"), **{k: it.get(k) for k in ("seconds", "faces", "peak_vram_gb")}}
            for s, it in zip(seeds, r["items"])]


def score_candidate(br, glb: Path, reference_fg: Path, out: Path, views: int = 12) -> dict:
    """How well a raw model's outline matches the reference: rendered every 360/``views`` degrees round the
    vertical (TRELLIS.2 picks its own facing), the best overlap with the reference's mask (mirror-free: quality's
    silhouette_iou also tries the mirror). -> {"iou", "azimuth"}."""
    from PIL import Image

    from eu4unitgen.ref3d.quality import silhouette_iou
    r = br.run_job("render_model", {"model": str(Path(glb).resolve()), "out_dir": str(Path(out).resolve()), "name": "cand",
                                    "size": 320, "views": {f"a{a}": [a, 0] for a in range(0, 360, 360 // views)}}).result
    fg = Image.open(reference_fg)
    best = max(((silhouette_iou(Image.open(p), fg), v) for v, p in r["images"].items()), key=lambda t: t[0])
    return {"iou": best[0], "azimuth": int(best[1][1:])}
# How the two candidates are weighed (the user, 2026-10-28: outline, colour AND deformation, not outline alone):
# the outline's overlap, less 0.01 per unit of colour distance (Lab ΔE, median) to the reference, less 0.05 per unit
# the worst stretch goes past vanilla's own worst (a stretch of 1 is none; vanilla's horse reaches about 1.6).
PICK = {"colour_per_de": 0.01, "stretch_per_unit": 0.05}


def colour_distance(model, reference_rgba, step: int = 4) -> float | None:
    """The median Lab distance between a textured render and the reference where both show the subject, each
    framed by its box (the better of as-is and mirrored, like the outline). None: no overlap."""
    from PIL import Image, ImageChops

    from eu4unitgen.core import palette as pal
    from eu4unitgen.ref3d.part_gen import _framed

    def subject(im):  # no transparency (a render on its background): the subject is what isn't the corner's colour
        im = im.convert("RGBA")
        if im.getchannel("A").getextrema()[0] > 250:
            bg = Image.new("RGB", im.size, im.getpixel((2, 2))[:3])
            im.putalpha(ImageChops.difference(im.convert("RGB"), bg).convert("L").point(lambda v: 255 if v > 18 else 0))
        return im
    a, b = _framed(subject(model)), _framed(subject(reference_rgba))
    best = None
    for m in (a, a.transpose(Image.FLIP_LEFT_RIGHT)):
        pm, pr = m.load(), b.load()
        des = sorted(pal.delta_e(pm[x, y][:3], pr[x, y][:3]) for y in range(0, m.height, step) for x in range(0, m.width, step)
                     if pm[x, y][3] > 127 and pr[x, y][3] > 127)
        if des and (best is None or len(des) > best[0]):  # the orientation that overlaps more
            best = (len(des), des[len(des) // 2])
    return round(best[1], 1) if best else None


def score_colour(br, glb: Path, reference_rgba: Path, out: Path, azimuth: int) -> float | None:
    """The raw model with its own texture (EEVEE) from the outline's best side, against the reference's colours."""
    from PIL import Image
    r = br.run_job("render_model", {"model": str(Path(glb).resolve()), "engine": "EEVEE", "out_dir": str(Path(out).resolve()),
                                    "name": "colour", "size": 320, "views": {"best": [azimuth, 0]}}).result
    return colour_distance(Image.open(r["images"]["best"]), Image.open(reference_rgba))


def trial_deform(glb: Path, kind: str, out: Path, blender: str | None = None) -> dict:
    """How a candidate deforms once fitted to the game skeleton, before the full build: only the shape and weights
    (fresh_horse / fresh_rider; no bakes, no parts) and the deformation check against vanilla's.
    -> {"worst_stretch", "vanilla_worst", "flips": {anim: n}}."""
    from eu4unitgen.blender.runner import BlenderRunner
    from eu4unitgen.core.fresh_mesh import base_arrays
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    br = BlenderRunner(blender or _config().blender(), log_dir=out / "logs", timeout=1800)
    src = {"path": str(Path(glb).resolve()), "merge": 1e-5}
    if kind == "horse":
        base = (eu4() / "gfx" / "models" / "Units" / "cavalry_horse.mesh").read_bytes()
        br.run_job("fresh_horse", {"source": {"kind": "direct", **src},
                                   **{k: PROCESS[k] for k in ("target_tris", "smooth", "influences", "chain_weights", "uv_margin", "hair")},
                                   "reins": PROCESS.get("reins", True), "vanilla": base_arrays(base),
                                   "geometry_out": str(out / "low_geometry.json"), "preview_out": str(out / "low.obj"),
                                   "fitted_out": str(out / "fitted.obj"), "textured_out": str(out / "textured.obj")})
        d = deform(br, out, base)
    else:
        base_file = eu4() / "gfx" / "models" / "Units" / "western_generic_01_type1.mesh"
        base = base_file.read_bytes()
        anim = eu4() / "gfx" / "models" / "Units" / "animation" / "cavalry_rider_idle.anim"
        frame = br.run_job("rider_frames", {"mesh": str(base_file), "anim": str(anim), "node": "Left_hand_node",
                                            "elbow": "L_elbow", "shoulder": "L_shoulder",
                                            "bind": node_bind(base, "Left_hand_node")}).result
        (out / "parts").mkdir(exist_ok=True)
        br.run_job("fresh_rider", {"vanilla": base_arrays(base), "source": src, **RIDER, "shield_frame": frame, "plume": "part",
                                   "geometry_out": str(out / "low_geometry.json"), "preview_out": str(out / "low.obj"),
                                   "fitted_out": str(out / "fitted.obj"), "textured_out": str(out / "textured.obj"),
                                   "parts_dir": str(out / "parts")})
        pin_head(out / "low_geometry.json")
        d = rider_deform(br, out, base, base_file, renders=False)
    ours, van = d.get("ours") or {}, d.get("vanilla") or {}
    return {"worst_stretch": max((a["max"] for a in ours.values()), default=None),
            "vanilla_worst": max((a["max"] for a in van.values()), default=None),
            "flips": {k: a["flips"] for k, a in ours.items()}}


def pick_score(c: dict) -> float:
    """One number per candidate from its outline, colour and worst stretch (``PICK``); a missing measure costs
    nothing (so an old candidate scored by outline alone still compares)."""
    s = float(c.get("iou") or 0.0)
    if c.get("colour_de") is not None:
        s -= PICK["colour_per_de"] * float(c["colour_de"])
    if c.get("worst_stretch") is not None:
        s -= PICK["stretch_per_unit"] * max(0.0, float(c["worst_stretch"]) - float(c.get("vanilla_worst") or 1.0))
    return round(s, 4)


def glb_metallic(glb: Path, out: Path) -> Path | None:
    """The metallicRoughness image of a .glb's first material (blue: metallic), as a png; None without one."""
    from PIL import Image
    data = Path(glb).read_bytes()
    off, chunks = 12, {}
    while off < len(data):
        length, kind = struct.unpack_from("<II", data, off)
        chunks[kind] = data[off + 8: off + 8 + length]
        off += 8 + length
    doc, binary = json.loads(chunks[0x4E4F534A]), chunks[0x004E4942]
    ref = (doc.get("materials") or [{}])[0].get("pbrMetallicRoughness", {}).get("metallicRoughnessTexture")
    if not ref:
        return None
    bv = doc["bufferViews"][doc["images"][doc["textures"][ref["index"]]["source"]]["bufferView"]]
    Image.open(io.BytesIO(binary[bv.get("byteOffset", 0): bv.get("byteOffset", 0) + bv["byteLength"]])).save(out)
    return out


def glb_basecolor(glb: Path, out: Path) -> Path:
    """The base-colour image of a .glb's first material (TRELLIS.2 writes one), as a png."""
    from PIL import Image
    data = Path(glb).read_bytes()
    off, chunks = 12, {}
    while off < len(data):
        length, kind = struct.unpack_from("<II", data, off)
        chunks[kind] = data[off + 8: off + 8 + length]
        off += 8 + length
    doc, binary = json.loads(chunks[0x4E4F534A]), chunks[0x004E4942]
    ref = doc["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]
    bv = doc["bufferViews"][doc["images"][doc["textures"][ref["index"]]["source"]]["bufferView"]]
    im = Image.open(io.BytesIO(binary[bv.get("byteOffset", 0): bv.get("byteOffset", 0) + bv["byteLength"]]))
    im.save(out)
    return out


def process(glb: Path, root: Path, emblem: Path | None = None, country_mask: bool = False, blender: str | None = None,
            cfg: dict | None = None) -> dict:
    """``glb`` (TRELLIS.2) into ``root/vNNN`` (module doc); ``cfg`` overrides ``PROCESS`` (e.g. a detail budget's
    ``target_tris``). -> the manifest."""
    from eu4unitgen.blender.runner import BlenderRunner
    from eu4unitgen.core.fresh_mesh import base_arrays
    out = next_version(Path(root)).resolve()
    br = BlenderRunner(blender or _config().blender(), log_dir=out / "logs", timeout=3600)
    base = (eu4() / "gfx" / "models" / "Units" / "cavalry_horse.mesh").read_bytes()
    van = base_arrays(base)
    t0, P = time.time(), {**PROCESS, **(cfg or {})}
    basecolor = glb_basecolor(glb, out / "raw_basecolor.png")
    fr = br.run_job("fresh_horse", {
        "source": {"kind": "direct", "path": str(Path(glb).resolve()), "merge": 1e-5},
        **{k: P[k] for k in ("target_tris", "smooth", "influences", "chain_weights", "uv_margin", "hair")},
        "reins": P.get("reins", True),
        "vanilla": van, "geometry_out": str(out / "low_geometry.json"), "preview_out": str(out / "low.obj"),
        "fitted_out": str(out / "fitted.obj"), "textured_out": str(out / "textured.obj")}).result
    (out / "fresh.json").write_text(json.dumps(fr, indent=1, default=str), encoding="utf-8")
    # the raw model lit, from the side the fit turned to the game's left: the colour target
    az = 90.0 - float(fr["fit"].get("turned_deg", 0.0))
    raw = br.run_job("render_model", {"model": str(Path(glb).resolve()), "engine": "EEVEE", "out_dir": str(out / "raw"),
                                      "name": "raw", "size": 900, "views": {"side": [az % 360, 5]}}).result
    head = fr["pose"]["head"]
    poll, axis = head["poll"], head["axis"]
    geo = json.loads((out / "low_geometry.json").read_text(encoding="utf-8"))
    pp, length = geo["p"], 0.0
    for i in range(0, len(pp), 3):  # the head's length: the furthest point along its axis near it
        rel = [pp[i + k] - poll[k] for k in range(3)]
        t = sum(r * a for r, a in zip(rel, axis))
        if t > 0 and math.sqrt(max(0.0, sum(r * r for r in rel) - t * t)) < 2.5:
            length = max(length, t)
    hair = fr.get("hair") or {}
    args = {"low": str(out / "low_geometry.json"), "high": str(out / "textured.obj"), "image": str(basecolor),
            "part_colours": {**({"6": hair["tail_colour_p25_p75"]} if hair.get("tube") else {}),
                             **({"7": (fr.get("reins") or {})["colour_p25_p75"]} if (fr.get("reins") or {}).get("built") else {})},
            "gloss": P["gloss"], "head_band": {"poll": poll, "axis": axis},
            "head_marks": {"poll": poll, "axis": axis, "neck_base": van["bones"]["HorseNeckBase"], "length": length,
                           "ear_box": [[poll[0] - 1.2, poll[1] - 0.3, poll[2] - 1.2], [poll[0] + 1.2, poll[1] + 3.0, poll[2] + 1.2]]}
            if P.get("tassels", True) else None,
            "bones": van["bones"], "country_mask": bool(country_mask), "colour_render": raw["images"]["side"],
            "size": P["size"], **P["bake"], "out_dir": str(out), "name": "horse"}
    # horses keep the colour rule for steel: their metallic maps call the coat metal (horse 2: 59%) and lose the
    # bit's and buckles' gloss (regression 2026-10-29); the metallic map is for people's pale cloth (bake_colour)
    if emblem:
        args["emblem"] = {"decal": str(Path(emblem).resolve()), "size": 0.7}
    bc = br.run_job("bake_colour", args).result
    nb = br.run_job("bake_horse", {"low": str(out / "low_geometry.json"), "high": [str(out / "textured.obj")],
                                   "diffuse": bc["diffuse"], "size": P["size"], **P["normal"], "preview": False,
                                   "out_dir": str(out), "name": "horseb"}).result
    tex = {"diffuse": str(out / "horseb_diffuse.png"), "specular": bc["specular"], "normal": str(out / "horseb_normal.png")}
    manifest = {"stage": "game version (M21 standard horse pipeline)", "glb": str(glb), "version": out.name,
                "fresh": {k: fr.get(k) for k in ("fit", "pose", "hair", "reins", "final", "back", "uv", "stages")},
                "colour": bc, "normal": {k: nb.get(k) for k in ("tilt", "match")}, "textures": tex,
                "geometry": str(out / "low_geometry.json"), "seconds": round(time.time() - t0, 1)}
    manifest["compare"] = compare_sheet(br, out, basecolor, tinted(Path(tex["diffuse"]), Path(bc["specular"]),
                                                                    bc.get("default_primary"), out / "game_tinted.png"))
    manifest["deform"] = deform(br, out, base)
    manifest["seconds"] = round(time.time() - t0, 1)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    return manifest


GUN = {"target_tris": 2400, "budget": 2500, "uv_margin": 0.003, "size": 1024}  # the base game's gun: 682; at 1,500 the trail tore


def gun_base() -> dict:
    """The base game's gun (``artillery.mesh``, read at run time): its bones, its cannon_muzzle locator in world
    space (the locator's p is relative to its parent bone, Cannon) and its extent."""
    from eu4unitgen.core.fresh_mesh import base_arrays
    from eu4unitgen.library.scanner import parse_pdx_mesh
    data = (eu4() / "gfx" / "models" / "Units" / "artillery.mesh").read_bytes()
    van = base_arrays(data)
    tree = parse_pdx_mesh(data)
    loc = next((c for c in tree["children"] if c["name"] == "locator"), {"children": []})
    mz = next((c for c in loc["children"] if c["name"] == "cannon_muzzle"), None)
    parent = van["bones"][mz["props"]["pa"][0]] if mz else [0.0, 0.0, 0.0]
    muzzle = [mz["props"]["p"][k] + parent[k] for k in range(3)] if mz else None
    p = van["p"]
    lo = [min(p[k::3]) for k in range(3)]
    hi = [max(p[k::3]) for k in range(3)]
    return {"bones": van["bones"], "muzzle": muzzle, "extent": [lo, hi], "bytes": data}


def process_gun(glb: Path, root: Path, blender: str | None = None, cfg: dict | None = None) -> dict:
    """Track A: a TRELLIS.2 gun (``glb``) into ``root/vNNN``: the rigid model on the base game's gun skeleton
    (job ``fresh_gun``), its colours baked from the raw model, a normal map, the raw-against-game comparison.
    -> the manifest."""
    from eu4unitgen.blender.runner import BlenderRunner
    out = next_version(Path(root)).resolve()
    br = BlenderRunner(blender or _config().blender(), log_dir=out / "logs", timeout=3600)
    P = {**GUN, **(cfg or {})}
    t0 = time.time()
    base = gun_base()
    basecolor = glb_basecolor(glb, out / "raw_basecolor.png")
    fr = br.run_job("fresh_gun", {"source": str(Path(glb).resolve()),
                                  "vanilla": {k: base[k] for k in ("bones", "muzzle", "extent")},
                                  "target_tris": P["target_tris"], "uv_margin": P["uv_margin"],
                                  "geometry_out": str(out / "low_geometry.json"), "preview_out": str(out / "low.obj"),
                                  "textured_out": str(out / "fitted.obj")}).result
    (out / "fresh.json").write_text(json.dumps(fr, indent=1, default=str), encoding="utf-8")
    raw = br.run_job("render_model", {"model": str(Path(glb).resolve()), "engine": "EEVEE", "out_dir": str(out / "raw"),
                                      "name": "raw", "size": 900, "views": {"side": [90, 5]}}).result
    bc = br.run_job("bake_colour", {"low": str(out / "low_geometry.json"), "high": str(out / "fitted.obj"),
                                    "image": str(basecolor), "part_colours": {}, "gloss": PROCESS["gloss"],
                                    "bones": base["bones"], "country_mask": False, "colour_render": raw["images"]["side"],
                                    # the brightness only: the raw render's hues are dull and the transfer greyed the
                                    # bombard's wood and bronze
                                    "colour_strength": (1.0, 0.0, 0.0),
                                    "size": P["size"], **PROCESS["bake"], "out_dir": str(out), "name": "gun",
                                    **({"metallic_image": str(m_)} if (m_ := glb_metallic(glb, out / "raw_metallic.png"))
                                       else {})}).result
    nb = br.run_job("bake_horse", {"low": str(out / "low_geometry.json"), "high": [str(out / "fitted.obj")],
                                   "diffuse": bc["diffuse"], "size": P["size"], **PROCESS["normal"], "preview": False,
                                   "out_dir": str(out), "name": "horseb"}).result
    tex = {"diffuse": str(out / "horseb_diffuse.png"), "specular": bc["specular"], "normal": str(out / "horseb_normal.png")}
    manifest = {"stage": "gun game version (Track A)", "glb": str(glb), "version": out.name, "fresh": fr, "colour": bc,
                "normal": {k: nb.get(k) for k in ("tilt", "match")}, "textures": tex,
                "geometry": str(out / "low_geometry.json"), "budget": P["budget"]}
    manifest["compare"] = compare_sheet(br, out, basecolor)
    manifest["seconds"] = round(time.time() - t0, 1)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    return manifest


# the rider (M23, test 30 round): slimmed to at least 0.75 of its width towards vanilla's rider, the low mesh evened
# (3 passes) and its weights smoothed 24 times: flips idle 30 / moving 29 / attack 41 (12 / 0 / 12 for vanilla's)
RIDER = {"target_tris": 2600, "part_tris": {"lance": 200, "shield": 400}, "close_voxel": 0.1, "slim_low": 0.75,
         "weight_smooth": 24, "even": 3}
# the standing fit (infantry and the gun crew; the user, T37/T38): slimmed further towards vanilla's infantry (the
# rider build stood bulky), and more triangles for the hands and boots (the English soldier's came out as blobs and
# spikes at 2,600; the budget is the fresh_mesh rule's 3,000). With the joint fit its weights aren't smoothed after the
# region rules (the share of his surface squashed past 0.8 in polearm_twohanded_attack, same frames: the English
# soldier 25.7% smoothed 12 times, 20.3% at 3, 18.6% at none; the cataphract in spear_shield 26.7 / 21.0 / 20.3;
# vanilla's body 10.5)
STANDING = {"slim_low": 0.6, "target_tris": 2950, "weight_smooth": 0}
# the sword in its scabbard (the user, test 30: decoration, never drawn): in the reference's colours; hung at 65 degrees
# from vertical for a rider, 25 for a man standing (at 65 the English soldier's stuck out behind him like a pole)
SWORD = {"hang_deg": 65.0, "length": 6.5, "grip": 1.3, "guard": 1.4, "out": 1.0, "up": 0.2, "forward": 0.6}  # out 0.35: inside the skirt
SWORD_COLOURS = {"gold": (200, 160, 70), "grip": (60, 38, 26), "scabbard": (150, 38, 32), "steel": (150, 150, 152)}  # lit, 105 read black
SWORD_GLOSS = {"gold": 12, "grip": 4, "scabbard": 4, "steel": 14}


def sword_colours(pal: dict) -> dict:
    """The sword in the soldier's own colours (the cataphract's gold and red were fixed; the English soldier's are
    steel and leather): the hilt fittings his gold if he has any, else his steel; the scabbard his darkest brown;
    the blade's steel his silver. -> SWORD_COLOURS' keys."""
    from eu4unitgen.core import palette as palmod
    out = {}
    silver = (pal.get("silver") or {}).get("colour") if (pal.get("silver") or {}).get("share", 0) > 0.01 else None
    if silver:
        out["steel"] = tuple(int(v) for v in silver)
        if (pal.get("gold") or {}).get("share", 0) <= 0.005:
            out["gold"] = tuple(int(v * 0.85) for v in silver)
    browns = [m["colour"] for m in pal.get("matte") or []
              if 10 <= palmod.hue(m["colour"]) <= 45 and palmod.lum(m["colour"]) < 120]
    # his darkest brown, else plain leather (the fixed red is the cataphract's: on the English soldier it read red)
    out["scabbard"] = tuple(int(v) for v in min(browns, key=palmod.lum)) if browns else (88, 58, 38)
    out["grip"] = tuple(int(v * 0.7) for v in out["scabbard"])
    return out


def sword_part(out: Path, frame: dict, hip_half_width: float, cfg: dict | None = None, gold=None,
               colours: dict | None = None) -> dict:
    """M23: a sword in its scabbard as a rigid belt part (``mesh_belt``, the pelvis bone): our own geometry, plain
    boxes (pommel, grip, crossguard, a tapered scabbard with a gold locket and chape) on a four-swatch texture in
    the reference's colours (gold hilt, dark grip, dark red scabbard). Placed with ``frame`` (rider_frames on
    ``hip``, the seated pose: the part's own PDX directions for out, up and forward): the hilt at the hip's side
    (``hip_half_width`` + out), a little forward and up; the scabbard hanging down and back at ``hang_deg`` from
    vertical, its flat facing outward. -> a loadout part entry (files under ``out``)."""
    import math

    from PIL import Image
    c = {**SWORD, **(cfg or {})}
    norm = lambda v: [x / math.sqrt(sum(y * y for y in v)) for x in v]  # noqa: E731
    cross = lambda u, v: [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]  # noqa: E731
    o, up, fwd = norm(frame["out"]), norm(frame["up"]), norm(frame["forward"])
    th = math.radians(c["hang_deg"])
    ax = norm([-up[k] * math.cos(th) - fwd[k] * math.sin(th) for k in range(3)])  # down and back
    w = norm(cross(o, ax))  # the flat's width (in the plane facing outward)
    t = norm(cross(ax, w))  # its thickness (outward)
    hilt = [o[k] * (hip_half_width + c["out"]) + up[k] * c["up"] + fwd[k] * c["forward"] for k in range(3)]
    swatch = {"gold": (0, 0), "grip": (1, 0), "scabbard": (0, 1), "steel": (1, 1)}
    p, tri, uv = [], [], []

    def box(s0, s1, w0, w1, t0, zone):
        """A box from s0 to s1 along the sword (half widths w0 -> w1, half thickness t0), one swatch."""
        corners = []
        for s, hw in ((s0, w0), (s1, w1)):
            for a_, b_ in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                corners.append([hilt[k] + ax[k] * s + w[k] * hw * a_ + t[k] * t0 * b_ for k in range(3)])
        u0, v0 = swatch[zone]
        quad_uv = [(u0 * 0.5 + 0.05, v0 * 0.5 + 0.05), (u0 * 0.5 + 0.45, v0 * 0.5 + 0.05),
                   (u0 * 0.5 + 0.45, v0 * 0.5 + 0.45), (u0 * 0.5 + 0.05, v0 * 0.5 + 0.45)]
        for f in ((0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)):
            base = len(p) // 3
            for i in f:
                p.extend(corners[i])
            for q in quad_uv:
                uv.extend(q)
            tri.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    L, g = c["length"], c["grip"]
    box(-g - 0.35, -g, 0.24, 0.24, 0.22, "gold")  # pommel
    box(-g, -0.1, 0.13, 0.13, 0.12, "grip")
    box(-0.1, 0.15, c["guard"] / 2, c["guard"] / 2, 0.15, "gold")  # crossguard
    box(0.15, 0.6, 0.3, 0.29, 0.15, "gold")  # locket
    box(0.6, L - 0.6, 0.29, 0.2, 0.13, "scabbard")
    box(L - 0.6, L, 0.2, 0.12, 0.12, "gold")  # chape
    out.mkdir(parents=True, exist_ok=True)
    (out / "sword_part.json").write_text(json.dumps({"p": [round(x, 5) for x in p], "tri": tri, "uv": uv}), encoding="utf-8")
    # Track P1: no flat-colour placeholders: each swatch worn (grime and grain) with a normal map from the grain;
    # the gold the unit's own (``gold``: the palette's)
    from eu4unitgen.core import gear
    colours = dict(SWORD_COLOURS, **(colours or {}), **({"gold": tuple(int(v) for v in gold)} if gold else {}))
    S = 256
    diff, spec = Image.new("RGB", (S, S)), Image.new("RGBA", (S, S))
    for zone, (u0, v0) in swatch.items():
        box_px = (u0 * S // 2, v0 * S // 2, u0 * S // 2 + S // 2, v0 * S // 2 + S // 2)
        diff.paste(colours[zone], box_px)
        spec.paste((0, 0, 0, SWORD_GLOSS[zone]), box_px)
    grain = gear._noise(S, 48, 3, 0.82, 1.0)
    diff = gear._shade(gear._shade(diff, gear._noise(S, 6, 4, 0.8, 1.0)), grain)
    diff.convert("RGBA").save(out / "sword_diffuse.png")
    spec.save(out / "sword_specular.png")
    gear.normal_from_height(grain, 1.5).save(out / "sword_normal.png")
    return {"kind": "mesh_belt", "mesh": {"geometry": str(out / "sword_part.json"), "diffuse": str(out / "sword_diffuse.png"),
                                          "normal": str(out / "sword_normal.png"), "specular": str(out / "sword_specular.png")}}


# the horse's gold (the user, test 34: "same gold as the chanfron"; the median of its texture's gold texels)
FINIAL_GOLD, FINIAL_GLOSS = (147, 116, 80), 12
# the reference's helmet (new_refs/cataphract_rider.png, its head): a tall turned finial on the crown (a flange on the
# dome, a waisted neck with a ring, a flared lip the plume rises out of: about 0.4 of the dome's width tall), a brow
# band, a nasal guard and ribs up the dome to the finial, riveted. Radii and heights in helmet radii; the bands
# ray-cast onto the low body's dome (from its centre, the crown less a radius), front (-z here) the plume's opposite
HELMET = {"finial": [(0.36, -0.12), (0.36, 0.05), (0.21, 0.12), (0.18, 0.30), (0.25, 0.34), (0.25, 0.42),
                     (0.18, 0.46), (0.18, 0.62), (0.27, 0.78), (0.29, 0.84), (0.2, 0.84)],
          "sides": 12, "brow_elev": 0.05, "brow_w": 0.085, "rib_w": 0.06, "nasal_w": 0.07, "nasal_down": 0.6,
          "lift": 0.05, "sink": 0.04, "ribs": (0, 90, 180, 270), "rib_top": 0.82, "rivets": 12, "rivet_r": 0.045}
# the plume as horsehair (the user, test 32: a round tube read as a sausage): a flat blade along the raw plume's line,
# its height across the line from the raw plume's spread (clamped), a jagged top; strands in the reference's colours
PLUME_HAIR = {"base_h": 0.3, "min_h": 0.75, "max_h": 1.15, "spread_k": 2.8, "end_h": 0.35, "half_thick": 0.11,
              "end_thick": 0.04, "smooth": 2}
# new_refs/cataphract_rider.png's plume, sampled: its dark base (p20-p50) and its lighter strands (p80-p95)
PLUME_DARK, PLUME_LIGHT = (86, 14, 22), (150, 38, 52)


# the helmet rigid on the head bone (test 35): fresh_rider's weight smoothing (24 passes at the standard budget) carried
# the neck's weights up into the dome (about 70% head): posed, the dome sagged ~0.4 under the head's rigid parts (the
# finial floated, the helmet bands hung round the face). Above y0 the dome's pinned back to the head, fully by y1
HEAD_PIN = (14.45, 14.75)


def pin_head(geometry: Path, y0: float = HEAD_PIN[0], y1: float = HEAD_PIN[1]) -> int:
    """Pin the low body's helmet to the head bone (``low_geometry.json`` in place; vertices above ``y0`` mixed
    towards ``head`` 1.0, fully above ``y1``). -> the vertices changed."""
    g = json.loads(Path(geometry).read_text(encoding="utf-8"))
    n = 0
    for i, row in enumerate(g["weights"]):
        y = g["p"][3 * i + 1]
        if y <= y0:
            continue
        t = min(1.0, (y - y0) / (y1 - y0))
        f = t * t * (3 - 2 * t)
        new = {k: w * (1 - f) for k, w in row.items()}
        new["head"] = new.get("head", 0.0) + f
        g["weights"][i] = {k: w for k, w in new.items() if w > 1e-4}
        n += 1
    Path(geometry).write_text(json.dumps(g), encoding="utf-8")
    return n


# the rider's legs clear of the horse's barrel (Track P1: "rider seating and legs clear"; measured on the seated unit,
# quality.legs_clear): the legs were fitted to vanilla's horse; on a wider one (the cataphract's caparison: 0.6-1.05
# wider at the thighs' heights) they sank into it (23% of the lower body inside against vanilla's 9%)
SEAT_BAND = (10.5, 11.5)  # the horse's heights where the seated thighs lie
LEG_BONES = ("L_thigh", "L_knee", "L_foot", "L_toe", "R_thigh", "R_knee", "R_foot", "R_toe")


def barrel_half(p: list, ys=SEAT_BAND, z=(-2.5, 2.5)) -> list[float]:
    """A horse's half-width at each of the seated thighs' heights: the 95th percentile of |x| in a band 1 tall."""
    out = []
    for y in ys:
        xs = sorted(abs(p[i]) for i in range(0, len(p), 3) if y - 0.5 < p[i + 1] < y + 0.5 and z[0] < p[i + 2] < z[1])
        out.append(xs[int(len(xs) * 0.95)] if xs else 0.0)
    return out


def leg_spread(horse_p: list, vanilla_p: list, cap: float = 1.5) -> float:
    """How far the rider's legs move out for this horse: its barrel's largest extra half-width over vanilla's at
    the same height (0 if nowhere wider), at most ``cap``."""
    d = max(a - b for a, b in zip(barrel_half(horse_p), barrel_half(vanilla_p)))
    return round(max(0.0, min(cap, d)), 3)


def spread_legs(geometry: Path, d: float) -> int:
    """The rider's legs moved out sideways by ``d`` (``low_geometry.json`` in place): each vertex by ``d`` times its
    legs' share of its weights, away from the centre line. -> the vertices moved."""
    if d <= 0:
        return 0
    g = json.loads(Path(geometry).read_text(encoding="utf-8"))
    n = 0
    for i, row in enumerate(g["weights"]):
        w = sum(v for k, v in row.items() if k in LEG_BONES)
        if w <= 0:
            continue
        x = g["p"][3 * i]
        side = 1.0 if (sum(v for k, v in row.items() if k.startswith("L_")) >= sum(v for k, v in row.items() if k.startswith("R_"))) else -1.0
        if abs(x) > 1e-6 and (x > 0) != (side > 0):
            side = 1.0 if x > 0 else -1.0
        g["p"][3 * i] = x + side * d * min(1.0, w)
        n += 1
    Path(geometry).write_text(json.dumps(g), encoding="utf-8")
    return n


def helmet_gold(m, info: dict, body: dict, cfg: dict | None = None) -> dict:
    """The reference's helmet gold on ``m`` (a ``core.gear.Mesh`` in the head bone's frame): the brow band, the
    nasal guard, the ribs and rivets on the low body's dome (``body``: {"geometry": low_geometry.json, "head": the
    head bone's position in it}; the helmet's rigid on the head bone, so the bands stay put). -> counts."""
    from eu4unitgen.core import gear
    c = {**HELMET, **(cfg or {})}
    g = json.loads(Path(body["geometry"]).read_text(encoding="utf-8"))
    hx, hy, hz = body["head"]
    P = [[g["p"][i] - hx, g["p"][i + 1] - hy, g["p"][i + 2] - hz] for i in range(0, len(g["p"]), 3)]
    R = float(info.get("helmet_radius", 1.0))
    bx, by, bz = info["finial_base"]
    centre = [bx, by + 0.08 - R, bz]  # the crown (finial_base: 0.08 under the low body's top) less a radius
    near = [(P[a], P[b], P[c2]) for a, b, c2 in zip(g["tri"][0::3], g["tri"][1::3], g["tri"][2::3])
            if any(sum((P[k][j] - centre[j]) ** 2 for j in range(3)) < (2.0 * R) ** 2 for k in (a, b, c2))]
    line = info.get("line") or []
    back = 1.0 if not line or line[-1][2] >= bz else -1.0  # the plume flows back

    def hit(az, el):  # azimuth 0 = front, 90 = the rider's left (x+); elevation up from the dome's centre
        a, e = math.radians(az), math.radians(el)
        d = [math.cos(e) * math.sin(a), math.sin(e), -back * math.cos(e) * math.cos(a)]
        h = gear.raycast_far(near, centre, d, 2.5 * R)
        return (gear._add(centre, d, h[0]), h[1]) if h else None

    def smooth(ns):
        return [gear._norm(gear._add(gear._add(ns[i - 1] if i else ns[i], ns[i]), ns[i + 1] if i + 1 < len(ns) else ns[i]))
                for i in range(len(ns))]
    counts = {}
    shows = c.get("shows") or {"brow": True, "nasal": True, "ribs": True}  # read_helmet's flags (Track P1)
    brow_el = math.degrees(c["brow_elev"])
    ring = [hit(az, brow_el) for az in range(0, 360, 10)]
    ring = [q for q in ring if q]
    if len(ring) > 20 and shows.get("brow"):
        ns = smooth([n for _, n in ring])
        gear.surface_band(m, [p for p, _ in ring], ns, [c["brow_w"] * R] * len(ring), c["lift"] * R, c["sink"] * R, closed=True)
        for k in range(int(c["rivets"])):
            p, n = ring[(k * len(ring)) // int(c["rivets"])]
            gear.stud(m, gear._add(p, ns[(k * len(ring)) // int(c["rivets"])], c["lift"] * R), n, c["rivet_r"] * R,
                      0.6 * c["rivet_r"] * R, (0.875, 0.5))
        counts["brow"] = len(ring)
    top_el = math.degrees(math.asin(min(0.99, c["rib_top"])))
    for az in (c["ribs"] if shows.get("ribs") else ()):
        pts = [hit(az, brow_el + (top_el - brow_el) * k / 9) for k in range(10)]
        pts = [q for q in pts if q]
        if len(pts) > 3:
            gear.surface_band(m, [p for p, _ in pts], smooth([n for _, n in pts]), [c["rib_w"] * R] * len(pts),
                              0.8 * c["lift"] * R, c["sink"] * R)
            counts[f"rib_{az}"] = len(pts)
    nasal = [hit(0, brow_el - k * 8) for k in range(int(c["nasal_down"] * 10))]
    nasal = [q for q in nasal if q and q[0][1] > centre[1] - c["nasal_down"] * R]
    if len(nasal) > 2 and shows.get("nasal"):
        gear.surface_band(m, [p for p, _ in nasal], smooth([n for _, n in nasal]),
                          [c["nasal_w"] * R * (1 - 0.3 * k / len(nasal)) for k in range(len(nasal))], c["lift"] * R, c["sink"] * R)
        counts["nasal"] = len(nasal)
    return counts


# the cavalry_rider_* animations scale the rider's root to 0.805 (measured: every bone's posed scale at frame 1 of
# cavalry_rider_idle on western_generic_01_type1); the body shrinks with it. The gallery (render_unit) places an
# attached part at its bone without that scale, so a head part built in the body's units came out 1.24x too big and
# 1.24x too far from the head bone (test 35: the finial floated, the helmet bands hung round the face).
# UNVERIFIED: whether the game applies the bone's animated scale to an attached entity. Test 35 shows both (tier 1
# scaled by this, as the gallery wants; tier 2 unscaled); TODO: keep the one that sits right in game
RIDER_ANIM_SCALE = 0.805


def plume_part(parts_dir: Path, info: dict, cfg: dict | None = None, colours: tuple | None = None,
               body: dict | None = None, scale: float = 1.0, gold=None) -> dict:
    """M23 (the user, tests 31 and 32): the rider's plume as a rigid head part (``mesh_head``) with its gold holder
    (finial): our own geometry on ``fresh_rider``'s ``plume_line`` (the raw plume's centre line and side-to-side
    spread, in the head bone's frame). The finial: a socket, a post and a knob on the helmet's top. The plume: a
    horsehair blade (``core.gear``) out of the knob, low at the base, tall along the line, low at its end; strand
    texture (``colours``: dark, light; the reference's by default) beside a gold patch. The finial (test 35): the
    reference's tall turned one (``HELMET``), the plume out of its lip; with ``body`` the helmet's gold bands too
    (``helmet_gold``). ``scale``: the whole part about the head bone (``RIDER_ANIM_SCALE``: the gallery's model).
    ``gold``: the unit's gold (Track P1: the palette's; ``FINIAL_GOLD``, the cataphract's, by default).
    -> a loadout part entry."""
    from PIL import Image

    from eu4unitgen.core import gear
    c = {**PLUME_HAIR, **(cfg or {})}
    dark, light = colours or (PLUME_DARK, PLUME_LIGHT)
    bx, by, bz = info["finial_base"]
    R = float(info.get("helmet_radius", 1.0))
    prof = [(r * R, y * R) for r, y in HELMET["finial"]]
    h = prof[-1][1]
    tip = [bx, by + h * 0.95, bz]
    rise = [bx, tip[1] + 0.4, bz]
    line = [list(map(float, q)) for q in info["line"]] or [[bx, by + h + 1.3, bz + 0.3]]
    # the raw line's first points under the rise, close over the finial: a dip; dropped
    line = [q for q in line if q[1] > rise[1] - 0.1 or math.hypot(q[0] - bx, q[2] - bz) > 0.6] or line[-1:]
    pts = [tip, rise] + line
    heights = [c["base_h"], c["base_h"] * 1.6] + [min(max(c["spread_k"] * w, c["min_h"]), c["max_h"])
                                                  for w in info.get("half_width", [])]
    heights = (heights + [c["min_h"]] * len(pts))[:len(pts)]
    heights[-1] = c["end_h"]
    pts, heights = gear.chaikin(pts, heights, int(c["smooth"]))
    n = len(pts)
    thick = [c["half_thick"] + (c["end_thick"] - c["half_thick"]) * max(0.0, (i / (n - 1) - 0.6) / 0.4) for i in range(n)]
    blade = gear.horsehair_blade(pts, heights, thick)
    # the hair's uv into the left 3/4 of the image, the finial on the gold strip at the right
    m = gear.Mesh()
    m.p, m.tri, m.uv = list(blade["p"]), list(blade["tri"]), [x * (0.75 if k % 2 == 0 else 1.0) for k, x in enumerate(blade["uv"])]
    gold_uv = (0.875, 0.5)

    gear.lathe(m, prof, [bx, by, bz], int(HELMET["sides"]), gold_uv)
    if body:
        helmet_gold(m, info, body, cfg)
    if scale != 1.0:
        m.p = [x * scale for x in m.p]
    parts_dir.mkdir(parents=True, exist_ok=True)
    (parts_dir / "plume_part.json").write_text(json.dumps(m.geometry()), encoding="utf-8")
    tex = gear.strand_texture(dark, light, 256)
    S = 256
    images = {}
    gold = tuple(int(v) for v in (gold or FINIAL_GOLD))
    for kind, fill in (("diffuse", gold + (255,)), ("specular", (0, 0, 0, FINIAL_GLOSS)), ("normal", (128, 128, 255, 255))):
        im = Image.new("RGBA", (S, S), fill)
        im.paste(tex[kind].resize((int(S * 0.75), S), Image.LANCZOS), (0, 0))
        if kind == "diffuse":  # the gold strip: darker at its edges, a raised band's light
            for x in range(int(S * 0.75), S):
                t = abs((x / S - 0.875) / 0.125)
                k = 1.08 - 0.4 * t * t
                im.paste(tuple(min(255, int(v * k)) for v in gold) + (255,), (x, 0, x + 1, S))
        if kind == "normal":  # the gold strip's raised band: a ridge across u (Track P1: no flat normal maps)
            for x in range(int(S * 0.75), S):
                t = (x / S - 0.875) / 0.125
                nx = int(128 + 90 * max(-1.0, min(1.0, t)) * 0.6)
                im.paste((nx, 128, 255, 255), (x, 0, x + 1, S))
        images[kind] = parts_dir / f"plume_{kind}.png"
        im.save(images[kind])
    return {"kind": "mesh_head", "mesh": {"geometry": str(parts_dir / "plume_part.json"), **{k: str(v) for k, v in images.items()}}}


def node_bind(mesh_bytes: bytes, node: str) -> list | None:
    """A bone's bind pose in a .mesh (the skeleton summary's 12 numbers, as the gallery passes render_unit)."""
    from eu4unitgen.library.scanner import parse_pdx_mesh, summarize_mesh
    b = (summarize_mesh(parse_pdx_mesh(mesh_bytes))["skeleton"].get(node) or {}).get("bind")
    return list(b) if b else None


def kite_part(parts_dir: Path, frame: dict, height: float, primary=None, cfg: dict | None = None) -> dict:
    """M23 (the user, test 32): the rider's shield as the reference's kite (``core.gear.kite_shield``), strapped to
    the left forearm like the raw shield was (``rider_frames``' Left_hand_node frame: half way to the elbow, 0.6
    out, 1.5 down; facing out, its top up, its width along the rider's forward), ``height`` tall; the field on the
    country colour's mask against ``primary``. -> a loadout part entry (``mesh_shield``)."""
    from eu4unitgen.core import gear
    c = {"out": 1.2, "down": -0.3, **(cfg or {})}  # out 0.6, down 1.5 (the raw disc's): the kite's point sank into the caparison
    centre = [0.5 * frame["elbow"][k] + c["out"] * frame["out"][k] - c["down"] * frame["up"][k] for k in range(3)]
    geo, tex = gear.kite_shield(height, centre, frame["out"], frame["up"], frame["forward"], primary=primary, cfg=cfg,
                                **(c.get("kite_kw") or {}))
    parts_dir.mkdir(parents=True, exist_ok=True)
    (parts_dir / "kite_part.json").write_text(json.dumps(geo), encoding="utf-8")
    paths = {}
    for kind, im in tex.items():
        paths[kind] = str(parts_dir / f"kite_{kind}.png")
        im.save(paths[kind])
    return {"kind": "mesh_shield", **({"colours": {"own": "country1"}} if primary is not None else {}),
            "mesh": {"geometry": str(parts_dir / "kite_part.json"), **paths}}


RIDER_GROUPS = {"head": ["head"], "arms": ["R_shoulder", "R_arm", "R_elbow", "R_wrist", "R_hand", "L_shoulder", "L_arm",
                                          "L_elbow", "L_wrist", "L_hand"],
                "legs": ["R_thigh", "R_knee", "R_foot", "R_toe", "L_thigh", "L_knee", "L_foot", "L_toe"],
                "torso": ["hip", "lower_spine", "mid_spine", "top_spine", "root"]}


def default_reference(glb: Path) -> dict | None:
    """The cut-out reference ``run`` left beside a TRELLIS.2 model (``input/``), or None."""
    d = Path(glb).parent / "input"
    if (d / "input_rgb.png").is_file() and (d / "input_fg.png").is_file():
        return {"image": str(d / "input_rgb.png"), "fg": str(d / "input_fg.png")}
    return None


def mirror_part(geometry: Path, frame_from: dict, frame_to: dict, out: Path) -> None:
    """A part in one hand node's frame moved into the other hand's as its mirror image (the shield side swapped):
    each point's outward, up and forward parts (``rider_frames``, the seated pose) laid on the other node's outward,
    up and forward; the winding reversed (a mirror turns faces inside out)."""
    g = json.loads(Path(geometry).read_text(encoding="utf-8"))
    A = [frame_from[k] for k in ("out", "up", "forward")]
    B = [frame_to[k] for k in ("out", "up", "forward")]
    p = g["p"]
    q = []
    for i in range(0, len(p), 3):
        c = [sum(p[i + j] * a[j] for j in range(3)) for a in A]
        q += [sum(c[k] * B[k][j] for k in range(3)) for j in range(3)]
    tri = g["tri"]
    g["p"] = [round(v, 5) for v in q]
    g["tri"] = [v for t in range(0, len(tri), 3) for v in (tri[t], tri[t + 2], tri[t + 1])]
    Path(out).write_text(json.dumps(g), encoding="utf-8")


HAND = {"left": "Left_hand_node", "right": "Right_hand_node"}
HAND_KIND = {"Right_hand_node": "mesh_weapon", "Left_hand_node": "mesh_shield"}  # core.parts: the kind gives the bone


def process_rider(glb: Path, root: Path, country_mask: bool = True, blender: str | None = None, plume: str = "part",
                  sword: bool = False, kite: bool = True, cfg: dict | None = None, reference: dict | None = None,
                  ai_python: str | None = None, palette_from: dict | None = None, part_method: str = "auto",
                  part_cache: Path | None = None, shield_side: str = "auto", fit: str = "seated") -> dict:
    """A TRELLIS.2 rider (.glb from a front view) into ``root/vNNN`` (SPEC M23): Blender job fresh_rider (the body on
    the infantry skeleton; the lance and shield split off as parts in their hand nodes' frames); bake_colour (the
    body and each part; the cloth as the country colour's mask) and bake_horse (the body's normal and AO); the
    deformation check against vanilla's rider (cavalry_rider_* animations, per bone group).
    Parts (Track P1.2, ``part_gen``): with the ``reference`` (default: ``run``'s cut-out beside the model) and the AI
    environment, each part is found and cut out of the reference; the shield is TRELLIS.2's own model of its crop
    (``part_method`` "auto"), checked against the crop, else the procedural kite traced from the crop; the plume's
    colours and the helmet's gold come from their crops; every colour is moved onto the unit's palette
    (``palette_from``: {"diffuse", "specular", "primary"} of the horse it rides, else the rider's own).
    ``shield_side``: "left" or "right" forearm ("auto": where the reference has it); the weapon goes to the other
    hand, mirrored (``mirror_part``). ``plume`` "none" leaves the plume off.
    -> the manifest (``parts``: ready loadout part entries; ``part_checks``)."""
    from PIL import Image

    from eu4unitgen.blender.runner import BlenderRunner
    from eu4unitgen.core.fresh_mesh import base_arrays, geometry_from_job, write_fresh
    out = next_version(Path(root)).resolve()
    (out / "parts").mkdir(parents=True)
    br = BlenderRunner(blender or _config().blender(), log_dir=out / "logs", timeout=3600)
    base_file = eu4() / "gfx" / "models" / "Units" / "western_generic_01_type1.mesh"
    base = base_file.read_bytes()
    t0 = time.time()
    basecolor = glb_basecolor(glb, out / "raw_basecolor.png")
    anim = eu4() / "gfx" / "models" / "Units" / "animation" / "cavalry_rider_idle.anim"
    frame = br.run_job("rider_frames", {"mesh": str(base_file), "anim": str(anim), "node": "Left_hand_node",
                                        "elbow": "L_elbow", "shoulder": "L_shoulder",
                                        "bind": node_bind(base, "Left_hand_node")}).result  # the strapped shield's frame
    fr = br.run_job("fresh_rider", {"vanilla": base_arrays(base), "source": {"path": str(Path(glb).resolve()), "merge": 1e-5},
                                    **{**RIDER, **(STANDING if fit == "standing" else {}), **(cfg or {})}, "fit": fit,
                                    "shield_frame": frame, "plume": "part" if plume in ("part", "none") else "cut", "geometry_out": str(out / "low_geometry.json"), "preview_out": str(out / "low.obj"),
                                    "fitted_out": str(out / "fitted.obj"), "textured_out": str(out / "textured.obj"),
                                    "parts_dir": str(out / "parts")}).result
    fr["head_pinned"] = pin_head(out / "low_geometry.json")
    raw = br.run_job("render_model", {"model": str(Path(glb).resolve()), "engine": "EEVEE", "out_dir": str(out / "raw"),
                                      "name": "raw", "size": 900, "views": {"front": [0, 5]}}).result
    common = {"gloss": PROCESS["gloss"], **PROCESS["bake"], "image": str(basecolor), "colour_render": raw["images"]["front"]}
    metallic = glb_metallic(glb, out / "raw_metallic.png")
    if metallic:  # metal from TRELLIS.2's own map where it varies (bake_colour.baked_metal)
        common["metallic_image"] = str(metallic)
    bc = br.run_job("bake_colour", {**common, "size": PROCESS["size"], "country_mask": bool(country_mask),
                                    "low": str(out / "low_geometry.json"), "high": str(out / "textured.obj"),
                                    "out_dir": str(out), "name": "rider"}).result
    nb = br.run_job("bake_horse", {"low": str(out / "low_geometry.json"), "high": [str(out / "textured.obj")],
                                   "diffuse": bc["diffuse"], "size": PROCESS["size"], **PROCESS["normal"], "preview": False,
                                   "out_dir": str(out), "name": "riderb"}).result
    tex_body = {"diffuse": str(out / "riderb_diffuse.png")}
    parts = {}
    for key in ("lance", "shield"):
        low = out / "parts" / f"{key}_low.json"
        if not low.is_file():
            continue
        pb = br.run_job("bake_colour", {**common, "size": 512, "country_mask": bool(country_mask) and key == "shield",
                                        "low": str(low), "high": str(out / "parts" / f"{key}_textured.obj"),
                                        "out_dir": str(out / "parts"), "name": key}).result
        # its normal map baked from the raw model too (Track P1: no flat normal maps on parts)
        br.run_job("bake_horse", {"low": str(low), "high": [str(out / "parts" / f"{key}_textured.obj")],
                                  "diffuse": pb["diffuse"], "size": 512, "cage": 0.05, "distance": 0.2, "high_smooth": 0,
                                  "strength": 0.8, "samples": 16, "ao_strength": 0.0, "preview": False,
                                  "out_dir": str(out / "parts"), "name": f"{key}b"})
        (out / "parts" / f"{key}b_normal.png").replace(out / "parts" / f"{key}_normal.png")
        parts[key] = {"kind": "mesh_weapon" if key == "lance" else "mesh_shield",
                      **({"colours": {"own": "country1"}} if pb.get("default_primary") else {}),
                      "mesh": {"geometry": str(out / "parts" / f"{key}_part.json"), "diffuse": pb["diffuse"],
                               "normal": str(out / "parts" / f"{key}_normal.png"), "specular": pb["specular"]}}
    # the unit's palette: the horse's it rides (the shield "same red and gold as the caparison"), else the rider's
    from eu4unitgen.core import palette as palmod
    from eu4unitgen.ref3d import part_gen
    ps = palette_from or {"diffuse": tex_body["diffuse"], "specular": bc["specular"], "primary": bc.get("default_primary")}
    pal = palmod.classes(ps["diffuse"], ps["specular"], ps.get("primary"))
    primary = (ps.get("primary") or bc.get("default_primary") or [255, 77, 70]) if country_mask else None
    checks, crops_res = {}, {}
    ref = reference if reference is not None else default_reference(glb)
    ai_py = ai_python or _config().ai()
    cached = Path(glb).parent / "crops_cache"  # a starter part's crops (cut from its picture when it was made)
    if (cached / "crops.json").is_file() and not (out / "parts" / "crops" / "crops.json").is_file():
        import shutil
        shutil.copytree(cached, out / "parts" / "crops", dirs_exist_ok=True)
    have_crops = (out / "parts" / "crops" / "crops.json").is_file()
    if part_method != "procedural" and ref and (ai_py or have_crops) and (out / "parts" / "fitted_points.json").is_file():
        try:
            loc = part_gen.locate(json.loads((out / "parts" / "fitted_points.json").read_text(encoding="utf-8")),
                                  Image.open(ref["fg"]).convert("L"))
            checks["locate"] = {"mirror": loc["mirror"], "fit_iou": loc["fit_iou"]}
            crops_res = part_gen.crops(Path(ref["image"]), Path(ref["fg"]), loc, out / "parts" / "crops", ai_py,
                                       ai_paths(ai_py)["sam2"])
        except Exception as exc:  # noqa: BLE001 - the parts fall back to procedural; recorded
            checks["crops_error"] = f"{type(exc).__name__}: {exc}"
    frames = {"Left_hand_node": frame}

    def hand_frame(node):
        if node not in frames:
            side = "R" if node.startswith("Right") else "L"
            frames[node] = br.run_job("rider_frames", {"mesh": str(base_file), "anim": str(anim), "node": node,
                                                        "elbow": f"{side}_elbow", "shoulder": f"{side}_shoulder",
                                                        "bind": node_bind(base, node)}).result
        return frames[node]
    lance_node = ((fr.get("parts") or {}).get("lance") or {}).get("node")
    split_shield = ((fr.get("parts") or {}).get("shield") or {}).get("node")
    other = {"Left_hand_node": "Right_hand_node", "Right_hand_node": "Left_hand_node"}
    shield_node = HAND.get(shield_side) or split_shield or (other[lance_node] if lance_node else "Left_hand_node")
    checks["shield_side"] = {"shield": shield_node, "asked": shield_side}
    if "lance" in parts and lance_node == shield_node:  # the weapon to the other hand, mirrored
        to = other[lance_node]
        mirror_part(Path(parts["lance"]["mesh"]["geometry"]), hand_frame(lance_node), hand_frame(to),
                    out / "parts" / "lance_part.json")
        parts["lance"]["mesh"]["geometry"] = str(out / "parts" / "lance_part.json")
        parts["lance"]["kind"] = HAND_KIND[to]
        checks["shield_side"]["weapon_moved_to"] = to
    frame = hand_frame(shield_node)
    if "shield" in parts:
        parts["shield"]["kind"] = HAND_KIND[shield_node]
    if kite and "shield" in parts:  # the reference's shield in the raw shield's place, about as tall (the user, test 32)
        b = (fr.get("parts") or {}).get("shield", {}).get("bounds")
        tall = 0.88 * max(hi - lo for lo, hi in zip(*b)) if b else 5.6
        parts["shield"], checks["shield"] = shield_from_reference(
            br, out / "parts", frame, tall, crops_res.get("shield"), pal, primary,
            {k: v for k, v in common.items() if k != "colour_render"}, part_method, part_cache or (Path(glb).parent / "parts_cache"),
            Path(out / "parts" / "shield_part.json"))
        parts["shield"]["kind"] = HAND_KIND[shield_node]
    if plume == "part" and (fr.get("plume_line") or {}).get("line"):  # with its gold holder (the user, test 31)
        hair = None
        if (crops_res.get("crest") or {}).get("crop"):
            hair = part_gen.hair_colours(Image.open(crops_res["crest"]["crop"]))
        helmet = part_gen.read_helmet(Image.open(crops_res["helmet"]["crop"])) if (crops_res.get("helmet") or {}).get("crop") else None
        shows = {k: helmet[k] for k in ("brow", "nasal", "ribs")} if helmet else None
        checks["plume"] = {"colours": hair, "from": "crop" if hair else "default"}
        checks["helmet"] = helmet or {"from": "default (no crop)"}
        gold = (pal.get("gold") or {}).get("colour") if (pal.get("gold") or {}).get("share", 0) > 0.005 else None
        parts["plume"] = plume_part(out / "parts", fr["plume_line"], colours=hair,
                                    body={"geometry": str(out / "low_geometry.json"), "head": base_arrays(base)["bones"]["head"]}
                                    if (helmet is None or helmet.get("gold_share", 0) > 0.03) else None,
                                    scale=RIDER_ANIM_SCALE, gold=gold, cfg={"shows": shows} if shows else None)
    if sword:  # on the side the shield doesn't cover (the reference hangs it on the shield's side; the user, test 30)
        hip_anim = anim if fit != "standing" else anim.with_name("polearm_twohanded_idle.anim")
        hip_frame = br.run_job("rider_frames", {"mesh": str(base_file), "anim": str(hip_anim), "node": "hip",
                                                "elbow": "R_thigh", "shoulder": "R_shoulder", "bind": node_bind(base, "hip")}).result
        parts["sword"] = sword_part(out / "parts", hip_frame, float(fr.get("slim", {}).get("hip_half", 1.9)),
                                    cfg={"hang_deg": 25.0} if fit == "standing" else None,
                                    gold=(pal.get("gold") or {}).get("colour") if (pal.get("gold") or {}).get("share", 0) > 0.005
                                    else None, colours=sword_colours(pal))
    if palette_from and palette_from.get("geometry"):  # the legs clear of the horse it rides
        hp = json.loads(Path(palette_from["geometry"]).read_text(encoding="utf-8"))["p"]
        vp = base_arrays((eu4() / "gfx" / "models" / "Units" / "cavalry_horse.mesh").read_bytes())["p"]
        d = leg_spread(hp, vp)
        fr["leg_spread"] = {"by": d, "vertices": spread_legs(out / "low_geometry.json", d)}
    summary = rider_deform(br, out, base, base_file)
    tex ={"diffuse": str(out / "riderb_diffuse.png"), "specular": bc["specular"], "normal": str(out / "riderb_normal.png")}
    # the raw model and the game body from the front, one camera (the quality report's outline against the reference)
    # the raw model and the game version, one camera per view, their overlay and colours by region (the user, T38:
    # "the game version must look like TRELLIS.2's model, just simpler"); the game version as the game tints it
    cmp_ = compare_sheet(br, out, basecolor, tinted(Path(tex["diffuse"]), Path(bc["specular"]), bc.get("default_primary"),
                                                    out / "game_tinted.png"), views=RIDER_VIEWS)
    manifest = {"stage": "rider game version (M23)", "compare": cmp_, "glb": str(glb), "version": out.name, "fresh": fr, "colour": bc,
                "normal": {k: nb.get(k) for k in ("tilt", "match")}, "textures": tex, "parts": parts,
                "part_checks": checks, "palette": pal,
                "geometry": str(out / "low_geometry.json"), "deform": summary, "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    return manifest


def rider_deform(br, out: Path, base: bytes, base_file: Path, renders: bool = True, prefix: str = "cavalry_rider_",
                 anims=("idle", "moving", "attack"), geometry: Path | None = None) -> dict:
    """The deformation check (check_mesh_deform, per bone group) on the rider's game mesh (``geometry``, default
    ``out/low_geometry.json``) and vanilla's body under ``<prefix><anim>`` animations (a soldier on foot: its
    stance's, e.g. ``spear_shield_``)."""
    from eu4unitgen.core.fresh_mesh import geometry_from_job, write_fresh
    dd = out / ("deform" if prefix == "cavalry_rider_" else f"deform_{prefix.rstrip('_')}")
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "rider.mesh").write_bytes(write_fresh(base, geometry_from_job(geometry or out / "low_geometry.json")))
    jobs = [{"id": k, "path": str(eu4() / "gfx" / "models" / "Units" / "animation" / f"{prefix}{k}.anim")} for k in anims]
    res = {name: br.run_job("check_mesh_deform", {"mesh": str(m), "anims": jobs, "groups": RIDER_GROUPS,
                                                  "renders": {"out_dir": str(dd), "name": name, "size": 700} if renders else None}).result
           for name, m in (("ours", dd / "rider.mesh"), ("vanilla", base_file))}
    (dd / "deform.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return {who: {k: {"stretch": [round(x["str_p05"], 2), round(x["str_p95"], 2)], "max": round(x["str_max"], 2),
                      "flips": x["flips"], "groups": {g: m.get("flips") for g, m in (x.get("groups") or {}).items()}}
                  for k, x in res[who]["anims"].items()} for who in res}


def foot_stance(manifest: dict) -> str:
    """The infantry animations a soldier on foot plays (``core.loadout``'s auto stance for its own parts): a shield
    -> spear_shield, a weapon alone -> polearm_twohanded."""
    parts = manifest.get("parts") or {}
    kinds = {p.get("kind") for p in parts.values()}
    return "spear_shield" if "mesh_shield" in kinds else "polearm_twohanded" if "mesh_weapon" in kinds else "spear_shield"


def foot_deform(vdir: Path, out: Path, blender: str | None = None) -> dict:
    """A soldier's game version (``vdir``) checked on foot: its stance's idle, moving and attack (vanilla's files,
    e.g. spear_shield_idle.anim), against vanilla's body under the same ones. -> rider_deform's summary + stance."""
    from eu4unitgen.blender.runner import BlenderRunner
    m = json.loads((Path(vdir) / "manifest.json").read_text(encoding="utf-8"))
    stance = foot_stance(m)
    base_file = eu4() / "gfx" / "models" / "Units" / "western_generic_01_type1.mesh"
    br = BlenderRunner(blender or _config().blender(), log_dir=Path(out) / "logs", timeout=1800)
    d = rider_deform(br, Path(out), base_file.read_bytes(), base_file, prefix=f"{stance}_", geometry=Path(m["geometry"]))
    return {**d, "stance": stance}


# A hand part on foot follows the base game's convention for its stance, not the seated pose it was split off in
# (2026-10-28, the cataphract on foot: the lance hung point down, the kite behind the body). Read from the install:
# muslim_spear (indiangfx's spear_shield infantry) points along +z, which spear_shield_idle and polearm_twohanded_idle
# turn straight up; indian_generic_01_type1_shield faces +x (out and forward in spear_shield_idle), its back at the
# hand. The centre and back as core.parts' hoplon (in game, test 7).
FOOT = {"shield_centre_y": 1.2, "shield_back_x": -0.1}


def _unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _basis(a, b):
    """(a, b made perpendicular to a, a x b): a right-handed orthonormal basis."""
    a = _unit(a)
    d = sum(x * y for x, y in zip(a, b))
    b = _unit([y - d * x for x, y in zip(a, b)])
    return [a, b, [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]]


def _long_axis(pts):
    """The points' principal axis (power iteration on the covariance; pure Python)."""
    n = len(pts)
    m = [sum(p[k] for p in pts) / n for k in range(3)]
    c = [[sum((p[i] - m[i]) * (p[j] - m[j]) for p in pts) for j in range(3)] for i in range(3)]
    v = [1.0, 1.0, 1.0]
    for _ in range(50):
        v = _unit([sum(c[i][j] * v[j] for j in range(3)) for i in range(3)])
    return v


def foot_part(p: list, kind: str, seated: dict, foot: dict) -> list:
    """A hand part's points (flat x, y, z) from the seated hand frame (``rider_frames`` on cavalry_rider_idle) into
    the base game's convention on foot (``FOOT``): a weapon's point (the end that pointed forward and up when seated)
    along +z, rolled so what faced out still faces out, the grip kept at the hand; a shield's face (out when seated)
    along +x, its top towards the standing pose's up, its back at ``shield_back_x`` and its centre at
    ``shield_centre_y`` above the hand. ``foot`` is the hand's frame in the stance's idle. A rotation: no winding
    change. -> the new flat points."""
    pts = [p[i:i + 3] for i in range(0, len(p), 3)]
    if kind == "mesh_weapon":
        t = _long_axis(pts)
        lead = [f + u for f, u in zip(seated["forward"], seated["up"])]
        if sum(a * b for a, b in zip(t, lead)) < 0:
            t = [-x for x in t]
        A, B = _basis(t, seated["out"]), _basis([0.0, 0.0, 1.0], foot["out"])
    else:
        A, B = _basis(seated["out"], seated["up"]), _basis([1.0, 0.0, 0.0], foot["up"])
    q = []
    for v in pts:
        c = [sum(v[j] * a[j] for j in range(3)) for a in A]
        q.append([sum(c[k] * B[k][j] for k in range(3)) for j in range(3)])
    if kind != "mesh_weapon":
        lo = [min(v[k] for v in q) for k in range(3)]
        hi = [max(v[k] for v in q) for k in range(3)]
        shift = [FOOT["shield_back_x"] - lo[0], FOOT["shield_centre_y"] - (lo[1] + hi[1]) / 2, -(lo[2] + hi[2]) / 2]
        q = [[v[k] + shift[k] for k in range(3)] for v in q]
    return [round(x, 5) for v in q for x in v]


# the shield's back this far out from the forearm; flagged past the base game's own: its Indian shield strays up to
# 0.89 from the forearm's middle (0.27 on average) through spear_shield's 50 sampled frames (measured 2026-10-29)
STRAP = {"out": 0.15, "max_gap": 0.95, "mean_gap": 0.3}


def strap_part(p: list, seated: dict, frames: list[dict]) -> tuple[list, dict]:
    """A shield's points (flat x, y, z, in the seated hand frame) strapped to the forearm through every sampled frame
    of the soldier's animations on foot (T37 in game: placed by the base game's convention for its own small shield,
    our kite floated in front of him and over the hand). ``frames``: ``rider_frames``' per-frame records on
    Left_hand_node (the elbow, the outward side and world up, all in the hand's frame). The shield turned so its face
    (out when seated) points along the frames' mean outward side and its top along their mean up, its back's centre on
    the forearm's mean middle, ``STRAP["out"]`` outwards. -> (the new flat points, the check: each frame's gap
    between the forearm's middle and the shield's back plane, and how far the middle lies off the shield)."""
    pts = [p[i:i + 3] for i in range(0, len(p), 3)]
    mean = lambda vs: [sum(v[k] for v in vs) / len(vs) for k in range(3)]  # noqa: E731
    out = _unit(mean([f["out"] for f in frames]))
    up = mean([f["up"] for f in frames])
    A, B = _basis(seated["out"], seated["up"]), _basis(out, up)
    q = []
    for v in pts:
        c = [sum(v[j] * a[j] for j in range(3)) for a in A]
        q.append([sum(c[k] * B[k][j] for k in range(3)) for j in range(3)])
    d = [sum(v[k] * B[0][k] for k in range(3)) for v in q]  # along the face's outward normal
    back = [v for v, dv in zip(q, d) if dv <= min(d) + 0.2 * (max(d) - min(d))]
    centre = mean(back)
    mids = [[0.5 * f["elbow"][k] for k in range(3)] for f in frames]
    strap = [mean(mids)[k] + STRAP["out"] * B[0][k] for k in range(3)]
    shift = [strap[k] - centre[k] for k in range(3)]
    q = [[v[k] + shift[k] for k in range(3)] for v in q]
    # the check: each frame's forearm middle against the back plane (through ``strap``, normal B[0]) and the shield's
    # extent across it (B[1] up, B[2] across)
    half = [max(abs(sum((v[k] - strap[k]) * B[i][k] for k in range(3))) for v in q) for i in (1, 2)]
    gaps, off = [], []
    for m_ in mids:
        rel = [m_[k] - strap[k] for k in range(3)]
        gaps.append(abs(sum(rel[k] * B[0][k] for k in range(3))))
        off.append(max(abs(sum(rel[k] * B[i][k] for k in range(3))) - half[j] for j, i in enumerate((1, 2))))
    check = {"frames": len(frames), "max_gap": round(max(gaps), 3), "mean_gap": round(sum(gaps) / len(gaps), 3),
             "max_off_shield": round(max(off), 3),
             "ok": max(gaps) <= STRAP["max_gap"] and sum(gaps) / len(gaps) <= STRAP["mean_gap"] and max(off) <= 0.0}
    return [round(x, 5) for v in q for x in v], check


FOOT_ANIMS = ("idle", "moving", "attack")  # the crew is the infantry entity: it plays these same states at the gun


def to_foot(vdir: Path, blender: str | None = None) -> dict:
    """A soldier's built version (``vdir``) for use on foot: each hand part re-placed by ``foot_part`` into a
    ``<key>_foot.json`` beside it, the manifest's part geometry pointed at it (``on_foot``: the stance and frames).
    -> the manifest."""
    from eu4unitgen.blender.runner import BlenderRunner
    vdir = Path(vdir)
    m = json.loads((vdir / "manifest.json").read_text(encoding="utf-8"))
    stance = foot_stance(m)
    base_file = eu4() / "gfx" / "models" / "Units" / "western_generic_01_type1.mesh"
    base = base_file.read_bytes()
    anims = eu4() / "gfx" / "models" / "Units" / "animation"
    br = BlenderRunner(blender or _config().blender(), log_dir=vdir / "logs", timeout=900)
    node_of = {v: k for k, v in HAND_KIND.items()}
    frames = {}
    for key, part in (m.get("parts") or {}).items():
        node = node_of.get(part.get("kind"))
        if node is None:
            continue
        side = "R" if node.startswith("Right") else "L"
        for pose, anim in (("seated", "cavalry_rider_idle"), ("foot", f"{stance}_idle")):
            if (pose, node) not in frames:
                frames[(pose, node)] = br.run_job("rider_frames", {
                    "mesh": str(base_file), "anim": str(anims / f"{anim}.anim"), "node": node, "elbow": f"{side}_elbow",
                    "shoulder": f"{side}_shoulder", "bind": node_bind(base, node)}).result
        src = Path(part["mesh"]["geometry"])
        g = json.loads(src.read_text(encoding="utf-8"))
        if part["kind"] == "mesh_shield":  # strapped to the forearm in every frame it plays on foot and as crew
            per = []
            for anim in [f"{stance}_{x}" for x in FOOT_ANIMS]:
                f = anims / f"{anim}.anim"
                if f.is_file():
                    r = br.run_job("rider_frames", {"mesh": str(base_file), "anim": str(f), "node": node,
                                                    "elbow": f"{side}_elbow", "shoulder": f"{side}_shoulder",
                                                    "bind": node_bind(base, node), "frames": "all"}).result
                    per += [{**x, "anim": anim} for x in r.get("per_frame") or []]
            g["p"], strap_check = strap_part(g["p"], frames[("seated", node)], per)
            m.setdefault("on_foot_checks", {})[key] = strap_check
        else:
            g["p"] = foot_part(g["p"], part["kind"], frames[("seated", node)], frames[("foot", node)])
        dst = src.with_name(f"{key}_foot.json")
        dst.write_text(json.dumps(g), encoding="utf-8")
        part["mesh"]["seated_geometry"] = str(src)  # the rider's (ref3d.assemble seats the soldier on a horse)
        part["mesh"]["geometry"] = str(dst)
    m["on_foot"] = {"stance": stance, "frames": {f"{p}/{n}": {k: fr[k] for k in ("out", "up", "forward")}
                                                  for (p, n), fr in frames.items()}}
    (vdir / "manifest.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    return m


def _config():
    from eu4unitgen.core import config
    return config.load()


def strap(frame: dict, cfg: dict | None = None):
    """Where a shield sits on the forearm (``kite_part``'s rule) and its axes: (centre, {"across", "up", "out"})."""
    c = {"out": 1.2, "down": -0.3, **(cfg or {})}
    centre = [0.5 * frame["elbow"][k] + c["out"] * frame["out"][k] - c["down"] * frame["up"][k] for k in range(3)]
    return centre, {"across": frame["forward"], "up": frame["up"], "out": frame["out"]}


SHIELD_DARKEN = 0.12  # the user, test 34: a little darker than the armour


def shield_from_reference(br, parts_dir: Path, frame: dict, tall: float, crop: dict | None, pal: dict, primary,
                          common: dict, method: str, cache: Path, raw_part: Path) -> tuple[dict, dict]:
    """The shield (Track P1.2): TRELLIS.2's model of its crop, built and checked (``part_gen``), else the procedural
    kite with the crop's outline, emblem and colours, else (no crop) the default kite in the palette's colours.
    Either way its colours are moved onto the palette and it is ``SHIELD_DARKEN`` darker. -> (the loadout entry,
    what was tried and how each scored)."""
    from PIL import Image

    from eu4unitgen.core import gear
    from eu4unitgen.ref3d import part_gen
    record = {"tried": []}
    centre, axes = strap(frame)
    cloth = (pal.get("cloth") or {}).get("colour") or gear.KITE_COLOURS["field"]
    gold = (pal.get("gold") or {}).get("colour") if (pal.get("gold") or {}).get("share", 0) > 0.005 else gear.KITE_COLOURS["gold"]
    rows = []
    cim = Image.open(crop["crop"]).convert("RGBA") if crop and crop.get("crop") else None
    tc = part_gen.two_colours(cim) if cim is not None else None
    mapping = [(tc[0], cloth), (tc[1], gold)] if tc else [(gear.KITE_COLOURS["field"], cloth), (gear.KITE_COLOURS["gold"], gold)]
    if cim is not None and method == "auto":
        try:
            from eu4unitgen.ai.env import AIJobRunner  # noqa: F401 - the AI environment is needed for the crop path
            root = _config().ai_root_path()
            g = part_gen.generate({"shield": Path(crop["crop"])}, cache, root)["shield"]
            built = part_gen.build(br, Path(g["glb"]), Path(crop["crop"]), parts_dir / "gen", "shield", tall, False, common)
            fin = part_gen.finish_texture(Path(built["diffuse"]), Path(built["specular"]), mapping, primary, cloth,
                                          SHIELD_DARKEN, parts_dir / "gen" / "shield_final", PROCESS["gloss"])
            geo = json.loads(Path(built["geometry"]).read_text(encoding="utf-8"))
            look = part_gen.palette_crop(cim, mapping, SHIELD_DARKEN)  # the crop as the unit's palette shows it
            r, front = part_gen.check_turned(geo, Image.open(fin["diffuse"]), look)
            if tc:  # the emblem crisp, seen as the crop sees the shield; then checked again
                part_gen.project_emblem(Path(built["geometry"]), Path(fin["diffuse"]), tc[2], part_gen.inner_face(cim),
                                        part_gen.emblem_box(cim.getchannel("A")), r["yaw"],
                                        [int(v * (1 - SHIELD_DARKEN)) for v in gold],
                                        [int(v * (1 - SHIELD_DARKEN)) for v in cloth], Path(fin["diffuse"]))
                r, front = part_gen.check_turned(geo, Image.open(fin["diffuse"]), look)
            if tc:  # the emblem crisp (the user, 2026-10-25)
                r["emblem_iou"] = part_gen.emblem_match(front, look)
                if r["emblem_iou"] is not None and r["emblem_iou"] < part_gen.EMBLEM_MIN:
                    r["ok"] = False
            record["tried"].append({"method": "trellis", **r, "view": built["view"], "tris": built["tris"], "cached": g.get("cached")})
            rows.append(("shield: TRELLIS.2 from its crop", cim, front, r))
            if r["ok"]:
                part_gen.place(Path(built["geometry"]), axes, centre, tall, raw_part)
                record["method"] = "trellis"
                part_gen.check_sheet(rows, parts_dir / "part_checks.png")
                return ({"kind": "mesh_shield", **({"colours": {"own": "country1"}} if primary is not None else {}),
                         "mesh": {"geometry": str(raw_part), "diffuse": fin["diffuse"], "specular": fin["specular"],
                                  "normal": built["normal"]}}, record)
        except Exception as exc:  # noqa: BLE001 - the fallback below; recorded
            record["tried"].append({"method": "trellis", "error": f"{type(exc).__name__}: {exc}"})
    # the procedural kite, from the crop when there is one
    kw = {"colours": {"field": cloth, "gold": gold, "stud": tuple(min(255, int(v * 1.15)) for v in gold)}}
    if cim is not None:
        ol = part_gen.outline(cim, int(gear.KITE["sides"]))
        # the crop is foreshortened (the shield turned from the camera): its width as the raw model's own shield
        try:
            rp = json.loads(raw_part.read_text(encoding="utf-8"))["p"]
            q = [rp[i:i + 3] for i in range(0, len(rp), 3)]
            w3 = max(sum(a * b for a, b in zip(v, axes["across"])) for v in q) - min(sum(a * b for a, b in zip(v, axes["across"])) for v in q)
            h3 = max(sum(a * b for a, b in zip(v, axes["up"])) for v in q) - min(sum(a * b for a, b in zip(v, axes["up"])) for v in q)
            want = max(0.35, min(0.8, w3 / max(h3, 1e-6)))
        except (OSError, ValueError, KeyError):
            want = gear.KITE["aspect"]
        have = (max(x for x, _ in ol) - min(x for x, _ in ol)) or 1.0
        kx = want / have
        kw["outline_unit"] = [(x * kx, y) for x, y in ol]
        if tc:
            kw["emblem"] = {"mask": tc[2], "box": part_gen.emblem_box(cim.getchannel("A")), "stretch": kx}
        record["outline_aspect"] = round(want, 3)
    geo0, tex0 = gear.kite_shield(tall, [0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], primary=primary, **kw)
    entry = kite_part(parts_dir, frame, tall, primary=primary, cfg={"kite_kw": kw})
    if cim is not None:
        look = part_gen.palette_crop(cim, mapping, SHIELD_DARKEN)
        r, front = part_gen.check_turned({"p": geo0["p"], "tri": geo0["tri"], "uv": geo0["uv"]}, tex0["diffuse"], look)
        if tc:
            r["emblem_iou"] = part_gen.emblem_match(front, look)
        record["tried"].append({"method": "procedural (from the crop)", **r})
        rows.append(("shield: procedural, traced from its crop", cim, front, r))
        part_gen.check_sheet(rows, parts_dir / "part_checks.png")
    record["method"] = "procedural (from the crop)" if cim is not None else "procedural (default kite)"
    return entry, record


def tinted(diffuse: Path, specular: Path, primary, out: Path) -> Path:
    """The diffuse as the game draws it with its country colour ``primary`` (PdxMeshColor: lerp(col, col * spec.r *
    Primary, spec.r)): the masked cloth is stored grey for the game to tint, so a render of the plain diffuse showed the
    English soldier's red cross and hose grey. Also the "country colours off" look (the cloth keeps the picture's
    colour: ``primary`` the default)."""
    from PIL import Image, ImageChops
    d = Image.open(diffuse).convert("RGB")
    r = Image.open(specular).convert("RGBA").getchannel("R").resize(d.size)
    if r.getextrema()[1] == 0 or not primary:
        d.save(out)
        return out
    full = ImageChops.multiply(d, Image.new("RGB", d.size, tuple(int(c) for c in primary[:3])))
    once = Image.composite(ImageChops.multiply(full, Image.merge("RGB", (r, r, r))), d, r)  # lerp by r of col*r*P
    once.save(out)
    return out


REGIONS = (("top", 0.0, 0.2), ("middle", 0.2, 0.55), ("bottom", 0.55, 1.0))  # head; body; legs (of the outline's height)


def region_colours(raw: Path, game: Path) -> dict:
    """The raw model's and the game version's colours compared by region (one camera): in each height band of their
    outline, the median colour of the pixels both show, and the distance (CIE76) between them. -> {region: dE}."""
    from PIL import Image

    from eu4unitgen.core.palette import delta_e
    a, b = Image.open(raw).convert("RGBA"), Image.open(game).convert("RGBA")
    both = Image.composite(Image.new("L", a.size, 255), Image.new("L", a.size, 0),
                           Image.eval(a.getchannel("A"), lambda v: 255 if v > 127 else 0)).point(lambda v: v)
    both = Image.composite(both, Image.new("L", a.size, 0), Image.eval(b.getchannel("A"), lambda v: 255 if v > 127 else 0))
    box = both.getbbox()
    if not box:
        return {}
    out = {}
    ar, br_ = list(a.convert("RGB").getdata()), list(b.convert("RGB").getdata())
    mask = list(both.getdata())
    w = a.width
    for name, lo, hi in REGIONS:
        y0, y1 = int(box[1] + lo * (box[3] - box[1])), int(box[1] + hi * (box[3] - box[1]))
        idx = [y * w + x for y in range(y0, y1) for x in range(box[0], box[2], 2) if mask[y * w + x]]
        if len(idx) < 50:
            continue
        med = lambda px: [sorted(px[i][k] for i in idx)[len(idx) // 2] for k in range(3)]  # noqa: E731
        out[name] = round(delta_e(med(ar), med(br_)), 1)
    return out


RIDER_VIEWS = {"front": [0, 0], "side": [90, 0], "front34": [45, 12]}  # a person faces -Z; a horse is seen from the side


def compare_sheet(br, out: Path, basecolor: Path, game_texture: Path | None = None, views: dict | None = None) -> dict:
    """The raw model (fitted only) and the game version through one camera per view, their overlay, and their
    colours by region (``region_colours``). ``game_texture``: the game version as the game tints it (``tinted``)."""
    from PIL import Image, ImageChops, ImageDraw, ImageFont
    views = views or {"side": [90, 0], "front34": [45, 12], "rear34": [225, 12]}
    colour_de = {}
    game_tex = game_texture or out / "horseb_diffuse.png"
    r = br.run_job("render_compare", {"models": {"raw": {"obj": str(out / "fitted.obj"), "texture": str(basecolor)},
                                                 "game": {"obj": str(out / "low.obj"), "texture": str(game_tex)}},
                                      "views": views, "size": 900, "out_dir": str(out / "compare")}).result
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    C, P, H = 900, 10, 36
    sheet = Image.new("RGB", (P + 3 * (C + P), P + len(views) * (C + H + P)), (232, 232, 236))
    d = ImageDraw.Draw(sheet)
    iou = {}
    for i, v in enumerate(views):
        raw, game = Image.open(r["raw"][v]).convert("RGBA"), Image.open(r["game"][v]).convert("RGBA")
        a1 = raw.getchannel("A").point(lambda x: 255 if x > 127 else 0)
        a2 = game.getchannel("A").point(lambda x: 255 if x > 127 else 0)
        ov = Image.new("RGB", raw.size, (255, 255, 255))
        ov.paste((220, 40, 40), mask=ImageChops.subtract(a1, a2))
        ov.paste((30, 120, 230), mask=ImageChops.subtract(a2, a1))
        ov.paste((150, 150, 150), mask=ImageChops.multiply(a1, a2))
        both, union = ImageChops.multiply(a1, a2).histogram()[255], ImageChops.lighter(a1, a2).histogram()[255]
        iou[v] = round(both / max(union, 1), 3)
        colour_de[v] = region_colours(r["raw"][v], r["game"][v])
        y = P + i * (C + H + P)
        for j, (label, im) in enumerate([(f"raw TRELLIS.2, fitted only ({v})", raw), (f"game version ({v})", game),
                                         (f"overlay: grey both, red raw only, blue game only; IoU {iou[v]:.3f}", ov)]):
            cell = Image.new("RGBA", (C, C), (232, 232, 236, 255))
            cell.alpha_composite(im.convert("RGBA"))
            sheet.paste(cell.convert("RGB"), (P + j * (C + P), y + H))
            d.text((P + j * (C + P) + 4, y + 6), label, fill=(20, 20, 24), font=font)
    sheet.save(out / "compare_sheet.png")
    return {"sheet": str(out / "compare_sheet.png"), "iou": iou, "colour_de": colour_de,
            "images": {k: dict(r[k]) for k in ("raw", "game")}}


def deform(br, out: Path, base: bytes) -> dict:
    """The deformation check (check_mesh_deform) on the game mesh and vanilla's, idle, moving and attack."""
    from eu4unitgen.core.fresh_mesh import geometry_from_job, write_fresh
    dd = out / "deform"
    dd.mkdir(exist_ok=True)
    (dd / "horse.mesh").write_bytes(write_fresh(base, geometry_from_job(out / "low_geometry.json")))
    anims = [{"id": k, "path": str(eu4() / "gfx" / "models" / "Units" / "animation" / f"cavalry_horse_{k}.anim")}
             for k in ("idle", "moving", "attack")]
    res = {name: br.run_job("check_mesh_deform", {"mesh": str(m), "anims": anims,
                                                  "renders": {"out_dir": str(dd), "name": name, "size": 700} if name == "ours" else None}).result
           for name, m in (("ours", dd / "horse.mesh"), ("vanilla", eu4() / "gfx" / "models" / "Units" / "cavalry_horse.mesh"))}
    (dd / "deform.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return {who: {k: {"stretch": [round(a["str_p05"], 2), round(a["str_p95"], 2)], "max": round(a["str_max"], 2),
                      "flips": a["flips"]} for k, a in res[who]["anims"].items()} for who in res}


def stage(test: str, entries: list[tuple[Path, Path]], names: str) -> dict:
    """Game versions (``(version dir, subject.json)``: horses) as tiers 1, 2, ... of a new ``build/<test>`` (the next
    tier vanilla); unit names ``<names> ...``; the first's cloth red as the tag's primary when it has a mask."""
    from eu4unitgen.core import config
    from eu4unitgen.ref3d import unitstage
    folder = config.load().work() / "build" / test
    if folder.exists():
        raise SystemExit(f"build/{test} exists: pick a new test name")
    horses = {}
    colours = list(unitstage.TEST["unit_colours"])
    for k, (vd, subj) in enumerate(entries):
        h = unitstage.horse_entry(Path(vd), json.loads(Path(subj).read_text(encoding="utf-8")))
        horses[str(k + 1)] = h
        if k == 0 and h.get("unit_colours"):
            colours = h["unit_colours"]
    t = unitstage.TEST
    sp = {"schema": 1, "name": f"{test}: game horses at tiers 1-{len(horses)}; vanilla at tier {len(horses) + 1}",
          "set": {"tags": [t["tag"]], "gfx": t["gfx"], "set_name": t["set_name"], "base_gfx": t["base_gfx"],
                  "unit_colours": {t["tag"]: colours},
                  "unit_names": {u: f"{names} {u.title()}" for u in ("cavalry", "infantry", "artillery")}},
          "loadout": {"schema": 1, "horses": horses}, "test": {"sprite_techs": t["sprite_techs"]}}
    return unitstage.stage(folder, sp, test)


def run(image: Path, name: str, rights: str, ai_python: str, emblem: Path | None = None, country_mask: bool = False,
        test: str | None = None, rider: bool = False) -> dict:
    from eu4unitgen.ref3d.writeguard import ai_root
    folder = Path("assets_in") / name
    if (folder / "trellis").exists():
        raise SystemExit(f"{folder / 'trellis'} exists: pick a new name")
    root = ai_root(ai_python)
    if root is None:
        raise SystemExit(f"{ai_python}: not inside an AI root")
    t0 = time.time()
    tr = folder / "trellis"
    tr.mkdir(parents=True)
    subject = {"schema": 1, "name": name, "template": "horse", "rights": rights,
               "original": os.path.relpath(Path(image).resolve(), folder.resolve()), "pipeline": "trellis",
               "written_by": "ref3d.trellis run (M21: the standard horse pipeline)"}
    (folder / "subject.json").write_text(json.dumps(subject, indent=1), encoding="utf-8")
    rgba = cutout(Path(image), tr / "input", ai_python)
    gen = generate(rgba, (tr / "raw.glb").resolve(), root)
    if rider:  # M23: a rider (front view): the body and its lance and shield; staged with a horse by the caller
        m = process_rider(tr / "raw.glb", tr / "game", True)
        rep = {"generate": gen, "game": m["version"], "deform": m["deform"], "seconds": round(time.time() - t0, 1)}
        (tr / "run.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
        return rep
    m = process(tr / "raw.glb", tr / "game", emblem, country_mask)
    st = stage(test, [(Path(m["geometry"]).parent, folder / "subject.json")], test.upper()) if test else None
    rep = {"generate": gen, "game": m["version"], "deform": m["deform"], "compare": m["compare"], "test": st,
           "seconds": round(time.time() - t0, 1)}
    (tr / "run.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("image", type=Path)
    a.add_argument("--name", required=True)
    a.add_argument("--rights", required=True, help="the rights declaration (rule 8): who made the image and that you may use it")
    a.add_argument("--ai-python", required=True)
    a.add_argument("--emblem", type=Path)
    a.add_argument("--country-mask", action="store_true")
    a.add_argument("--test")
    a.add_argument("--rider", action="store_true", help="a rider (front view, lance and shield in hand), not a horse")
    b = sub.add_parser("process")
    b.add_argument("glb", type=Path)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--emblem", type=Path)
    b.add_argument("--country-mask", action="store_true")
    c = sub.add_parser("stage")
    c.add_argument("--test", required=True)
    c.add_argument("--names", required=True)
    c.add_argument("entries", nargs="+", help="<game version dir>@<subject.json>, tier 1 first")
    args = p.parse_args(argv)
    if args.cmd == "run":
        out = run(args.image, args.name, args.rights, args.ai_python, args.emblem, args.country_mask, args.test, args.rider)
    elif args.cmd == "process":
        m = process(args.glb, args.out, args.emblem, args.country_mask)
        out = {k: m[k] for k in ("version", "compare", "deform", "seconds")}
    else:
        out = stage(args.test, [tuple(Path(x) for x in e.rsplit("@", 1)) for e in args.entries], args.names)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
