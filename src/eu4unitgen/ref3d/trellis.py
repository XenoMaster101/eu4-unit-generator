"""The standard horse pipeline (SPEC M21): one image -> TRELLIS.2 -> processing -> the EU4 game version.

  run      the whole chain for one image, into ``assets_in/<name>/trellis/``:
             1. cut-out: the horse masked by BiRefNet (ai job ref3d_mask, the Windows AI environment), RGBA;
             2. TRELLIS.2 in WSL2 (ai job trellis2_generate in ``~/eu4unitgen_ai/trellis2_env``, built by
                tools/wsl/trellis2_setup.sh; the weights read from the AI root through /mnt): ``raw.glb``;
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
import struct
import subprocess
import time
from pathlib import Path, PureWindowsPath

from eu4unitgen.ref3d import s1

# TRELLIS.2 in WSL (SPEC M21): the environment on the Linux disk, the checkout and weights on the AI root via /mnt
WSL = {"python": "~/eu4unitgen_ai/trellis2_env/bin/python", "repo": "TRELLIS.2", "weights": "weights/TRELLIS.2-4B",
       "ss_decoder": "weights/TRELLIS-image-large", "dinov3": "weights/dinov3-vitl16-pretrain-lvd1689m",
       "seed": 42, "pipeline_type": "1024_cascade"}
PROCESS = {"target_tris": 2800, "smooth": 12, "influences": 3, "chain_weights": True, "uv_margin": 0.002,
           "hair": {"mane": True, "tail": True}, "size": 1024,
           "gloss": {"silver": 14, "gold": 12, "matte": 5},
           "bake": {"cage": 0.3, "distance": 0.6, "margin": 8},
           "normal": {"cage": 0.1, "distance": 0.35, "high_smooth": 0, "strength": 0.6, "samples": 32, "ao_strength": 0.35}}


def wsl_path(p: Path) -> str:
    """A Windows path as WSL sees it (C:\\x -> /mnt/c/x)."""
    w = PureWindowsPath(p)
    if not w.drive:
        w = PureWindowsPath(os.path.abspath(p))
    return "/mnt/" + w.drive[0].lower() + w.as_posix()[2:]


def eu4() -> Path:
    p = os.environ.get("EU4_PATH")
    if not p:
        raise SystemExit("set EU4_PATH to the EU4 install (the vanilla horse is read from it)")
    return Path(p)


def cutout(image: Path, out: Path, ai_python: str) -> Path:
    """The horse cut out (BiRefNet: ai job ref3d_mask) as an RGBA png: TRELLIS.2 takes the alpha as is."""
    from PIL import Image

    from eu4unitgen.ai.env import AIJobRunner
    from eu4unitgen.ref3d.auto import ai_paths
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
    w = {**WSL, **(wsl or {})}
    job = Path(__file__).resolve().parents[1] / "ai" / "jobs" / "trellis2_generate.py"
    result = glb.with_suffix(".result.json")
    args = {"image": wsl_path(rgba), "output": wsl_path(glb), "result_path": wsl_path(result),
            "repo": wsl_path(ai_root / w["repo"]), "weights": wsl_path(ai_root / w["weights"]),
            "ss_decoder": wsl_path(ai_root / w["ss_decoder"]), "dinov3": wsl_path(ai_root / w["dinov3"]),
            "seed": w["seed"], "pipeline_type": w["pipeline_type"]}
    args_path = glb.with_suffix(".args.json")
    args_path.write_text(json.dumps(args, indent=1), encoding="utf-8")
    log = glb.with_suffix(".log")
    cmd = f"cd /tmp && {w['python']} {wsl_path(job)} {wsl_path(args_path)} > {wsl_path(log)} 2>&1"
    rc = subprocess.run(["wsl.exe", "-e", "bash", "-lc", cmd]).returncode
    if rc != 0 or not result.is_file():
        raise SystemExit(f"TRELLIS.2 failed (exit {rc}); see {log}")
    return json.loads(result.read_text(encoding="utf-8"))


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


def process(glb: Path, root: Path, emblem: Path | None = None, country_mask: bool = False, blender: str | None = None) -> dict:
    """``glb`` (TRELLIS.2) into ``root/vNNN`` (module doc). -> the manifest."""
    from eu4unitgen.blender.runner import BlenderRunner
    from eu4unitgen.core.fresh_mesh import base_arrays
    from eu4unitgen.core.project import default_blender_path
    out = s1.next_version(Path(root)).resolve()
    br = BlenderRunner(blender or default_blender_path(), log_dir=out / "logs", timeout=3600)
    base = (eu4() / "gfx" / "models" / "Units" / "cavalry_horse.mesh").read_bytes()
    van = base_arrays(base)
    t0, P = time.time(), PROCESS
    basecolor = glb_basecolor(glb, out / "raw_basecolor.png")
    fr = br.run_job("fresh_horse", {
        "source": {"kind": "direct", "path": str(Path(glb).resolve()), "merge": 1e-5},
        **{k: P[k] for k in ("target_tris", "smooth", "influences", "chain_weights", "uv_margin", "hair")},
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
            "part_colours": {"6": hair["tail_colour_p25_p75"]} if hair.get("tube") else {},
            "gloss": P["gloss"], "head_band": {"poll": poll, "axis": axis},
            "head_marks": {"poll": poll, "axis": axis, "neck_base": van["bones"]["HorseNeckBase"], "length": length,
                           "ear_box": [[poll[0] - 1.2, poll[1] - 0.3, poll[2] - 1.2], [poll[0] + 1.2, poll[1] + 3.0, poll[2] + 1.2]]},
            "bones": van["bones"], "country_mask": bool(country_mask), "colour_render": raw["images"]["side"],
            "size": P["size"], **P["bake"], "out_dir": str(out), "name": "horse"}
    if emblem:
        args["emblem"] = {"decal": str(Path(emblem).resolve()), "size": 0.7}
    bc = br.run_job("bake_colour", args).result
    nb = br.run_job("bake_horse", {"low": str(out / "low_geometry.json"), "high": [str(out / "textured.obj")],
                                   "diffuse": bc["diffuse"], "size": P["size"], **P["normal"], "preview": False,
                                   "out_dir": str(out), "name": "horseb"}).result
    tex = {"diffuse": str(out / "horseb_diffuse.png"), "specular": bc["specular"], "normal": str(out / "horseb_normal.png")}
    manifest = {"stage": "game version (M21 standard horse pipeline)", "glb": str(glb), "version": out.name,
                "fresh": {k: fr.get(k) for k in ("fit", "pose", "hair", "final", "back", "uv", "stages")},
                "colour": bc, "normal": {k: nb.get(k) for k in ("tilt", "match")}, "textures": tex,
                "geometry": str(out / "low_geometry.json"), "seconds": round(time.time() - t0, 1)}
    manifest["compare"] = compare_sheet(br, out, basecolor)
    manifest["deform"] = deform(br, out, base)
    manifest["seconds"] = round(time.time() - t0, 1)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    return manifest


def compare_sheet(br, out: Path, basecolor: Path) -> dict:
    """The raw model (fitted only) and the game version through one camera per view, and their overlay."""
    from PIL import Image, ImageChops, ImageDraw, ImageFont
    views = {"side": [90, 0], "front34": [45, 12], "rear34": [225, 12]}
    r = br.run_job("render_compare", {"models": {"raw": {"obj": str(out / "fitted.obj"), "texture": str(basecolor)},
                                                 "game": {"obj": str(out / "low.obj"), "texture": str(out / "horseb_diffuse.png")}},
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
        y = P + i * (C + H + P)
        for j, (label, im) in enumerate([(f"raw TRELLIS.2, fitted only ({v})", raw), (f"game version ({v})", game),
                                         (f"overlay: grey both, red raw only, blue game only; IoU {iou[v]:.3f}", ov)]):
            cell = Image.new("RGBA", (C, C), (232, 232, 236, 255))
            cell.alpha_composite(im.convert("RGBA"))
            sheet.paste(cell.convert("RGB"), (P + j * (C + P), y + H))
            d.text((P + j * (C + P) + 4, y + 6), label, fill=(20, 20, 24), font=font)
    sheet.save(out / "compare_sheet.png")
    return {"sheet": str(out / "compare_sheet.png"), "iou": iou}


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
    """Game versions (``(version dir, subject.json)``) as tiers 1, 2, ... of ``test``; unit names ``<names> ...``;
    the first's cloth red as the tag's primary when it has a country-colour mask."""
    from eu4unitgen.ref3d import game
    horses, prim = [], None
    for k, (vd, subj) in enumerate(entries):
        m = json.loads((Path(vd) / "manifest.json").read_text(encoding="utf-8"))
        s = json.loads(Path(subj).read_text(encoding="utf-8"))
        horses.append({"base": "cavalry_horse", "geometry": str(Path(m["geometry"]).resolve()),
                       "textures": {k2: str(Path(v).resolve()) for k2, v in m["textures"].items()},
                       "rights": s["rights"], "source": f"TRELLIS.2 (M21) from {s.get('original', '?')}, {Path(vd).name}",
                       "_name": s["name"]})
        if k == 0:
            prim = m["colour"].get("default_primary")
    cfg = dict(game.GAME_DEFAULTS)
    t = dict(cfg["test"], unit_names={u: f"{names} {u.title()}" for u in ("cavalry", "infantry", "artillery")})
    if prim:
        t["unit_colours"] = [prim] + t["unit_colours"][1:]
    cfg["test"] = t
    return game.stage_test(test, horses, cfg)


def run(image: Path, name: str, rights: str, ai_python: str, emblem: Path | None = None, country_mask: bool = False,
        test: str | None = None) -> dict:
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
        out = run(args.image, args.name, args.rights, args.ai_python, args.emblem, args.country_mask, args.test)
    elif args.cmd == "process":
        m = process(args.glb, args.out, args.emblem, args.country_mask)
        out = {k: m[k] for k in ("version", "compare", "deform", "seconds")}
    else:
        out = stage(args.test, [tuple(Path(x) for x in e.rsplit("@", 1)) for e in args.entries], args.names)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
