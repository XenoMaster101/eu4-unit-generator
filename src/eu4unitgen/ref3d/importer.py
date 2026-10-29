"""Import a model (the user, 2026-10-29): Make a part from a 3D model instead of a picture. A .glb (preferred) or
.fbx takes TRELLIS.2's place; everything after it (the fit, weights, parts, bake, checks) runs as usual.

  licence   one of ``LICENCES`` (CLAUDE.md rule 8: every model the app builds from is the user's own or licensed
            for the use): CC0, CC BY (its credit required: who made it and where it's from) or the user's own work.
            Kept in the run's request and subject (``source``), shown as its rights, and a CC BY model's credit is
            written into the mod with it (``export`` writes ``eu4unitgen_credits.txt``)
  stage     ``import`` (in place of cut-out and TRELLIS.2): the model copied into the run, made into what TRELLIS.2
            hands the build (``blender/jobs/prepare_import``: one mesh, one base-colour texture, a soldier stood up
            and facing the front, the unit cube) as ``trellis/raw.glb``, and its render from the kind's view as the
            reference picture (``trellis/input``: the part crops, the outline check and the result sheet read it)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

FORMATS = (".glb", ".fbx")
LICENCES = {"cc0": "CC0 (public domain)", "cc_by": "CC BY (credit the author)", "own": "My own work"}
MAX_MB = 300


def problem(model: Path | None, licence: str | None, credit: str = "") -> str:
    """Why the model can't be used yet ('' when it can)."""
    if model is None:
        return "Pick a model file (.glb or .fbx)."
    model = Path(model)
    if model.suffix.lower() not in FORMATS:
        return f"{model.name} isn't a .glb or .fbx file."
    if not model.is_file():
        return f"{model} isn't there."
    if model.stat().st_size > MAX_MB * 1024 * 1024:
        return f"{model.name} is over {MAX_MB} MB."
    if licence not in LICENCES:
        return "Say what licence the model has (CC0, CC BY or your own work)."
    if licence == "cc_by" and len(credit.strip()) < 3:
        return "A CC BY model needs its credit: who made it and where it's from."
    return ""


def metadata(model: Path) -> dict:
    """What a .glb says about itself (``asset.extras``: Sketchfab writes title, author, licence and source) and the
    licence and credit it suggests: {"title", "author", "license", "source", "licence": key or None, "credit"}.
    Only a suggestion: the user confirms it (a model's own label can be wrong)."""
    import struct
    out = {"licence": None, "credit": ""}
    try:
        data = Path(model).read_bytes()[:4 << 20]
        n = struct.unpack_from("<I", data, 12)[0]
        ex = (json.loads(data[20:20 + n]).get("asset") or {}).get("extras") or {}
    except (OSError, ValueError, struct.error):
        return out
    out.update({k: str(ex[k]) for k in ("title", "author", "license", "source") if ex.get(k)})
    lic = out.get("license", "").upper().replace(" ", "")
    import re  # only plain CC BY (not BY-SA, BY-NC, BY-ND: not the choices offered)
    out["licence"] = "cc0" if lic.startswith("CC0") else "cc_by" if re.match(r"CC-?BY(-?\d\.\d)?(\(|$)", lic) else None
    bits = [f"“{out['title']}”" if out.get("title") else "", f"by {out['author']}" if out.get("author") else "",
            out.get("source", ""), out.get("license", "")]
    out["credit"] = ", ".join(b for b in bits if b)
    return out


def rights_text(licence: str, credit: str = "") -> str:
    """The run's rights statement from its licence."""
    credit = credit.strip()
    if licence == "cc0":
        return "3D model under CC0 (public domain)" + (f": {credit}" if credit else "")
    if licence == "cc_by":
        return f"3D model under CC BY, credit: {credit}"
    return "3D model: my own work" + (f" ({credit})" if credit else "")


def source_record(model: Path, licence: str, credit: str = "") -> dict:
    """The request's ``source``: where the model came from and its licence (rule 8)."""
    return {"model": str(Path(model).resolve()), "file": Path(model).name, "format": Path(model).suffix.lower()[1:],
            "licence": licence, "licence_text": LICENCES[licence], "credit": credit.strip()}


def is_import(req: dict) -> bool:
    return bool((req.get("source") or {}).get("model"))


def stage_import(folder: Path, req: dict, cfg) -> dict:
    """The model into the run (module doc). -> the stage's record (the prepare job's result)."""
    from PIL import Image

    from eu4unitgen.blender.runner import BlenderRunner
    folder = Path(folder).resolve()  # Blender writes from its own working folder
    tr, inp = folder / "trellis", folder / "trellis" / "input"
    if (tr / "raw.glb").is_file() and (inp / "input_fg.png").is_file() and (tr / "import.json").is_file():
        return {"kept": True}
    src = req["source"]
    keep = tr / f"imported{Path(src['model']).suffix.lower()}"  # the run never depends on a file that may move
    if not keep.is_file():
        shutil.copyfile(src["model"], keep)
    br = BlenderRunner(cfg.blender(), log_dir=tr / "logs", timeout=1800)
    turn = (req.get("options") or {}).get("turn")
    res = br.run_job("prepare_import", {"model": str(keep), "out_glb": str(tr / "raw.glb"), "kind": req["kind"],
                                        "turn": turn, "out_dir": str(inp), "size": 1024, "bake_size": 2048}).result
    rgba = Image.open(inp / "input_rgba.png").convert("RGBA")
    rgba.getchannel("A").save(inp / "input_fg.png")
    white = Image.new("RGB", rgba.size, (255, 255, 255))
    white.paste(rgba, mask=rgba.getchannel("A"))
    white.save(inp / "input_rgb.png")
    (tr / "import.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    r = json.loads((folder / "request.json").read_text(encoding="utf-8"))
    r["image"] = str((inp / "input_rgb.png").resolve())  # its picture: the render (the Library, the result sheet)
    (folder / "request.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    return {k: res.get(k) for k in ("triangles", "materials", "baked", "rig_applied", "stood_up", "turn", "turn_from")}


def credit_line(name: str, source: dict) -> str:
    return f"{name}: {source.get('file')} - {rights_text(source.get('licence', 'own'), source.get('credit', ''))}"


def credits_for(req: dict) -> tuple[list[str], bool]:
    """The credit lines a mod owes for a run (a part, a cavalry run's horse and soldier, a unit's parts at every
    tier): one per imported model (CC0 and own work listed too, so the credits say where every model came from).
    -> (lines, whether any is CC BY: then the credits must ship with the mod)."""
    from eu4unitgen.ref3d import flow
    subjects = []
    if req.get("kind") == "unit":
        from eu4unitgen.ref3d import assemble
        for slots in (req.get("tiers") or {}).values():
            for ref in slots.values():
                if ref:
                    try:
                        subjects.append((ref.split(":", 1)[1], assemble.part_dir(ref)[1]))
                    except (FileNotFoundError, ValueError, OSError):
                        continue
    elif req.get("kind") == "cavalry":
        for n in (req.get("horse"), req.get("soldier")):
            p = flow.folder_of(n) / "subject.json" if n else None
            if p and p.is_file():
                subjects.append((n, json.loads(p.read_text(encoding="utf-8"))))
    else:
        subjects.append((req.get("name"), req))
    lines, by = [], False
    for name, s in subjects:
        src = s.get("source") or {}
        if src.get("model"):
            line = credit_line(name, src)
            if line not in lines:
                lines.append(line)
            by = by or src.get("licence") == "cc_by"
    return lines, by


CREDITS_FILE = "eu4unitgen_credits.txt"


def credits_text(lines: list[str]) -> bytes:
    head = "3D models this mod's units were made from (CC BY models need this credit kept with the mod)"
    return (head + "\n\n" + "\n".join(lines) + "\n").encode("utf-8")
