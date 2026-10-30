"""The Example (the Help and Home screens' "Show me an example"): a finished unit walked through step by step with its
real results: the pictures, the cut-outs, TRELLIS.2's two candidates and their scores, the model fitted to the game,
its parts, the soldier on foot, the cavalry riding, the quality report, the turntable and Add to mod.

The example is the cataphract: three runs of the app's own flow (the horse, the soldier and the cavalry joining
them) in the projects folder. Its results are read from this computer's runs (renders of the game's own bodies are
never bundled); its pictures come with the app (``PICTURES``: the author's, bundling allowed 2026-10-28), so on any PC
the Example shows them and New unit can start from them. ``steps()`` -> the steps, or None when the runs aren't here."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

RUNS = {"horse": "cataphract_horse", "soldier": "cataphract", "cavalry": "cataphract_cavalry"}
PICTURES_DIR = Path(__file__).with_name("example_pictures")
PICTURES = {"soldier": [("the cataphract: a front view, A-pose", "cataphract_rider.png")],
            "horse": [("the cataphract's horse: a side view", "cataphract_side_clean.png"),
                      ("a light cavalry horse: a side view", "horse2_light_cavalry.png")]}
RIGHTS = ("Example picture bundled with the app: generated with ChatGPT from the app author's own original; the "
          "author owns it and allows its use (declared 2026-10-28)")


def pictures(kind: str) -> list[tuple[str, Path]]:
    """The example pictures for a kind (soldier, horse) that come with the app: (caption, path)."""
    return [(c, PICTURES_DIR / n) for c, n in PICTURES.get(kind, []) if (PICTURES_DIR / n).is_file()]


@dataclass
class Step:
    title: str
    text: str
    images: list[tuple[str, str]] = field(default_factory=list)  # (caption, path)
    lines: list[str] = field(default_factory=list)  # numbers and results in plain words
    frames: list[str] = field(default_factory=list)  # a turntable


def _json(p: Path) -> dict:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _have(paths) -> list[tuple[str, str]]:
    return [(c, str(p)) for c, p in paths if p and Path(p).is_file()]


def available(root: Path | None = None) -> bool:
    from eu4unitgen.ref3d import flow
    root = Path(root) if root else flow.root()
    return all((_json(root / n / "status.json")).get("done") for n in RUNS.values())


def steps(root: Path | None = None) -> list[Step] | None:
    from eu4unitgen.ref3d import flow
    root = Path(root) if root else flow.root()
    if not available(root):
        return None
    f = {k: root / n for k, n in RUNS.items()}
    req = {k: _json(p / "request.json") for k, p in f.items()}
    out = [Step("1. Two pictures",
                "A unit starts from pictures. Cavalry needs two: the horse from the side, and the soldier from the front "
                "with the arms a little out (an A-pose) so the arms, body, lance and shield can be told apart.",
                _have([("the horse: a side view", req["horse"].get("image")),
                       ("the soldier: a front view, A-pose", req["soldier"].get("image"))]),
                [f"Rights given for the horse: {req['horse'].get('rights', '?')}",
                 f"Rights given for the soldier: {req['soldier'].get('rights', '?')}"]),
           Step("2. Cut out",
                "The background is removed first (BiRefNet, an AI that finds the subject), so only the horse and the "
                "soldier become part of the models.",
                _have([("the horse, cut out", f["horse"] / "trellis" / "input" / "input_rgba.png"),
                       ("the soldier, cut out", f["soldier"] / "trellis" / "input" / "input_rgba.png")]))]
    # TRELLIS.2's two candidates per picture, scored on outline, colour and worst stretch
    imgs, lines = [], []
    for k in ("horse", "soldier"):
        c = flow.candidates(f[k]) or {}
        for i, cand in enumerate(c.get("candidates") or []):
            tag = "kept" if i == c.get("best") else "not used"
            imgs.append((f"the {k}, version {i + 1} ({tag})", f[k] / "trellis" / f"cand_s{cand['seed']}" / "colour_best.png"))
            lines.append(f"The {k}, version {i + 1}: {flow.candidate_text(cand)}; score {cand.get('score', cand.get('iou'))}"
                         + (" (kept)" if i == c.get("best") else ""))
    out.append(Step("3. Two 3D models each, and the better one kept",
                    "TRELLIS.2 guesses a 3D model from each picture. It makes two versions; the app scores each on how "
                    "well its outline and colours match the picture and how much it stretches when it moves on the "
                    "game's skeleton, and keeps the better one. You can pick the other later in the Library.",
                    _have(imgs), lines))
    # fitted onto the game's skeleton (the raw model against the game version, one camera)
    fitted = []
    for k, view in (("horse", "side"), ("soldier", "front")):
        st = _json(f[k] / "status.json")
        v = ((st.get("stages") or {}).get("build") or {}).get("version")
        m = _json(f[k] / "trellis" / "game" / v / "manifest.json") if v else {}
        im = ((m.get("compare") or {}).get("images") or {})
        fitted += [(f"the {k}: TRELLIS.2's model", (im.get("raw") or {}).get(view)),
                   (f"the {k}: the game version", (im.get("game") or {}).get(view))]
    out.append(Step("4. Made fit for the game",
                    "Each model is cut down to the size the game can draw (about 2,800 triangles), fitted onto the game's "
                    "own skeleton so it moves with the game's animations, and painted from the model's colours.",
                    _have(fitted)))
    q = _json(f["soldier"] / "preview" / "quality.json")
    parts = ((q.get("measures") or {}).get("parts") or {}).get("text")
    sv = ((_json(f["soldier"] / "status.json").get("stages") or {}).get("build") or {}).get("version")
    crops = f["soldier"] / "trellis" / "game" / (sv or "-") / "parts" / "crops"
    out.append(Step("5. Shield, lance and plume as their own parts",
                    "Thin or separate things (a shield, a lance, a plume) are cut out of the picture one by one, made "
                    "as parts of their own and fixed to the hand or head, so they stay crisp and move right.",
                    _have([(f"cut out of the picture: the {p}", r.get("crop"))
                           for p, r in _json(crops / "crops.json").items() if isinstance(r, dict)]),
                    [parts] if parts else []))
    g = f["soldier"] / "preview" / "gallery"
    out.append(Step("6. The soldier on foot",
                    "On foot the soldier is your infantry (and the crew of your guns). It is drawn here the way the "
                    "game draws it, next to the game's own infantry.",
                    _have([("close, three-quarter", g / "close_three_quarter_1.png"), ("close, side", g / "close_side_1.png"),
                           ("at map zoom", g / "map_three_quarter_1.png")])))
    g = f["cavalry"] / "preview" / "gallery"
    out.append(Step("7. Riding the horse",
                    "The cavalry unit puts the soldier on the horse. The soldier is built again for this horse: its "
                    "legs move out to clear the horse's width, and its shield takes the horse's colours.",
                    _have([("close, three-quarter", g / "close_three_quarter_1.png"), ("close, side", g / "close_side_1.png"),
                           ("at map zoom", g / "map_three_quarter_1.png")])))
    q = _json(f["cavalry"] / "preview" / "quality.json")
    from eu4unitgen.ref3d.quality import TITLES
    out.append(Step("8. The quality report",
                    "Every unit gets a report in plain words: how close it is to the picture, how it moves compared with "
                    "the game's own, whether parts were kept or rebuilt, and how it looks at map zoom.",
                    _have([("the result sheet", f["cavalry"] / "preview" / "result_sheet.png")]),
                    [q.get("summary", "")] + [f"{TITLES.get(k, k)}: {x.get('text', '')}" for k, x in (q.get("measures") or {}).items()]))
    out.append(Step("9. Turn it round",
                    "The Library shows every finished unit turning round, as the game draws it.",
                    frames=[str(p) for p in flow.turntable_frames(f["cavalry"])]))
    out.append(Step("10. Add to mod",
                    "Last, Add to mod writes the unit into a mod folder you choose: a normal mod (a new unit for one "
                    "country, with its own stats), an achievement-safe one (graphics only: it re-dresses a graphical "
                    "culture's unit and leaves the game's checksum alone), or a culture pack (a new graphical culture "
                    "any nation can be given). Every export writes a README.txt listing each file, what the game uses "
                    "it for and how to install it. If Windows blocks the mod folder, the app saves it elsewhere and "
                    "tells you exactly what to copy."))
    out.append(Step("11. Try it yourself",
                    "This unit comes with the app: the Library's “Byzantine cataphract cavalry” is the cataphract "
                    "riding the cataphract horse, for Byzantium, built from the default library on first open (a few "
                    "minutes, no AI). Open it, change a tier or its colours, and press Add to mod to see the whole "
                    "export and its README."))
    return out
