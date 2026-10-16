"""Spike R0 report (app venv): contact sheets, map-scale grid, worst-frame sheets and the
metrics table (pass/fail per SPEC Track R criteria 1-5) from build/spike_r0/result.json."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[2] / "build" / "spike_r0"
if len(sys.argv) > 1:
    OUT = OUT / sys.argv[1]  # e.g. rerun_ponytail
SHORT = {"polearm_twohanded_idle": "idle", "polearm_twohanded_moving": "walk", "polearm_twohanded_attack": "attack"}
VARIANT = {"A": "A auto", "B": "B rigid", "C": "C gradient", "D": "D short rigid", "E": "E long chain"}


def criteria(anim: str, part: str, rows: list[dict]) -> dict:
    agg = {
        "str_min": min(r["str_min"] for r in rows), "str_max": max(r["str_max"] for r in rows),
        "p05": min(r["str_p05"] for r in rows), "p95": max(r["str_p95"] for r in rows),
        "pen_depth": max(r["pen_depth"] for r in rows), "pen_pct": max(r["pen_pct"] for r in rows),
        "flips": max(r["flips"] for r in rows), "drift": max(r["root_drift"] for r in rows),
        "behind": min(r["behind"] for r in rows),
        "elev": (min(r["tip_elev"] for r in rows), max(r["tip_elev"] for r in rows)),
    }
    attack = SHORT[anim] == "attack"
    agg["c1"] = 0.85 <= agg["str_min"] and agg["str_max"] <= 1.20 and 0.92 <= agg["p05"] and agg["p95"] <= 1.08
    agg["c2"] = (agg["pen_pct"] <= 3 and agg["pen_depth"] <= 0.30) if attack else agg["pen_depth"] <= 0.10
    agg["c3"] = agg["flips"] == 0
    agg["c4"] = agg["drift"] <= 0.05
    agg["c5"] = agg["behind"] > 0
    agg["pass"] = all(agg[f"c{i}"] for i in range(1, 6))
    return agg


def label(img: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + 18), (255, 255, 255))
    out.paste(img, (0, 18))
    ImageDraw.Draw(out).text((4, 3), text, fill=(0, 0, 0))
    return out


def grid(cells: list[list[Image.Image]], pad: int = 4) -> Image.Image:
    w = max(c.width for row in cells for c in row)
    h = max(c.height for row in cells for c in row)
    cols = max(len(r) for r in cells)
    sheet = Image.new("RGB", (cols * (w + pad) + pad, len(cells) * (h + pad) + pad), (255, 255, 255))
    for i, row in enumerate(cells):
        for j, c in enumerate(row):
            sheet.paste(c, (pad + j * (w + pad), pad + i * (h + pad)))
    return sheet


def main() -> None:
    res = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    (OUT / "sheets").mkdir(exist_ok=True)
    table, overall = [], {}
    for anim, r in res["anims"].items():
        for key, rows in r["per_frame"].items():
            part, var = key.rsplit("_", 1)
            agg = criteria(anim, part, rows)
            table.append((part, var, SHORT[anim], agg))
            overall.setdefault((part, var), []).append(agg["pass"])
        # contact sheet per variant: rows = frames, columns = front / side / rear
        for var, c in r["renders"]["contact"].items():
            files, views = c["files"], c["views"]
            cells = []
            for i, f in enumerate(c["frames"]):
                cells.append([label(Image.open(files[i * len(views) + k]).convert("RGB"), f"{SHORT[anim]} f{f} {v}")
                              for k, v in enumerate(views)])
            grid(cells).save(OUT / "sheets" / f"contact_{SHORT[anim]}_{var}.png")
    anims = list(res["anims"])
    variants = sorted(res["anims"][anims[0]]["renders"]["map"])
    cells = []
    for a in anims:
        for mv in ("front_quarter", "rear_quarter"):
            row = []
            for v in variants:
                im = Image.open(res["anims"][a]["renders"]["map"][v][mv]).convert("RGB")
                row.append(label(im.resize((im.width * 4, im.height * 4), Image.NEAREST),
                                 f"{SHORT[a]} {VARIANT[v]} {mv.split('_')[0]} (x4)"))
            for v in variants:
                row.append(label(Image.open(res["anims"][a]["renders"]["map"][v][mv]).convert("RGB"), "48px"))
            cells.append(row)
    grid(cells).save(OUT / "sheets" / "map_scale.png")
    # worst frame per object: rows = animation, columns = views (half size)
    for key in res["anims"][anims[0]]["renders"]["worst"]:
        part, v = key.rsplit("_", 1)
        cells = []
        for a in anims:
            w = res["anims"][a]["renders"]["worst"][key]
            cells.append([label(Image.open(f).convert("RGB").resize((240, 240), Image.LANCZOS),
                                f"{VARIANT[v]} {SHORT[a]} f{w['frame']} {view}")
                          for f, view in zip(w["files"], ("front", "side", "rear"))])
        grid(cells).save(OUT / "sheets" / f"worst_{key}.png")
    # metrics table
    yes = lambda b: "pass" if b else "**FAIL**"  # noqa: E731
    lines = ["| Part | Variant | Anim | Stretch min-max (p05-p95) | Penetration max depth / % verts | Flips | "
             "Root drift | Behind (min) | Tip elevation | C1 | C2 | C3 | C4 | C5 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for part, var, anim, g in sorted(table, key=lambda t: (t[0] != "tail", t[0], t[1], ["idle", "walk", "attack"].index(t[2]))):
        lines.append(f"| {part} | {VARIANT[var]} | {anim} | {g['str_min']:.3f}-{g['str_max']:.3f} "
                     f"({g['p05']:.3f}-{g['p95']:.3f}) | {g['pen_depth']:.3f} / {g['pen_pct']:.1f}% | {g['flips']} | "
                     f"{g['drift']:.3f} | {g['behind']:.2f} | {g['elev'][0]:.0f}..{g['elev'][1]:.0f} deg | "
                     f"{yes(g['c1'])} | {yes(g['c2'])} | {yes(g['c3'])} | {yes(g['c4'])} | {yes(g['c5'])} |")
    lines += ["", "| Part | Variant | All 3 animations pass C1-C5 |", "|---|---|---|"]
    for (part, var), ok in overall.items():
        lines.append(f"| {part} | {VARIANT[var]} | {'yes' if all(ok) else 'no'} |")
    w = res["anims"][anims[0]]["weights"]
    lines += ["", "| Object | Tris | Average weight per bone |", "|---|---|---|"]
    for k, info in w.items():
        lines.append(f"| {k} | {info['tris']} | " + ", ".join(f"{b} {x:.2f}" for b, x in info["groups"].items()) + " |")
    (OUT / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
