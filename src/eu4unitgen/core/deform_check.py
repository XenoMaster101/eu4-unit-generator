"""Automatic deformation check of derived bodies (SPEC Track R; the spike R0 criteria as a pipeline step).

The Blender job ``check_deform`` plays the infantry's own idle, walk and attack animations
(base-game .anim files) on the derived body and measures every frame for each skinned addition.
``judge`` turns the results into validator findings:

  C1 stretch: every edge within 0.85-1.20 of its bind length, p05..p95 within 0.92-1.08   (error)
  C2 penetration into the base body, outside exempt rings: idle/walk none deeper than 0.10;
     attack <= 3% of vertices and none deeper than 0.30                                     (error)
  C3 no triangle flipped (> 90 degrees from bind, in the root bone's frame)                 (error)
  C4 root drift <= 0.05 in the root bone's frame                                            (warning)
  C5 stays behind its reference joint in the unit's facing frame (min > 0)                  (warning)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

ERROR, WARNING = "error", "warning"
STRETCH = (0.85, 1.20)
STRETCH_CORE = (0.92, 1.08)
PEN_DEPTH = 0.10
ATTACK_PEN = (3.0, 0.30)  # percent of vertices, depth
DRIFT = 0.05
STATES = ("idle", "walk", "attack")


def anim_files(pdxmesh: dict, idx) -> list[tuple[str, str]]:
    """(state, .anim path) for the pdxmesh's idle, walk and attack animation types."""
    out = []
    for state in STATES:
        t = (pdxmesh.get("animations") or {}).get(state)
        rec = idx.animation_type(t, base_only=True) if t else None
        if rec and rec.get("file"):
            out.append((state, rec["file"]))
    return out


def run_check(runner, mesh: bytes, base_vertices: int, adds: list, kinds: dict, anims: list[tuple[str, bytes]],
              renders: dict | None = None) -> dict:
    """Run ``check_deform`` on ``mesh`` (derived body bytes). ``adds``: core.derived_body.Addition in
    file order (with rings); ``kinds``: ADDITION_KINDS; ``anims``: (state, .anim bytes)."""
    specs, start = [], base_vertices
    for a in adds:
        k = kinds[a.name]
        specs.append({"name": a.name, "start": start, "count": a.count, "rings": list(a.rings),
                      "root": a.chain[0], "behind": k.get("behind"), "exempt": list(k.get("exempt", [0]))})
        start += a.count
    with tempfile.TemporaryDirectory(prefix="eu4unitgen_deform_") as tmp:
        body = Path(tmp) / "derived.mesh"
        body.write_bytes(mesh)
        paths = []
        for state, data in anims:
            p = Path(tmp) / f"{state}.anim"
            p.write_bytes(data)
            paths.append({"id": state, "path": str(p)})
        return runner.run_job("check_deform", {"body": str(body), "base_vertices": base_vertices, "additions": specs,
                                               "anims": paths, "renders": renders}, timeout=900).result


def judge(result: dict, accept: list[str] | tuple[str, ...] = ()) -> list[tuple[str, str]]:
    """(severity, message) for every criterion an addition fails in an animation. Kinds in
    ``accept`` (the loadout's accept_clipping) get their C2 findings as warnings: a known limit."""
    out = []
    for state, res in sorted((result or {}).get("anims", {}).items()):
        for name, m in sorted(res["additions"].items()):
            where = f"{name} in {state} (worst frame {m['worst_frame']})"
            if not (STRETCH[0] <= m["str_min"] and m["str_max"] <= STRETCH[1]
                    and STRETCH_CORE[0] <= m["str_p05"] and m["str_p95"] <= STRETCH_CORE[1]):
                out.append((ERROR, f"C1 {where}: stretch {m['str_min']:.2f}-{m['str_max']:.2f} "
                                   f"(p05-p95 {m['str_p05']:.2f}-{m['str_p95']:.2f})"))
            c2 = (WARNING, " (accepted as a known limit: loadout accept_clipping)") if name in accept else (ERROR, "")
            if state == "attack":
                if m["pen_pct"] > ATTACK_PEN[0] or m["pen_depth"] > ATTACK_PEN[1]:
                    out.append((c2[0], f"C2 {where}: {m['pen_pct']:.1f}% of vertices inside the body, "
                                       f"up to {m['pen_depth']:.2f} deep{c2[1]}"))
            elif m["pen_depth"] > PEN_DEPTH:
                out.append((c2[0], f"C2 {where}: a vertex {m['pen_depth']:.2f} inside the body{c2[1]}"))
            if m["flips"]:
                out.append((ERROR, f"C3 {where}: {m['flips']} triangles flip"))
            if m["drift"] > DRIFT:
                out.append((WARNING, f"C4 {where}: the root drifts {m['drift']:.3f} from its bone"))
            if m["behind"] is not None and m["behind"] <= 0:
                out.append((WARNING, f"C5 {where}: passes in front of its joint ({m['behind']:.2f})"))
    return out


def summary(result: dict) -> str:
    """One line per animation for notes: the worst numbers over all additions."""
    lines = []
    for state, res in sorted((result or {}).get("anims", {}).items()):
        ms = res["additions"].values()
        if not ms:
            continue
        lines.append(f"{state}: stretch {min(m['str_min'] for m in ms):.2f}-{max(m['str_max'] for m in ms):.2f}, "
                     f"penetration up to {max(m['pen_depth'] for m in ms):.2f}, "
                     f"flips {max(m['flips'] for m in ms)}, drift {max(m['drift'] for m in ms):.3f}")
    return "; ".join(lines)
