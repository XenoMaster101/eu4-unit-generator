"""Checks before presentation (the user, 2026-10-29: "if I were to give a silly or serious model it can do 1-3 checks
to make sure it's good before presentation"). The last stage of a person's run (soldier, rider): the build is
measured, a failure with a known fix gets that fix as a build option and the build, finish and preview run again;
at most ``PASSES`` passes; then the result is shown with what was checked, fixed and accepted.

  profile   ``standard`` (a person in the game's proportions) or ``stylised`` (a head over 0.30 of the height, from
            the fit or the model's own rig, or double-sided sheets: Mambo); it decides what is a fault: clipping a
            weapon through the body is one for a standard soldier, a note for a stylised one
  checks    each a measure on the build (its manifest, quality report, animation review) and, where one is known, a
            fix (build options); ``CHECKS``:
              arms_down       the hands below the shoulders, near the game's arm angle (a T-pose: Mambo's arms were
                              never swung; fix: a farther hand search, ``hand_reach``)
              limbs_on_limbs  the arm area's weight mostly on the arm bones, not the spine (her sleeves were 34-43%
                              spine: they stayed out in game)
              rigid_parts     no weapon, shield or plume changes shape between poses (W0)
              weapon_in_hand  no hand part floating off its hand (the review's floating)
              clipping        no hand part through the body (standard: a fault; stylised: accepted, noted)
              folds           folds under the game's animations within ``FOLDS`` times the game body's
-> ``preview/checks.json`` and a "checks" measure in the quality report.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

PASSES = 3
ARM_ANGLE = (18.0, 60.0)  # degrees below horizontal: the game's rest pose has 35
SPINE_SHARE = 0.5  # at most this much of the arm area's weight on the spine and hips
FOLDS = {"standard": 4.0, "stylised": 8.0}  # times the game body's folds (Mambo at 8,000 triangles: about 6)
SPINE_BONES = ("hip", "lower_spine", "mid_spine", "top_spine")
GAME_SHOULDER = {"L": (1.06, 13.37), "R": (-1.06, 13.37)}  # western_generic_01_type1 (read at run time in builds)


def profile(manifest: dict, imported: dict | None = None) -> dict:
    """Standard or stylised, with the reasons."""
    why = []
    head = ((manifest.get("fresh") or {}).get("head") or {}).get("head_share")
    marks = (imported or {}).get("landmarks") or {}
    if head is None and marks.get("neck") is not None:
        head = 1.0 - float(marks["neck"])
    if head is not None and head > 0.30:
        why.append(f"head {head:.0%} of the height")
    if (imported or {}).get("double_sided"):
        why.append("sheets drawn from both sides")
    return {"profile": "stylised" if why else "standard", "why": why or ["the game's proportions"]}


def _geometry(manifest: dict) -> dict | None:
    p = Path(manifest.get("geometry") or "")
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def arms_down(manifest: dict, **_) -> dict:
    """The hands (vertices mostly on the hand bones) below the shoulders at an angle in ``ARM_ANGLE``."""
    g = _geometry(manifest)
    if g is None:
        return {"ok": True, "skipped": "no geometry"}
    p, w = g["p"], g["weights"]
    out = {}
    for side in ("L", "R"):
        pts = [p[3 * i:3 * i + 3] for i, row in enumerate(w) if row.get(f"{side}_hand", 0) > 0.5]
        if len(pts) < 5:
            continue
        x = sum(q[0] for q in pts) / len(pts)
        y = sum(q[1] for q in pts) / len(pts)
        sx, sy = GAME_SHOULDER[side]
        out[side] = round(math.degrees(math.atan2(sy - y, abs(x - sx))), 1)
    bad = {s: a for s, a in out.items() if not ARM_ANGLE[0] <= a <= ARM_ANGLE[1]}
    return {"ok": not bad, "angles": out, "detail": (f"arms {', '.join(f'{s} {a} deg' for s, a in bad.items())} below "
                                                    f"horizontal (the game's 35)" if bad else "")}


def limbs_on_limbs(manifest: dict, **_) -> dict:
    """The share of the arm area's weight (|x| over 3.5, 8 to 14.5 high, the head's hair left out) on the spine."""
    g = _geometry(manifest)
    if g is None:
        return {"ok": True, "skipped": "no geometry"}
    p, w = g["p"], g["weights"]
    tot = spine = 0.0
    for i, row in enumerate(w):
        x, y = p[3 * i], p[3 * i + 1]
        if abs(x) > 3.5 and 8.0 < y < 14.5 and row.get("head", 0) < 0.5:
            tot += sum(row.values())
            spine += sum(row.get(b, 0.0) for b in SPINE_BONES)
    share = spine / tot if tot else 0.0
    return {"ok": share <= SPINE_SHARE, "spine_share": round(share, 3),
            "detail": f"{share:.0%} of the arms' weight on the spine" if share > SPINE_SHARE else ""}


def _review_worst(review: dict | None, key: str) -> float:
    vals = []
    for st in (review or {}).get("states", {}).values():
        for m in st.get("ours") or []:
            v = m.get(key)
            vals.append(max(v.values(), default=0.0) if isinstance(v, dict) else (v or 0.0))
    return max(vals, default=0.0)


def rigid_parts(manifest: dict, review: dict | None = None, **_) -> dict:
    from eu4unitgen.blender.jobs._rigid import RIGID_MAX
    worst = _review_worst(review, "rigid")
    return {"ok": worst <= RIGID_MAX, "worst": round(worst, 5), "detail": "a part bends" if worst > RIGID_MAX else ""}


def weapon_in_hand(manifest: dict, review: dict | None = None, **_) -> dict:
    flags = [f["flag"] for st in (review or {}).get("states", {}).values() for f in st.get("flags") or []
             if f["flag"].startswith("floating")]
    return {"ok": not flags, "detail": flags[0] if flags else ""}


def clipping(manifest: dict, review: dict | None = None, prof: str = "standard", **_) -> dict:
    flags = [f["flag"] for st in (review or {}).get("states", {}).values() for f in st.get("flags") or []
             if f["flag"].startswith("clipping")]
    if flags and prof == "stylised":
        return {"ok": True, "accepted": f"{len(flags)} poses (the stylised shape forces it: {flags[0]})"}
    return {"ok": not flags, "detail": f"{len(flags)} poses: {flags[0]}" if flags else ""}


def folds(manifest: dict, prof: str = "standard", **_) -> dict:
    d = manifest.get("deform") or {}
    ours, game = d.get("ours") or {}, d.get("vanilla") or {}
    bad = {}
    for a, x in ours.items():
        g = (game.get(a) or {}).get("flips") or 1
        if x.get("flips", 0) > FOLDS[prof] * max(g, 3):
            bad[a] = (x["flips"], g)
    return {"ok": not bad, "detail": ", ".join(f"{a} {o} folds (the game's {g})" for a, (o, g) in bad.items())}


# name -> (measure, the fix: build options for the next pass, or None)
CHECKS = {
    "arms_down": (arms_down, lambda opts: {**opts, "hand_reach": min(30.0, float(opts.get("hand_reach", 9.0)) * 1.5)}),
    "limbs_on_limbs": (limbs_on_limbs, None),
    "rigid_parts": (rigid_parts, None),
    "weapon_in_hand": (weapon_in_hand, None),
    "clipping": (clipping, None),
    "folds": (folds, None),
}


def run(manifest: dict, review: dict | None, imported: dict | None = None) -> dict:
    """One pass: every check, the profile. -> {"profile", "why", "checks": {name: result}, "failed", "fixable"}."""
    pr = profile(manifest, imported)
    res = {name: fn(manifest, review=review, prof=pr["profile"]) for name, (fn, _) in CHECKS.items()}
    failed = [n for n, r in res.items() if not r.get("ok")]
    return {**pr, "checks": res, "failed": failed, "fixable": [n for n in failed if CHECKS[n][1] is not None]}


def next_options(opts: dict, fixable: list[str]) -> dict:
    """The build options with each fixable failure's fix applied."""
    for n in fixable:
        opts = CHECKS[n][1](opts)
    return opts


def summary(passes: list[dict], fixes: list[dict]) -> dict:
    """The quality report's line: the profile, how many passes, what was fixed, what's accepted or still wrong."""
    last = passes[-1]
    accepted = [f"{n}: {r['accepted']}" for n, r in last["checks"].items() if r.get("accepted")]
    still = [f"{n}: {last['checks'][n].get('detail')}" for n in last["failed"]]
    verdict = "good" if not still else ("ok" if len(still) <= 1 else "check")
    text = (f"{last['profile']} model ({', '.join(last['why'])}); {len(passes)} pass{'es' if len(passes) > 1 else ''}"
            + (f"; fixed: {', '.join(f['fixed'] for f in fixes)}" if fixes else "")
            + (f"; accepted: {'; '.join(accepted)}" if accepted else "")
            + (f"; still: {'; '.join(still)}" if still else "; every check passed"))
    return {"verdict": verdict, "text": text}
