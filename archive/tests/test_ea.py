"""Track M spike E-A: the app-side verdict (the fitting runs in the AI environment)."""
from eu4unitgen.ref3d import ea


def _view(region, lm, region_min=0.85, lm_max=0.025):
    return {"gates": {"region_iou": {"pass": region >= region_min, "worst": region},
                      "landmarks": {"pass": lm <= lm_max, "worst": lm}}}


def _cand(l, r, costs=(1.0, 1.1, 1.2)):
    return {"views": {"L": _view(*l), "R": _view(*r)}, "restarts": [{"cost": c} for c in costs]}


def test_beats_needs_no_worse_in_every_camera_and_better_somewhere():
    b2 = _cand((0.80, 0.03), (0.82, 0.02))
    assert ea.beats(_cand((0.90, 0.02), (0.86, 0.02)), b2)
    assert not ea.beats(_cand((0.90, 0.02), (0.81, 0.02)), b2)  # worse region IoU in R
    assert not ea.beats(_cand((0.80, 0.03), (0.82, 0.02)), b2)  # equal everywhere: not better


def test_verdict_needs_gates_beating_b2_and_converged_restarts():
    res = {"candidates": {"b2": _cand((0.80, 0.03), (0.82, 0.02)),
                          "procedural": _cand((0.90, 0.02), (0.88, 0.015))}}
    v = ea.verdict(res)
    assert v["pass"] and v["best"] == "procedural"
    res["candidates"]["procedural"]["restarts"] = [{"cost": 1.0}, {"cost": 2.0}]
    assert not ea.verdict(res)["restarts_converge"]
    res["candidates"]["procedural"] = _cand((0.90, 0.03), (0.88, 0.015))
    assert not ea.verdict(res)["gates_pass"]


def test_verdict_picks_the_better_fitted_candidate():
    res = {"candidates": {"b2": _cand((0.70, 0.05), (0.70, 0.05)),
                          "procedural": _cand((0.86, 0.02), (0.84, 0.02)),
                          "hybrid": _cand((0.90, 0.02), (0.88, 0.02))}}
    assert ea.verdict(res)["best"] == "hybrid"
