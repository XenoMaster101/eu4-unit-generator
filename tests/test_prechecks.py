"""The checks before presentation (ref3d.prechecks): the profile, the measures, the fixes, the summary."""
import json

from eu4unitgen.ref3d import prechecks


def _manifest(tmp_path, hands_y=10.4, spine=0.1, head=None):
    # two hands (vertices on the hand bones) and an arm area with some spine weight
    p, w = [], []
    for side, x in (("L", 5.5), ("R", -5.5)):
        for k in range(6):
            p += [x, hands_y, 0.0]
            w.append({f"{side}_hand": 1.0})
        for k in range(10):
            p += [x * 0.8, 11.0, 0.0]
            w.append({f"{side}_arm": 1.0 - spine, "top_spine": spine})
    g = tmp_path / "g.json"
    g.write_text(json.dumps({"p": p, "weights": w}))
    m = {"geometry": str(g), "deform": {"ours": {"idle": {"flips": 10}}, "vanilla": {"idle": {"flips": 8}}}}
    if head is not None:
        m["fresh"] = {"head": {"head_share": head}}
    return m


def test_a_t_pose_fails_and_its_fix_widens_the_hand_search(tmp_path):
    ok = prechecks.run(_manifest(tmp_path), None)
    assert ok["profile"] == "standard" and not ok["failed"]
    tpose = prechecks.run(_manifest(tmp_path, hands_y=13.3), None)
    assert tpose["failed"] == ["arms_down"] and tpose["fixable"] == ["arms_down"]
    assert prechecks.next_options({}, tpose["fixable"]) == {"hand_reach": 13.5}


def test_limbs_on_the_spine_fail_and_the_profile_decides_clipping(tmp_path):
    assert "limbs_on_limbs" in prechecks.run(_manifest(tmp_path, spine=0.95), None)["failed"]
    review = {"states": {"walk": {"ours": [], "flags": [{"flag": "clipping: 30% of the weapon inside the body"}]}}}
    std = prechecks.run(_manifest(tmp_path), review)
    sty = prechecks.run(_manifest(tmp_path, head=0.6), review)
    assert "clipping" in std["failed"] and sty["profile"] == "stylised" and "clipping" not in sty["failed"]
    assert "accepted: clipping" in prechecks.summary([sty], [])["text"]
    s = prechecks.summary([std, std], [{"pass": 2, "fixed": "arms_down"}])
    assert "2 passes" in s["text"] and "fixed: arms_down" in s["text"]
