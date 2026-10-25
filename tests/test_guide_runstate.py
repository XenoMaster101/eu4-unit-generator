"""The horse image check (core.guide, from ref3d.job's precheck) and a run's status file (ref3d.runstate)."""
from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image, ImageDraw

from eu4unitgen.core import guide
from eu4unitgen.ref3d import runstate


def horse_like(path: Path, size=(600, 400), box=(100, 80, 500, 360), bg=(200, 200, 200), busy=False) -> Path:
    im = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(im)
    if busy:  # a scene: stripes all over
        for x in range(0, size[0], 6):
            d.line((x, 0, x, size[1]), fill=(30, 90, 30), width=2)
    d.rectangle(box, fill=(120, 60, 30))
    im.save(path)
    return path


def test_horse_check_passes_a_clean_side_view(tmp_path):
    assert guide.horse_check(horse_like(tmp_path / "a.png")) == []


def test_horse_check_flags_busy_background_tall_subject_and_crop(tmp_path):
    busy = guide.horse_check(horse_like(tmp_path / "b.png", busy=True))
    assert any("background" in w for w in busy)
    tall = guide.horse_check(horse_like(tmp_path / "c.png", box=(200, 40, 400, 380)))
    assert any("side view" in w for w in tall)
    crop = guide.horse_check(horse_like(tmp_path / "d.png", box=(0, 0, 500, 380)))
    assert any("edges" in w for w in crop)


def test_a_stale_heartbeat_reads_as_stopped(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"name": "x", "stages": {}, "heartbeat": time.time() - 3600,
                                                      "done": False, "error": None}), encoding="utf-8")
    assert runstate.status(tmp_path)["state"] == "stopped"


def test_status_stages_and_heartbeat(tmp_path):
    st = runstate.Status(tmp_path, {"stages": {}, "done": False, "error": None})
    try:
        st.stage("build", "running")
        assert runstate.status(tmp_path)["state"] == "running"
        st.stage("build", "done")
        st.stage("build", "running")  # run again (an edit): the old time doesn't stay
        assert "seconds" not in runstate.read_status(tmp_path)["stages"]["build"]
        st.write(error="boom")
        assert runstate.status(tmp_path)["state"] == "failed"
    finally:
        st.close()
