"""ref3d.job (M19 step 1): the app's horse run. The pipeline itself is stubbed: these check the
pre-check, the status file, resuming a stopped run and the estimates."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from eu4unitgen.ref3d import auto, job


def horse_like(path: Path, size=(600, 400), box=(100, 80, 500, 360), bg=(200, 200, 200), busy=False) -> Path:
    im = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(im)
    if busy:  # a scene: stripes all over
        for x in range(0, size[0], 6):
            d.line((x, 0, x, size[1]), fill=(30, 90, 30), width=2)
    d.rectangle(box, fill=(120, 60, 30))
    im.save(path)
    return path


def test_precheck_passes_a_clean_side_view(tmp_path):
    assert job.precheck(horse_like(tmp_path / "a.png")) == []


def test_precheck_flags_busy_background_tall_subject_and_crop(tmp_path):
    busy = job.precheck(horse_like(tmp_path / "b.png", busy=True))
    assert any("background" in w for w in busy)
    tall = job.precheck(horse_like(tmp_path / "c.png", box=(200, 40, 400, 380)))
    assert any("side view" in w for w in tall)
    crop = job.precheck(horse_like(tmp_path / "d.png", box=(0, 0, 500, 380)))
    assert any("edges" in w for w in crop)


def test_go_records_stages_and_resumes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(job, "BEAT", 0.05)
    img = horse_like(tmp_path / "h.png")
    calls = []

    def new(image, name, rights, ai_python, eu4, root):
        folder = Path(root) / name
        (folder / "subject.json").write_text("{}", encoding="utf-8")
        (folder / "auto_new.json").write_text(json.dumps({"seconds": 3, "parts": {"saddle": True}}), encoding="utf-8")
        calls.append("new")
        return {"seconds": 3, "side_view_window": True, "parts": {"saddle": True}}

    def run(subject, ai_python, blender=None, progress=None):
        progress("shells", "running")
        progress("shells", "done", version="v001", seconds=5)
        if not calls.count("run"):
            calls.append("run")
            raise SystemExit("fit: 3.0 GB of 32 GB RAM free")  # the first run stops
        calls.append("run")
        progress("fit", "done", version="v001", seconds=7)
        return {}

    monkeypatch.setattr(auto, "new", new)
    monkeypatch.setattr(auto, "run", run)
    monkeypatch.setattr(job, "final_sheet", lambda folder: Path(folder) / "final_sheet.png")
    with pytest.raises(SystemExit):
        job.go(img, "h1", "mine", "py", tmp_path)
    st = job.status(Path("assets_in/h1"))
    assert st["state"] == "failed" and "RAM" in st["error"] and st["stages"]["read"]["state"] == "done"
    job.go(img, "h1", "mine", "py", tmp_path)  # resume: the read isn't redone
    st = job.status(Path("assets_in/h1"))
    assert calls == ["new", "run", "run"] and st["state"] == "done" and st["error"] is None
    assert st["stages"]["fit"]["seconds"] == 7 and st["parts"] == {"saddle": True} and st["resumed"]


def test_a_stale_heartbeat_reads_as_stopped(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"name": "x", "stages": {}, "heartbeat": time.time() - 3600,
                                                      "done": False, "error": None}), encoding="utf-8")
    assert job.status(tmp_path)["state"] == "stopped"


def test_go_refuses_a_folder_it_did_not_start(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "assets_in" / "old").mkdir(parents=True)
    with pytest.raises(SystemExit, match="wasn't started from the app"):
        job.go(tmp_path / "x.png", "old", "mine", "py", tmp_path)


def test_estimates_from_earlier_runs(tmp_path):
    assert job.estimates(tmp_path)["fit"] == job.DEFAULT_SECONDS["fit"]  # nothing run yet: horse 2's
    d = tmp_path / "a"
    d.mkdir()
    (d / "auto_run.json").write_text(json.dumps({"fit_workers": 2, "steps": {"fit": {"seconds": 1000}}}), encoding="utf-8")
    (d / "auto_new.json").write_text(json.dumps({"seconds": 50}), encoding="utf-8")
    est = job.estimates(tmp_path, fit_workers=4)
    assert est["fit"] == 500 and est["read"] == 50 and est["shells"] == job.DEFAULT_SECONDS["shells"]
