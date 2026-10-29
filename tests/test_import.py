"""Import a model (ref3d.importer): the licence (rule 8), the request and stages of an imported part, its credits in
the mod (export.writer.plan_extra). The Blender job (prepare_import) runs in the app test run (tests/import_run.py)."""
import json
from pathlib import Path

import pytest

from eu4unitgen.export.writer import BLOCKED, CREATE, ExportPlan, plan_extra
from eu4unitgen.ref3d import flow, importer


def _model(tmp_path, name="knight.glb"):
    p = tmp_path / name
    p.write_bytes(b"glTF" + b"\0" * 64)
    return p


def test_the_licence_is_required_and_cc_by_needs_its_credit(tmp_path):
    m = _model(tmp_path)
    assert importer.problem(None, "cc0").startswith("Pick a model")
    assert "isn't a .glb or .fbx" in importer.problem(tmp_path / "knight.obj", "cc0")
    assert importer.problem(m, None).startswith("Say what licence")
    assert importer.problem(m, "cc_by", "").startswith("A CC BY model needs its credit")
    assert importer.problem(m, "cc_by", "Knight by Jane Doe, example.org") == ""
    assert importer.problem(m, "cc0") == "" and importer.problem(m, "own") == ""
    assert importer.rights_text("cc_by", "Knight by Jane Doe") == "3D model under CC BY, credit: Knight by Jane Doe"


def test_an_imported_part_records_its_licence_and_skips_the_ai(tmp_path, monkeypatch):
    monkeypatch.setenv("EU4UNITGEN_WORKSPACE", str(tmp_path / "ws"))
    m = _model(tmp_path)
    with pytest.raises(ValueError, match="credit"):
        flow.new_request("knight", None, "", "soldier", model={"path": str(m), "licence": "cc_by", "credit": ""})
    f = flow.new_request("knight", None, "", "soldier", 2,
                         model={"path": str(m), "licence": "cc_by", "credit": "Knight by Jane Doe, example.org"})
    req, subj = flow.request(f), json.loads((f / "subject.json").read_text(encoding="utf-8"))
    assert req["image"] is None and req["source"]["licence"] == "cc_by" and req["source"]["file"] == "knight.glb"
    assert req["rights"] == "3D model under CC BY, credit: Knight by Jane Doe, example.org" == subj["rights"]
    assert subj["pipeline"] == "import" and subj["source"]["credit"] == "Knight by Jane Doe, example.org"
    assert flow.stages_of("soldier", req) == ("import", "build", "finish", "preview")
    assert flow.stages_of("soldier") == flow.STAGES  # a picture's run as before
    assert set(flow.estimates("soldier", req)) == {"import", "build", "finish", "preview"}
    assert "import" in flow.RUN and "import" in flow.LABELS and "import" in flow.EXPLAIN
    with pytest.raises(ValueError, match="only a soldier, horse or gun"):
        flow.new_request("knights", None, "", "cavalry", model={"path": str(m), "licence": "own"})


def test_credits_follow_a_part_into_its_units(tmp_path, monkeypatch):
    monkeypatch.setenv("EU4UNITGEN_WORKSPACE", str(tmp_path / "ws"))
    m = _model(tmp_path)
    flow.new_request("knight", None, "", "soldier", model={"path": str(m), "licence": "cc_by", "credit": "Jane Doe"})
    img = tmp_path / "horse.png"
    from PIL import Image
    Image.new("RGB", (32, 32)).save(img)
    flow.new_request("bay", img, "mine", "horse")
    lines, by = importer.credits_for(flow.request(flow.folder_of("knight")))
    assert by and lines == ["knight: knight.glb - 3D model under CC BY, credit: Jane Doe"]
    c = flow.new_request("knight_on_bay", None, "", "cavalry", horse="bay", soldier="knight")
    assert importer.credits_for(flow.request(c)) == (lines, True)
    assert importer.credits_for(flow.request(flow.folder_of("bay"))) == ([], False)
    assert b"credit: Jane Doe" in importer.credits_text(lines)


def test_the_credits_file_is_never_written_over_a_file_the_tool_didnt_write(tmp_path):
    plan = ExportPlan(tmp_path, "knight_units", manifest={"files": {}})
    pf = plan_extra(plan, "knight_units/eu4unitgen_credits.txt", b"credits\n")
    assert pf.action == CREATE and "knight_units/eu4unitgen_credits.txt" in plan.manifest["files"]
    (tmp_path / "knight_units").mkdir()
    (tmp_path / "knight_units" / "eu4unitgen_credits.txt").write_text("the user's own notes")
    plan2 = ExportPlan(tmp_path, "knight_units", manifest={"files": {}})
    assert plan_extra(plan2, "knight_units/eu4unitgen_credits.txt", b"credits\n").action == BLOCKED
    assert "knight_units/eu4unitgen_credits.txt" not in plan2.manifest["files"]


def test_a_glbs_own_licence_is_suggested_only_when_it_is_cc0_or_plain_cc_by(tmp_path):
    import struct

    def glb(extras):
        doc = json.dumps({"asset": {"version": "2.0", "extras": extras}}).encode()
        doc += b" " * (-len(doc) % 4)
        p = tmp_path / "m.glb"
        p.write_bytes(b"glTF" + struct.pack("<II", 2, 20 + len(doc)) + struct.pack("<I", len(doc)) + b"JSON" + doc)
        return p
    m = importer.metadata(glb({"title": "Knight", "author": "Jane", "license": "CC-BY-4.0 (http://x)", "source": "http://s"}))
    assert m["licence"] == "cc_by" and m["credit"] == "“Knight”, by Jane, http://s, CC-BY-4.0 (http://x)"
    assert importer.metadata(glb({"license": "CC0 1.0"}))["licence"] == "cc0"
    for other in ("CC-BY-SA-4.0", "CC-BY-NC-4.0", "CC-BY-ND-4.0", "All rights reserved"):
        assert importer.metadata(glb({"license": other}))["licence"] is None
    assert importer.metadata(tmp_path / "missing.glb") == {"licence": None, "credit": ""}
