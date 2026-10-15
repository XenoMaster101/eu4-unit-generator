"""Unit sets shared by several tags: one sprite pack with an OR trigger, per-tag hand-over."""
from __future__ import annotations

import pytest

from eu4unitgen.cli import main as cli_main
from eu4unitgen.core.generator import GeneratedFile, generate, tag_trigger
from eu4unitgen.core.nation import Nation
from eu4unitgen.export.pipeline import Session, SetSpec, Settings, SpecError
from eu4unitgen.export.testmod import build_test_mod
from eu4unitgen.pdxscript.parser import parse_bytes
from eu4unitgen.validate import validate
from golden import check_golden

RED = ((150, 20, 30), (210, 170, 40), (25, 35, 90))
BLUE = ((20, 60, 200), (240, 200, 30), (20, 20, 20))


@pytest.fixture
def shared() -> Nation:
    return Nation(["AAA", "TST"], "hellenicgfx", set_name="shared", adjective="Shared",
                  unit_colours={"AAA": RED, "TST": BLUE})


def test_nation_rules():
    assert Nation("MAC", "hellenicgfx").tags == ["MAC"]
    with pytest.raises(ValueError, match="needs a set name"):
        Nation(["MAC", "EPI"], "hellenicgfx")
    with pytest.raises(ValueError, match="listed twice"):
        Nation(["MAC", "MAC"], "hellenicgfx", set_name="x")
    with pytest.raises(ValueError, match="aren't tags of this set"):
        Nation(["MAC", "EPI"], "hellenicgfx", set_name="x", unit_colours={"THR": RED})
    one_for_all = Nation(["MAC", "EPI"], "hellenicgfx", set_name="x", unit_colours=RED)
    assert one_for_all.unit_colours == {"MAC": RED, "EPI": RED}
    assert tag_trigger(["MAC"]) == "tag = MAC" and tag_trigger(["MAC", "EPI"]) == "OR = { tag = MAC tag = EPI }"


def test_shared_set_generates_one_pack_and_per_tag_handover(project, game, shared):
    files, idx = game
    out = generate(project, shared, files, idx)
    pack = parse_bytes(out.files["gfx/sprite_packs/hr_shared_pack.txt"].data).get_block("hr_shared_units")
    trigger = pack.get_block("country_trigger").get_block("OR")
    assert [v.value for v in trigger.get_all("tag")] == ["AAA", "TST"]
    names = {e.get_str("name") for e in parse_bytes(out.files["gfx/entities/hr_shared_units.asset"].data).get_all("entity")}
    assert "AAA_INFANTRY_1" in names and not any(n.startswith("TST_") for n in names)  # named after the first tag
    integ = out.integration
    assert integ.tags == ["AAA", "TST"]
    assert integ.country_files == {"AAA": "common/countries/Alpha.txt", "TST": "common/countries/Testland.txt"}
    text = out.files["INTEGRATION.md"].text
    check_golden("multitag/INTEGRATION.md", text)
    assert "- AAA: `common/countries/Alpha.txt`" in text and "- TST: `common/countries/Testland.txt`" in text
    assert "TST = {\n\tcolor1 = { 20 60 200 }" in text and "AAA = {\n\tcolor1 = { 150 20 30 }" in text
    report = validate(project, shared, out, files, idx)
    assert report.errors == [] and report.warnings == [], "\n".join(map(str, report.issues))
    assert not any(i.rule == "unit_colours" for i in report.issues)


def test_validator_catches_a_single_tag_trigger_and_missing_colours(project, game, shared):
    files, idx = game
    out = generate(project, shared, files, idx)
    rel = "gfx/sprite_packs/hr_shared_pack.txt"
    out.files[rel] = GeneratedFile(rel, out.files[rel].data.replace(b"OR = { tag = AAA tag = TST }", b"tag = AAA"))
    report = validate(project, shared, out, files, idx)
    assert any(i.rule == "sprite_packs" and "OR = { tag = AAA tag = TST }" in i.message for i in report.errors)
    half = Nation(["AAA", "TST"], "hellenicgfx", set_name="shared", unit_colours={"AAA": RED})
    report = validate(project, half, generate(project, half, files, idx), files, idx)
    notes = [i.message for i in report.issues if i.rule == "unit_colours"]
    assert len(notes) == 1 and notes[0].startswith("TST has no entry")


def test_pipeline_one_set_for_several_tags(fake_install, tmp_path):
    s = Session(Settings(eu4_path=fake_install, output_dir=tmp_path / "out", mod_dir=tmp_path / "mods"))
    try:
        with pytest.raises(SpecError, match="give the set a name"):
            s.validate(SetSpec(["AAA", "TST"], "hellenicgfx"))
        with pytest.raises(SpecError, match="different tech groups"):
            s.validate(SetSpec(["AAA", "BBB"], "hellenicgfx", set_name="mixed"))  # western + eastern
        res = s.validate(SetSpec(["AAA", "BBB"], "hellenicgfx", set_name="mixed", tech_group="western"))
        assert len(res) == 1 and res[0].nation.tags == ["AAA", "BBB"] and res[0].report.errors == []
        res = s.generate(SetSpec(["AAA", "TST"], "hellenicgfx", set_name="shared", unit_colours={"TST": BLUE}))
        assert (tmp_path / "out" / "shared" / "gfx" / "sprite_packs" / "hr_shared_pack.txt").is_file()
    finally:
        s.close()


def test_test_mod_patches_every_tag(project, fake_install, shared, tmp_path):
    from eu4unitgen.library.scanner import GameFiles, build_index
    from eu4unitgen.library.tags import unit_colours
    (fake_install / "common/country_colors").mkdir(parents=True)
    (fake_install / "common/country_colors/00_country_colors.txt").write_text(
        "# made up\nTST = {\n\tcolor1= { 1 2 3 }\n\tcolor2= { 4 5 6 }\n\tcolor3= { 7 8 9 }\n}")
    (tmp_path / "mods").mkdir()
    files = GameFiles(fake_install)
    try:
        tm = build_test_mod(project, shared, files, build_index(files), tmp_path / "mods")
    finally:
        files.close()
    for cfile in ("common/countries/Alpha.txt", "common/countries/Testland.txt"):
        assert parse_bytes(tm.package.files[cfile].data).get_str("graphical_culture") == "hellenicgfx"
    rel = "common/country_colors/00_country_colors.txt"
    patched = parse_bytes(tm.package.files[rel].data)
    assert [p.key.value for p in patched.pairs()] == ["TST", "AAA"]  # TST replaced in place, AAA appended
    from eu4unitgen.validate.rules import _OneFile
    assert unit_colours(_OneFile(rel, tm.package.files[rel].data)) == {"TST": list(BLUE), "AAA": list(RED)}
    assert tm.report.errors == [], "\n".join(map(str, tm.report.issues))


def test_cli_several_tags(fake_install, tmp_path, capsys):
    base = ["generate", "--gfx", "hellenicgfx", "--eu4", str(fake_install), "--out", str(tmp_path / "out"), "--dry-run"]
    assert cli_main(base + ["--tag", "AAA", "TST"]) == 1
    assert "give the set a name" in capsys.readouterr().out
    assert cli_main(base + ["--tag", "AAA", "TST", "--set", "shared",
                            "--tag-unit-colours", "TST", "20", "60", "200", "240", "200", "30", "20", "20", "20"]) == 0
    out = capsys.readouterr().out
    assert "0 errors" in out and "TST = {" in out


def test_gui_unit_colours_per_tag(qapp_window):
    page = qapp_window.unit_page
    page.tags.setText("AAA TST")
    page.set_unit_colours("TST", BLUE)
    assert page.spec().unit_colours == {"TST": BLUE}
    page.unit_colour_tag.setCurrentText("AAA")
    assert not page.unit_colours_on.isChecked()
    page.unit_colours_on.setChecked(True)
    assert set(page.spec().unit_colours) == {"AAA", "TST"}


@pytest.fixture
def qapp_window(fake_install, tmp_path):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets")
    from eu4unitgen.gui.app import MainWindow
    widgets.QApplication.instance() or widgets.QApplication([])
    w = MainWindow(Session(Settings(eu4_path=fake_install, output_dir=tmp_path / "out")), run_async=False)
    yield w
    w.session.close()
