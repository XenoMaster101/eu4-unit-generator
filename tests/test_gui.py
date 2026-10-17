"""Cheap GUI checks: the wizard loads, each screen exists, and its buttons call the pipeline.
Runs headless (Qt's offscreen platform); skipped if PySide6 isn't installed."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from eu4unitgen.export.pipeline import Session, Settings  # noqa: E402
from eu4unitgen.gui.app import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(qapp, fake_install, tmp_path):
    settings = Settings(eu4_path=fake_install, output_dir=tmp_path / "out", stage_dir=tmp_path / "stage",
                        mod_dir=tmp_path / "eu4mods")
    w = MainWindow(Session(settings), run_async=False)
    w.settings_page.output.setText(str(tmp_path / "out"))
    w.settings_page.stage.setText(str(tmp_path / "stage"))
    w.settings_page.user_mods.setText(str(tmp_path / "eu4mods"))
    yield w
    w.session.close()
    import shiboken6

    shiboken6.delete(w)  # a live window would be re-styled by every later theme change (slow)


def test_every_screen_loads(window):
    assert [window.steps.item(i).text() for i in range(window.steps.count())] == [
        "1. Settings", "2. Character", "3. Unit set", "4. Textures", "5. Parts", "6. Validate", "7. Generate",
        "8. Test mod"]
    for i, page in enumerate(window.pages):
        window.steps.setCurrentRow(i)
        assert window.stack.currentWidget() is page


def test_load_install_fills_the_forms(window):
    window.settings_page.load.click()
    assert "Loaded release_9.99" in window.settings_page.info.text()
    assert window.settings_page.dlc.count() == 1  # dlc999_test
    assert window.unit_page.base.findText("westerngfx") >= 0
    assert window.unit_page.gfx.findText("easterngfx") >= 0


def test_generate_button_calls_the_generator(window, monkeypatch):
    calls = []
    real = window.session.generate
    monkeypatch.setattr(window.session, "generate", lambda spec: calls.append(spec) or real(spec))
    window.unit_page.tags.setText("MAC")
    window.unit_page.adjective.setText("Macedonian")
    window.generate_page.run.click()
    assert [(s.tags, s.gfx, s.adjective) for s in calls] == [(["MAC"], "hellenicgfx", "Macedonian")]
    assert "Wrote" in window.generate_page.message.text()
    assert "Integration: mac (MAC, hellenicgfx)" in window.generate_page.integration.toPlainText()
    assert window.generate_page.files.topLevelItemCount() == 1
    assert (window.session.settings.output_dir / "mac" / "INTEGRATION.md").is_file()


def test_validate_lists_issues_and_bad_input_shows_an_error(window):
    window.unit_page.tags.setText("MAC")
    window.validate_page.run.click()
    top = window.validate_page.tree.topLevelItem(0)
    assert top.text(0) == "MAC / mac" and "0 errors" in top.text(2)
    window.unit_page.tags.setText("MAC EPI")
    window.unit_page.set_name.setText("")
    window.validate_page.run.click()
    assert "give the set a name" in window.validate_page.message.text()
    window.unit_page.set_name.setText("argead")
    window.validate_page.run.click()
    top = window.validate_page.tree.topLevelItem(0)
    assert top.text(0) == "MAC EPI / argead" and "0 errors" in top.text(2), top.text(2)


def test_test_mod_button_stages_into_the_staging_folder(window, tmp_path):
    window.unit_page.tags.setText("AAA")
    window.test_page.run.click()
    assert "Staged" in window.test_page.message.text(), window.test_page.message.text()
    assert (tmp_path / "stage" / "eu4unitgen_test.mod").is_file()
    desc = (tmp_path / "stage" / "eu4unitgen_test.mod").read_text()
    assert f'path="{(tmp_path / "eu4mods" / "eu4unitgen_test").as_posix()}"' in desc
    assert not (tmp_path / "eu4mods").exists()  # never writes the real mod folder
    window.test_page.run.click()  # rebuild: the staged mod is cleared first
    assert "Staged" in window.test_page.message.text()


def test_texture_test_checkbox_builds_the_texture_test(window, tmp_path, fake_install):
    from fakes import make_textures_decodable
    make_textures_decodable(fake_install)
    window.unit_page.tags.setText("AAA")
    window.test_page.suite.setChecked(True)
    window.test_page.textures.setChecked(True)
    assert not window.test_page.suite.isChecked()  # one or the other
    window.test_page.run.click()
    assert "Staged" in window.test_page.message.text(), window.test_page.message.text()
    assert (tmp_path / "stage" / "eu4unitgen_test_textures.mod").is_file()
    assert "TIER 1" in window.test_page.checklist.toPlainText()


def test_blender_path_is_pinned_and_editable(window, tmp_path, monkeypatch):
    from eu4unitgen.core.project import WINDOWS_BLENDER_PATH, default_blender_path
    monkeypatch.delenv("BLENDER_PATH", raising=False)
    if os.name == "nt":
        assert default_blender_path() == WINDOWS_BLENDER_PATH
        assert Settings().blender_path == WINDOWS_BLENDER_PATH
    monkeypatch.setenv("BLENDER_PATH", str(tmp_path / "blender"))
    assert Settings().blender_path == tmp_path / "blender"
    window.settings_page.blender.setText(str(tmp_path / "b.exe"))
    assert window.settings_page.settings().blender_path == tmp_path / "b.exe"


def test_clear_stage_refuses_foreign_files(tmp_path):
    from eu4unitgen.export.pipeline import clear_stage
    from eu4unitgen.export.writer import ExportError
    (tmp_path / "eu4unitgen_test_v1").mkdir()
    (tmp_path / "notes.txt").write_text("mine")
    with pytest.raises(ExportError, match="notes.txt"):
        clear_stage(tmp_path)
    assert (tmp_path / "notes.txt").is_file() and (tmp_path / "eu4unitgen_test_v1").is_dir()


def _textures_window(window, fake_install, tmp_path):
    from PIL import Image
    from fakes import make_textures_decodable
    make_textures_decodable(fake_install)
    ref = Image.new("RGB", (40, 40), (150, 20, 30))
    ref.paste((210, 170, 40), (25, 0, 40, 40))
    ref.save(tmp_path / "ref.png")
    window.unit_page.tags.setText("MAC")
    page = window.textures_page
    page.palette_image.setText(str(tmp_path / "ref.png"))
    return page


def test_textures_preview_and_guide(window, fake_install, tmp_path):
    page = _textures_window(window, fake_install, tmp_path)
    page.preview_btn.click()
    assert "Base generic_01 (PdxMeshColor)" in page.message.text(), page.message.text()
    assert all(not label.pixmap().isNull() for label in page.previews)
    page.guide_btn.click()
    assert "Wrote 3 PNGs" in page.message.text(), page.message.text()
    assert (tmp_path / "out" / "texture_guides" / "mac" / "1_infantry_regions.png").is_file()
    page.pattern_region.setCurrentText("hat")
    page.pattern_image.setText(str(tmp_path / "ref.png"))
    page.preview_btn.click()
    assert "region 'hat'" in page.message.text()


def test_textures_recipe_used_by_generate_and_saved(window, fake_install, tmp_path):
    page = _textures_window(window, fake_install, tmp_path)
    page.role.setCurrentText("horse")  # stores 1/infantry, switches to 1/horse
    page.cloth_on.setChecked(True)
    page.role.setCurrentText("infantry")
    assert page.palette_image.text().endswith("ref.png")  # 1/infantry came back
    assert "1/horse (recolour)" in page.slots_label.text() and "1/infantry (palette)" in page.slots_label.text()
    page.use.setChecked(True)
    window.generate_page.run.click()
    assert "Wrote" in window.generate_page.message.text(), window.generate_page.message.text()
    out = tmp_path / "out" / "mac" / "gfx" / "models" / "Units"
    assert (out / "hr_mac_infantry_1_diffuse.dds").is_file() and (out / "hr_mac_horse_1_diffuse.dds").is_file()

    page.save_file(tmp_path / "recipe.json")
    page._clear_slot()
    page.load_file(tmp_path / "recipe.json")
    assert page.palette_image.text().endswith("ref.png") and page.use.isChecked()
    page.use.setChecked(False)
    assert page.recipe_for_spec() is None


def test_check_blender_button(window, monkeypatch):
    monkeypatch.setattr(window.session, "check_blender", lambda: {"blender": "4.2.23", "addon_version": "0.91",
                                                                 "module": "bl_ext.user_default.io_pdx_mesh"})
    window.settings_page.check_blender.click()
    assert "Blender 4.2.23 OK; io_pdx_mesh 0.91 enabled" in window.settings_page.message.text()


def test_no_console_errors_go_to_a_log(monkeypatch, tmp_path):
    """pythonw.exe (the desktop shortcut) has no stdout/stderr: they go to the GUI log instead."""
    import sys
    from eu4unitgen.gui import app as app_mod
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)
    app_mod._install_error_reporting()
    print("hello from pythonw", file=sys.stderr)
    sys.stderr.flush()
    assert app_mod.log_path() == tmp_path / "eu4unitgen" / "gui.log"
    assert "hello from pythonw" in app_mod.log_path().read_text(encoding="utf-8")
    sys.stderr.close()


def _fake_blender(window, monkeypatch):
    """Generate runs Blender for a loadout: swap in test_parts' fake runner."""
    from test_parts import FakeRunner
    runner = FakeRunner()
    monkeypatch.setattr(window.session, "runner", lambda: runner)
    return runner


def test_parts_form_round_trip_and_crew_toggle(window, tmp_path):
    page = window.parts_page
    right, left, head = (page.boxes[b] for b in ("Right_hand_node", "Left_hand_node", "head"))
    right.kind.setCurrentIndex(right.kind.findData("dory"))
    right.params["tip"].setValue(18.0)
    left.kind.setCurrentIndex(left.kind.findData("hoplon"))
    left.zones["face"].set_value("country2")
    head.kind.setCurrentIndex(head.kind.findData("crest"))
    head.params["orientation"].setCurrentText("transverse")
    page.role.setCurrentText("rider")  # stores 1/infantry
    assert left.kind.findData("hoplon") == -1 and right.kind.findData("xiphos") >= 0  # no shields on riders
    assert not page.stance.isEnabled()
    page.role.setCurrentText("crew")
    assert not right.isEnabled() and "gun crew is the infantry entity" in page.message.text()
    page.crew_apart.setChecked(True)
    assert right.isEnabled() and (1, "crew") in page.loadout.slots
    page.role.setCurrentText("infantry")
    assert right.kind_name() == "dory" and right.params["tip"].value() == 18.0
    assert left.zones["face"].value() == "country2" and head.params["orientation"].currentText() == "transverse"
    assert right.params["butt"].value() == right.params["butt"].minimum()  # untouched: auto
    page.save_file(tmp_path / "loadout.json")
    saved = (tmp_path / "loadout.json").read_text(encoding="utf-8")
    assert '"tip": 18.0' in saved and '"1/crew"' in saved and '"butt"' not in saved
    page._clear_slot()
    assert right.kind_name() is None
    page.load_file(tmp_path / "loadout.json")
    assert right.kind_name() == "dory" and page.use.isChecked()
    assert "1/infantry (dory, hoplon, crest)" in page.slots_label.text()


def test_parts_loadout_used_by_generate_and_preview_button(window, tmp_path, monkeypatch):
    runner = _fake_blender(window, monkeypatch)
    window.unit_page.tags.setText("MAC")
    page = window.parts_page
    page.boxes["head"].kind.setCurrentIndex(page.boxes["head"].kind.findData("crest"))
    page.use.setChecked(True)
    window.generate_page.run.click()
    assert "Wrote" in window.generate_page.message.text(), window.generate_page.message.text()
    assert (tmp_path / "out" / "mac" / "gfx" / "models" / "units" / "hr_mac" / "hr_mac_crest_1.mesh").is_file()
    assert runner.calls
    calls = []
    monkeypatch.setattr(window.session, "parts_preview", lambda spec, tier, role, lo: calls.append((tier, role)) or
                        type("P", (), {"image": __import__("PIL.Image").Image.new("RGB", (720, 360)), "body": "generic_01",
                                       "path": tmp_path / "p.png", "parts": [], "notes": ["n1"]})())
    page.tier.setCurrentText("2")
    page.preview_btn.click()
    assert calls == [(2, "infantry")] and "Body generic_01" in page.message.text()
    assert not page.preview.pixmap().isNull()


def test_parts_bodies_points_and_shield_warning(window, tmp_path):
    window.settings_page.load.click()
    page = window.parts_page
    assert page.body.findText("generic_01") >= 0  # filled from Session.body_candidates on load
    page.tier.setCurrentText("3")
    page.body.setCurrentText("generic_01")
    page.points["rider"].setCurrentText("up")
    left = page.boxes["Left_hand_node"]
    left.kind.setCurrentIndex(left.kind.findData("hoplon"))
    assert "no weapon part" in page.warning.text()
    right = page.boxes["Right_hand_node"]
    right.kind.setCurrentIndex(right.kind.findData("xyston"))
    assert page.warning.text() == ""
    page.tier.setCurrentText("1")
    assert page.body.currentText() == "(the tier's own)"
    page.tier.setCurrentText("3")
    assert page.body.currentText() == "generic_01"
    page.save_file(tmp_path / "lo.json")
    saved = (tmp_path / "lo.json").read_text(encoding="utf-8")
    assert '"3": "generic_01"' in saved and '"rider": "up"' in saved and '"xyston"' in saved


def test_cultures_field_reaches_the_set(window):
    window.unit_page.tags.setText("")
    window.unit_page.cultures.setText("alphan, Testish")
    window.unit_page.set_name.setText("alphan_set")
    spec = window.unit_page.spec()
    assert spec.tags == [] and spec.cultures == ["alphan", "testish"]
    window.validate_page.run.click()
    top = window.validate_page.tree.topLevelItem(0)
    assert top.text(0) == "alphan testish / alphan_set" and "0 errors" in top.text(2), top.text(2)


def test_character_page_areas_finishes_masks_and_spec(window, fake_install, tmp_path):
    """A3b: a character spec loads into the pages; areas get finishes that land in the recipe;
    a painted mask becomes a row; saving writes a spec that loads back the same."""
    import json
    from PIL import Image
    from fakes import make_textures_decodable

    make_textures_decodable(fake_install)
    (tmp_path / "char").mkdir()
    spec_path = tmp_path / "char" / "hoplite.json"
    spec_path.write_text(json.dumps({
        "schema": 1, "name": "Test hoplite",
        "set": {"tags": ["MAC"], "gfx": "hellenicgfx", "set_name": "hop", "base_gfx": "westerngfx",
                "unit_colours": [[150, 28, 32], [195, 150, 60], [230, 222, 200]]},
        "textures": {"schema": 1, "slots": {"1/infantry": {"ops": [
            {"op": "recolour", "region": "cloth", "colour": [226, 214, 184]}]}}},
        "hair": {"colour": [40, 30, 20]}, "test": {"sprite_techs": [2, 3, 4]}}), encoding="utf-8")
    page = window.character_page
    window.load_character(spec_path)
    assert window.unit_page.tags.text() == "MAC" and window.unit_page.set_name.text() == "hop"
    assert window.textures_page.use.isChecked() and page.name.text() == "Test hoplite"

    page.tier.setCurrentText("1")
    window.character_areas()
    assert page.body is not None, page.message.text()
    assert page.table.rowCount() >= 10 and page.regions()[0] == "area:head"
    page.set_finish("area:arms", "skin")
    page.set_finish("area:shins", "bronze")
    page.set_brush("colour", (10, 20, 30))
    page.paint(Image.new("L", page.body.diffuse.size, 255))
    border = "mask:paint_colour_0a141e.png"
    assert border in page.regions() and not (tmp_path / "char" / border[5:]).exists()  # in memory until saved
    assert page.chosen()["area:shins"]["finish"] == "bronze"  # kept across the new row
    ops = window.textures_page.recipe.slots[(1, "infantry")].ops
    assert ops[0]["region"] == "cloth"
    assert [o["region"] for o in ops if o["op"] == "recolour"][1:] == ["area:arms", "area:shins", border]
    window.character_preview()
    assert page.after.pixmap() is not None and not page.after.pixmap().isNull(), page.message.text()

    out = tmp_path / "char" / "saved.json"
    window.save_character(out)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["hair"] == {"colour": [40, 30, 20]} and doc["test"]["sprite_techs"] == [2, 3, 4]
    assert any(o.get("finish") == "skin" for o in doc["textures"]["slots"]["1/infantry"]["ops"])
    assert (tmp_path / "char" / border[5:]).is_file()  # written beside the spec on save
    window.load_character(out)
    window.character_areas()
    assert page.chosen()["area:arms"]["finish"] == "skin"
    assert page.chosen()[border]["colour"] == (10, 20, 30)
    page.apply_finishes()  # re-applying replaces, never duplicates
    ops = window.textures_page.recipe.slots[(1, "infantry")].ops
    assert sum(1 for o in ops if o["op"] == "recolour" and o.get("region") == "area:arms") == 1


def _hoplite_spec(fake_install, tmp_path):
    import json
    from fakes import make_textures_decodable

    make_textures_decodable(fake_install)
    (tmp_path / "char").mkdir(exist_ok=True)
    spec_path = tmp_path / "char" / "hoplite.json"
    spec_path.write_text(json.dumps({
        "schema": 1, "name": "Test hoplite",
        "set": {"tags": ["MAC"], "gfx": "hellenicgfx", "set_name": "hop", "base_gfx": "westerngfx",
                "unit_colours": [[150, 28, 32], [195, 150, 60], [230, 222, 200]]},
        "textures": {"schema": 1, "slots": {"1/infantry": {"ops": [
            {"op": "recolour", "region": "cloth", "colour": [226, 214, 184]}]}}}}), encoding="utf-8")
    return spec_path


def test_u3_undo_redo_reset_and_unsaved_mark(window, fake_install, tmp_path):
    """U3: finishes land in the recipe as you pick them; undo/redo walk them back and forth; reset
    goes back to the file; the page and the window title show unsaved changes."""
    page = window.character_page
    spec_path = _hoplite_spec(fake_install, tmp_path)
    window.load_character(spec_path)
    page.tier.setCurrentText("1")
    window.character_areas()
    assert page.body is not None, page.message.text()
    assert window.windowTitle() == "eu4unitgen: hoplite.json" and not window.history.dirty
    assert not page.undo_btn.isEnabled() and not page.reset_btn.isEnabled()

    def finished():
        ops = window.textures_page.recipe.slots[(1, "infantry")].ops
        return {o["region"]: o["finish"] for o in ops if o["op"] == "recolour" and "finish" in o}

    page.set_finish("area:arms", "skin")  # no Apply: it's in the recipe now
    page.set_finish("area:shins", "bronze")
    assert finished() == {"area:arms": "skin", "area:shins": "bronze"}
    assert window.windowTitle().endswith(" *") and "Unsaved" in page.dirty.text()

    page.undo_btn.click()
    assert finished() == {"area:arms": "skin"} and page.chosen() == {"area:arms": page.chosen()["area:arms"]}
    page.redo_btn.click()
    assert finished() == {"area:arms": "skin", "area:shins": "bronze"}
    assert page._finish("area:shins") == "bronze"

    page.name.setText("Renamed")
    page.name.editingFinished.emit()
    page.reset_btn.click()
    assert page.name.text() == "Test hoplite" and finished() == {} and not window.history.dirty
    page.undo_btn.click()  # the reset is a step too
    assert page.name.text() == "Renamed" and finished() == {"area:arms": "skin", "area:shins": "bronze"}

    window.save_character(tmp_path / "char" / "saved.json")
    assert window.windowTitle() == "eu4unitgen: saved.json" and not window.history.dirty


def test_u5_band_rows_and_edges(window, fake_install, tmp_path):
    """U5: bands are rows like any area; a row's grow and feather go into its finish ops and load back."""
    page = window.character_page
    window.load_character(_hoplite_spec(fake_install, tmp_path))
    page.tier.setCurrentText("1")
    window.character_areas()
    assert {"area:collar", "area:belt", "area:upper_sleeves"} <= set(page.regions())
    assert page.regions().index("area:torso") < page.regions().index("area:belt")  # broad before narrow
    page.set_finish("area:belt", "leather")
    page.set_edge("area:belt", 3, 2)
    ops = [o for o in window.textures_page.recipe.slots[(1, "infantry")].ops if o.get("region") == "area:belt"]
    assert ops and all(o["grow"] == 3 and o["feather"] == 2 for o in ops)
    page.undo_btn.click()
    assert page.chosen()["area:belt"]["grow"] == 0
    page.redo_btn.click()
    window.save_character(tmp_path / "char" / "edges.json")
    window.load_character(tmp_path / "char" / "edges.json")
    window.character_areas()
    row = page.regions().index("area:belt")
    assert page.table.cellWidget(row, 3).value() == 3 and page.table.cellWidget(row, 4).value() == 2


def test_u6_brush_paints_finishes(window, fake_install, tmp_path):
    """U6: the brush paints a finish into an in-memory mask row; painting another finish over it
    takes that spot out of the first; erasing takes paint out; undo brings a stroke back; a click
    with the brush off picks the narrowest area under it."""
    from PIL import Image

    page = window.character_page
    window.load_character(_hoplite_spec(fake_install, tmp_path))
    page.tier.setCurrentText("1")
    window.character_areas()
    size = page.body.diffuse.size
    left = Image.new("L", size, 0)
    left.paste(255, (0, 0, size[0] // 2, size[1]))
    top = Image.new("L", size, 0)
    top.paste(255, (0, 0, size[0], size[1] // 4))

    page.set_brush("bronze")
    from eu4unitgen.core.finishes import default_colour
    assert page.brush_colour_value() == default_colour("bronze", page.body.skin)  # the finish's own colour
    page.paint(left)
    page.set_brush("linen", (90, 30, 30))  # a colour of your own tints the finish
    page.paint(top)
    images = window.textures_page.recipe.images
    bronze, linen = "mask:paint_bronze.png", "mask:paint_linen_5a1e1e.png"
    assert {bronze, linen} <= set(page.regions()) and {bronze[5:], linen[5:]} <= set(images)
    assert images[bronze[5:]].getpixel((0, 0)) == 0 and images[bronze[5:]].getpixel((0, size[1] - 1)) == 255
    assert images[linen[5:]].getpixel((size[0] - 1, 0)) == 255
    assert page.chosen()[linen] == {"finish": "linen", "colour": (90, 30, 30), "grow": 0, "feather": 0}
    assert page.chosen()[bronze]["colour"] is None
    ops = window.textures_page.recipe.slots[(1, "infantry")].ops
    assert [o["region"] for o in ops if o["op"] == "recolour"][-2:] == [bronze, linen]  # paint after areas

    page.paint(Image.new("L", size, 255), erase=True)
    assert bronze not in page.regions() and linen not in page.regions() and not images
    page.undo_btn.click()
    assert {bronze, linen} <= set(page.regions())
    assert window.textures_page.recipe.images[linen[5:]].getpixel((size[0] - 1, 0)) == 255

    small = min((r for r in page.regions() if page._sizes.get(r)), key=lambda r: page._sizes[r])
    m = page._all_masks()[small]
    x, y = next((x, y) for y in range(size[1]) for x in range(size[0]) if m.getpixel((x, y)) == 255)
    page.canvas.picked.emit(x, y)
    picked = page.regions()[page.table.currentRow()]
    assert page._sizes[picked] == page._sizes[small], page.message.text()  # the narrowest area there

    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    page.set_brush("iron")
    page.paint_btn.setChecked(True)
    QTest.mousePress(page.canvas, Qt.LeftButton, Qt.NoModifier, QPoint(100, 300))
    QTest.mouseMove(page.canvas, QPoint(300, 300))
    QTest.mouseRelease(page.canvas, Qt.LeftButton, Qt.NoModifier, QPoint(300, 300))
    iron = window.textures_page.recipe.images["paint_iron.png"]
    sx = size[0] / 512
    assert iron.getpixel((int(200 * sx), int(300 * sx))) == 255 and iron.getpixel((int(200 * sx), int(100 * sx))) == 0


def test_u6_live_map_follows_the_recipe(window, fake_install, tmp_path):
    """U6: the map shows the finished texture (recipe applied, unit colours) and follows every edit."""
    page = window.character_page
    window.load_character(_hoplite_spec(fake_install, tmp_path))
    page.tier.setCurrentText("1")
    window.character_areas()
    assert page.view.currentText() == "Finished (live)"
    from PIL import Image

    w, h = page.body.diffuse.size
    x, y = 1, h // 2
    before = page.canvas.pixmap().toImage().pixelColor(int(x * 512 / w), int(y * 512 / h))
    base = page.finished_map().getpixel((x, y))
    left = Image.new("L", (w, h), 0)
    left.paste(255, (0, 0, w // 2, h))
    page.set_brush("colour", (0, 230, 0))
    page.paint(left)
    px = page.finished_map().getpixel((x, y))
    assert px[1] > 150 and px[0] < 80, px
    after = page.canvas.pixmap().toImage().pixelColor(int(x * 512 / w), int(y * 512 / h))
    assert after != before and after.green() > after.red()
    page.undo_btn.click()
    assert page.finished_map().getpixel((x, y)) == base
    page.view.setCurrentText("Areas")
    assert not page.canvas.pixmap().isNull()


def test_u4_sections_and_a_tooltip_on_every_control(window, fake_install, tmp_path):
    """U4: the page is in five numbered sections, and every control says what it does."""
    from PySide6.QtWidgets import QAbstractButton, QComboBox, QGroupBox, QLineEdit, QSpinBox

    page = window.character_page
    window.load_character(_hoplite_spec(fake_install, tmp_path))
    window.character_areas()
    titles = sorted(g.title() for g in page.findChildren(QGroupBox))
    assert titles == ["1  Character file", "2  Body", "3  Areas and finishes", "4  Paint", "5  Preview"]
    bare = [f"{type(w).__name__} {getattr(w, 'text', lambda: '')()!r}"
            for kind in (QAbstractButton, QComboBox, QSpinBox, QLineEdit) for w in page.findChildren(kind)
            if not w.toolTip() and not isinstance(w.parent(), (QComboBox, QSpinBox)) and w.parent() is not page.table]
    assert not bare, bare
    assert "Part of the torso" in page.table.cellWidget(page.regions().index("area:collar"), 0).toolTip()


def test_u1_wheel_guard_splitters_and_empty_hint(window, qapp):
    """U1: the wheel never changes an unfocused dropdown; a big render keeps the areas table usable;
    the empty map explains itself."""
    from types import SimpleNamespace
    from PIL import Image
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    page = window.character_page
    assert "isn't loaded" in page.canvas.text()
    combo = page.role
    window.resize(1100, 760)
    window.steps.setCurrentRow(1)
    window.show()
    qapp.processEvents()

    def wheel(widget):
        ev = QWheelEvent(QPointF(5, 5), QPointF(widget.mapToGlobal(QPoint(5, 5))), QPoint(0, 0), QPoint(0, -120),
                         Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
        qapp.sendEvent(widget, ev)

    before = combo.currentIndex()
    combo.clearFocus()
    wheel(combo)
    assert combo.currentIndex() == before  # unfocused: ignored
    assert combo.focusPolicy() == Qt.StrongFocus

    page.show_render(SimpleNamespace(image=Image.new("RGB", (720, 2400), (90, 90, 90)), path="x.png", notes=[]))
    qapp.processEvents()
    assert page.table.height() >= 120 and page.after_scroll.height() >= 80
    window.hide()


def test_f8_home_screen_reads_rewrites_and_generates(window, fake_install, tmp_path, monkeypatch):
    """F8: the window opens on Home; typing makes chips; x and swaps rewrite the text; a leftover
    can be taught; template chips; the nation picker; Generate makes the spec the pages show."""
    from fakes import make_textures_decodable

    from eu4unitgen.core.look import templates

    make_textures_decodable(fake_install)
    monkeypatch.setitem(templates()["phalangite"], "body_base", "generic_01")  # the fake install's body,
    monkeypatch.setitem(templates()["phalangite"], "steps", {})  # which isn't skinned: no helmet to remove
    monkeypatch.setitem(templates()["phalangite"]["roles"]["infantry"], "helmet", None)
    home = window.home
    assert window.views.currentWidget() is home
    home.set_text("Hoplites with red crests, no shield, long spears, doru, bronze greaves")
    labels = [i.text() for i in home.result.items]
    assert labels[:3] == ["Hoplite", "Crest", "Red crest"] and home.result.leftovers == ["doru"]
    assert home.understood_flow.count() == len(labels) and home.left_row.isVisibleTo(window)

    home._remove(labels.index("Crest"), False)  # the chip's x
    assert home.text() == "Hoplites, no shield, long spears, doru, bronze greaves"
    home.teach("doru", "weapon:dory")
    assert home.result.leftovers == [] and (tmp_path / "appdata" / "eu4unitgen" / "vocabulary.json").is_file()
    home.use_template("peltast")
    assert home.text().startswith("Peltast") and home.result.look().template == "peltast"
    home.set_text("bronze gr")
    assert home.suggest_row.isVisibleTo(window)
    home.complete("greaves")
    assert home.text() == "bronze greaves "

    window.settings_page.load.click()  # fills the nation picker
    assert home.nation.count() > 0
    home.set_nation("MAC")
    assert home.tag() == "MAC"
    home.set_text("Phalangites with pikes and small shields, crimson tunics; same look on all tiers")
    assert home.tiers.currentData() == "same"
    from test_gallery import GalleryRunner

    runner = GalleryRunner()
    monkeypatch.setattr(window.session, "runner", lambda: runner)
    home.generate.click()
    spec = window.character_page.spec
    assert spec is not None, window.result_page.message.text()
    assert spec.set["tags"] == ["MAC"] and spec.set["unit_names"]["infantry"] == "Pezhetairoi"
    parts = {p["kind"] for p in spec.loadout["slots"]["4/infantry"]["parts"]}
    assert {"sarissa", "hoplon"} <= parts
    assert window.views.currentWidget() is window.result_page and window.windowTitle().endswith(" *")
    assert window.textures_page.use.isChecked() and window.parts_page.use.isChecked()


def test_f10_result_screen_gallery_parts_colours_and_names(window, fake_install, tmp_path, monkeypatch):
    """F10: Generate shows the Result screen: 12 tiles, chips, names; a tile picks the unit type and
    tier the Parts cards edit ("this tier only" or all); colours, emblem and names land in the spec;
    a chip's x rewrites the description on both screens."""
    from pathlib import Path
    from fakes import make_textures_decodable
    from test_gallery import GalleryRunner
    from eu4unitgen.core.look import templates

    make_textures_decodable(fake_install)
    monkeypatch.setitem(templates()["hoplite"], "body_base", "generic_01")  # the fake install's body,
    monkeypatch.setitem(templates()["hoplite"], "steps", {})  # which isn't skinned: no helmet to remove
    monkeypatch.setitem(templates()["hoplite"]["roles"]["infantry"], "helmet", None)
    runner = GalleryRunner()
    monkeypatch.setattr(window.session, "runner", lambda: runner)
    window.settings_page.load.click()
    home, page = window.home, window.result_page
    home.set_text("Hoplites with round shields and long spears; bronze greaves. Cavalry with red crests")
    home.set_nation("MAC")
    window.home_generate()  # (a click would hide an exception in the slot)
    assert window.views.currentWidget() is page, page.message.text()
    assert sum(1 for t in page.tiles.values() if t.pixmap() and not t.pixmap().isNull()) == 12
    assert len([j for j, _ in runner.calls if j == "render_unit"]) == 3  # one run per row: tiles fill in
    assert page.names["infantry"].text() == "Hoplitai" and "no problems" in page.status.text()
    assert page.export.isEnabled() and page.understood_flow.count() >= 4

    def slot(key):
        return {p["kind"]: p for p in window.character_page.spec.loadout["slots"].get(key, {}).get("parts", [])}

    page.tiles[("cavalry", 2)].clicked.emit()
    assert page.selected == ("cavalry", 2) and "Cavalry, tier 2" in page.showing.text()
    assert not page.cards["shield"].isVisible()  # riders carry no shield
    page.scope_one.setChecked(True)
    page._part("cavalry", "weapon", "xiphos")
    assert "xiphos" in slot("2/rider") and "xyston" in slot("3/rider")  # tier 2 only
    page.tiles[("infantry", 1)].clicked.emit()
    page.scope_all.setChecked(True)
    page._shield("infantry", "hoplon", {"radius": 2.6})
    assert all(slot(f"{t}/infantry")["hoplon"]["params"]["radius"] == 2.6 for t in (1, 2, 3, 4))
    page._finish("arms", "iron")
    ops = window.character_page.spec.textures["slots"]["3/infantry"]["ops"]
    assert any(o.get("region") == "area:arms" and o.get("finish") == "iron" for o in ops)

    from eu4unitgen.core.army import Edit

    window.result_edit(Edit("unit_colour", (0,), [10, 20, 30]))
    assert window.character_page.spec.set["unit_colours"][0] == [10, 20, 30]
    window.result_emblem("star")
    blazon = slot("4/infantry")["hoplon"]["blazon"]
    assert Path(blazon).is_file() and "emblem_star" in blazon
    page.names["cavalry"].setText("Hetairoi")
    page._name_changed("cavalry")
    assert window.character_page.spec.set["unit_names"]["cavalry"] == "Hetairoi"

    interp = window.army.interpretation(home.interp)
    crest = next(n for n, i in enumerate(interp.items) if i.kind == "crest")
    window.result_chip_remove(crest, False)
    assert "crest" not in window.army.text.lower() and home.text() == window.army.text
    assert window.army.edits and window.character_page.spec.set["unit_names"]["cavalry"] == "Hetairoi"  # edits kept


def test_f11_export_confirms_then_writes_and_test_mod_stages(window, fake_install, tmp_path, monkeypatch):
    """F11: Export shows the dry run's file list to confirm (cancel writes nothing), then writes the
    package; Test in game stages the test mod and says where to copy it."""
    from fakes import make_textures_decodable
    from test_gallery import GalleryRunner
    from eu4unitgen.core.look import templates

    make_textures_decodable(fake_install)
    monkeypatch.setitem(templates()["hoplite"], "body_base", "generic_01")
    monkeypatch.setitem(templates()["hoplite"], "steps", {})
    monkeypatch.setitem(templates()["hoplite"]["roles"]["infantry"], "helmet", None)
    runner = GalleryRunner()
    monkeypatch.setattr(window.session, "runner", lambda: runner)
    window.settings_page.load.click()
    home, page = window.home, window.result_page
    home.set_text("Hoplites with round shields and long spears")
    home.set_nation("AAA")  # owns land at the start (the test mod needs that)
    window.home_generate()
    assert page.export.isEnabled(), page.status.text()

    asked = []
    monkeypatch.setattr(window, "confirm", lambda title, text, details="": asked.append((text, details)) or False)
    window.result_export()
    out = tmp_path / "out" / "aaa"
    assert asked and "Write" in asked[0][0] and "create" in asked[0][1] and not out.exists()
    assert "cancelled" in page.message.text()
    monkeypatch.setattr(window, "confirm", lambda *a, **k: True)
    window.result_export()
    assert "Exported" in page.message.text() and (out / "INTEGRATION.md").is_file(), page.message.text()
    assert any(p.suffix == ".yml" for p in out.rglob("*"))

    window.result_test()
    assert "Test mod ready" in page.message.text(), page.message.text()
    assert any((tmp_path / "stage").rglob("*.mod"))
