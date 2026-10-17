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
    page.save_mask("border", Image.new("L", page.body.diffuse.size, 255))
    assert (tmp_path / "char" / "border.png").is_file() and "mask:border.png" in page.regions()
    assert page.chosen()["area:shins"]["finish"] == "bronze"  # kept across the new row
    page.set_finish("mask:border.png", "colour", (10, 20, 30))
    page.apply_finishes()
    ops = window.textures_page.recipe.slots[(1, "infantry")].ops
    assert ops[0]["region"] == "cloth"
    assert [o["region"] for o in ops if o["op"] == "recolour"][1:] == ["area:arms", "area:shins", "mask:border.png"]
    window.character_preview()
    assert page.after.pixmap() is not None and not page.after.pixmap().isNull(), page.message.text()

    out = tmp_path / "char" / "saved.json"
    window.save_character(out)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["hair"] == {"colour": [40, 30, 20]} and doc["test"]["sprite_techs"] == [2, 3, 4]
    assert any(o.get("finish") == "skin" for o in doc["textures"]["slots"]["1/infantry"]["ops"])
    window.load_character(out)
    window.character_areas()
    assert page.chosen()["area:arms"]["finish"] == "skin"
    assert page.chosen()["mask:border.png"]["colour"] == (10, 20, 30)
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
