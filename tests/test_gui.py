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
        "1. Settings", "2. Unit set", "3. Textures", "4. Validate", "5. Generate", "6. Test mod"]
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
    window.unit_page.set_name.setText("both")
    window.validate_page.run.click()
    assert "leave the set name blank" in window.validate_page.message.text()


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
