"""Release 1.0 R2: the updater reads the latest GitHub release's installer and checksums, refuses a download that
doesn't match, and runs the installer in place."""
import hashlib
import io

import pytest

from eu4unitgen.core import updates


def release(sha):
    return {"tag_name": "v1.0.1", "html_url": "https://x/rel", "body": "fixes",
            "assets": [{"name": "eu4unitgen-1.0.1-setup.exe", "browser_download_url": "https://x/exe"},
                       {"name": "SHA256SUMS.txt", "browser_download_url": "https://x/sums"},
                       {"name": "eu4unitgen-1.0.1-windows.zip", "browser_download_url": "https://x/setup", "size": 5}]}


class Resp(io.BytesIO):
    headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_a_newer_release_with_an_installer_can_be_installed_in_place(tmp_path):
    data = b"SETUP"
    good = hashlib.sha256(data).hexdigest()
    u = updates.check("me/app", "1.0.0", fetch=lambda url: release(good))
    assert u.newer and u.setup_name == "eu4unitgen-1.0.1-windows.zip" and u.sums_url and u.notes == "fixes"
    assert "in place" in u.text
    files = {"https://x/sums": f"{good}  eu4unitgen-1.0.1-windows.zip\n{'0' * 64}  other\n".encode(),
             "https://x/setup": data}
    seen = []
    out = updates.download(u, tmp_path, progress=lambda d, t: seen.append(d), opener=lambda url: Resp(files[url]))
    assert out.read_bytes() == data and seen[-1] == len(data)
    cmd = updates.install_command(tmp_path / "files", tmp_path / "root", 123, tmp_path / "pythonw.exe")
    assert cmd[0] == "powershell" and "Wait-Process -Id 123" in cmd[-1] and "install.ps1" in cmd[-1]
    assert f"-Root '{tmp_path / 'root'}'" in cmd[-1] and "Start-Process" in cmd[-1]
    assert "-NoSystem" not in cmd[-1] and "update.log" in cmd[-1]
    other = updates.install_command(tmp_path / "files", tmp_path / "root", 123, tmp_path / "pythonw.exe", system=False)
    assert "-NoSystem" in other[-1]  # a copy Windows doesn't list keeps the user's shortcuts alone


def test_the_update_uses_the_zip_and_unpacks_its_installer(tmp_path):
    # Smart App Control blocks an unsigned setup.exe; the zip's install.ps1 runs on the app's signed Python
    import zipfile
    u = updates.check("me/app", "1.0.0", fetch=lambda url: release("x"))
    assert u.setup_name.endswith("-windows.zip")
    z = tmp_path / "eu4unitgen-1.0.1-windows.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("EU4 Unit Generator/files/install.ps1", "# installer")
    assert (updates.unpack(z) / "install.ps1").is_file()


def test_a_download_that_doesnt_match_its_checksum_is_refused_and_deleted(tmp_path):
    u = updates.check("me/app", "1.0.0", fetch=lambda url: release("x"))
    files = {"https://x/sums": f"{'a' * 64}  eu4unitgen-1.0.1-windows.zip\n".encode(), "https://x/setup": b"EVIL"}
    with pytest.raises(ValueError, match="checksum"):
        updates.download(u, tmp_path, opener=lambda url: Resp(files[url]))
    assert not list(tmp_path.iterdir())


def test_a_release_without_an_installer_points_to_its_page():
    u = updates.check("me/app", "1.0.0", fetch=lambda url: {"tag_name": "v1.1.0", "html_url": "https://x/rel"})
    assert u.newer and not u.setup_url and "page" in u.text
    with pytest.raises(ValueError):
        updates.download(u)


def test_this_copy_knows_its_repository_and_version():
    from eu4unitgen import __version__
    assert updates.built_repo() == "XenoMaster101/eu4-unit-generator" and __version__ == "1.0.7"
    assert updates.expected_sha256(f"{'b' * 64} *a.exe\n", "a.exe") == "b" * 64


def test_every_setup_step_lists_its_downloads_before_it_runs():
    from eu4unitgen.core import setup
    keys = {s.key for s in setup.steps()} - {"ai_folder"}
    assert keys <= {d.step for d in setup.DOWNLOADS}
    assert all(d.size_mb > 0 and d.licence and d.source and d.where for d in setup.DOWNLOADS)
    assert all(d.required for d in setup.DOWNLOADS)  # every one needed, each with its reason (the user, 2026-10-30)
    assert all(len(d.why) > 30 for d in setup.DOWNLOADS)
    html = setup.downloads_html()
    assert "In all:" in html and "DINOv3 License" in html and "optional" not in html.lower()


def test_setup_never_takes_another_blender_version(tmp_path, monkeypatch):
    monkeypatch.delenv("BLENDER_PATH", raising=False)
    # 1.0.2 on the user's PC: Blender 5.2 was taken and saved; io_pdx_mesh can't run in it ("No module named 'imp'")
    from pathlib import Path
    from eu4unitgen.core import config, install
    b52, b42 = tmp_path / "Blender 5.2" / "blender.exe", tmp_path / "Blender 4.2" / "blender.exe"
    exe, note = install.find_blender([(b52, "5.2.2")])
    assert exe is None and "5.2.2" in note and "can't use" in note
    assert install.find_blender([(b52, "5.2.2"), (b42, "4.2.23")])[0] == b42
    assert not config.blender_ok(b52) and config.blender_ok(b42)
    assert config.blender_ok(Path(r"D:\EU4 Unit Generator\tools\blender\blender.exe"))  # the app's own


def test_the_setup_wizard_closes(monkeypatch):
    # 1.0.0-1.0.2: the wizard kept its steps' state in ``self.done``, hiding QDialog.done(): Close crashed
    # ("'dict' object is not callable") on the user's PC
    from PySide6.QtWidgets import QApplication
    from eu4unitgen.core import setup
    from eu4unitgen.gui.setup_wizard import SetupWizard
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(setup, "state", lambda: {"blender": True})
    w = SetupWizard()
    keys = [s.key for s in w.steps]
    assert keys[:3] == ["overview", "blender", "io_pdx_mesh"] and "ai_pack" not in keys
    w.list.setCurrentRow(0)
    w.run_step()  # the overview goes to the first step not done
    assert w.steps[w.list.currentRow()].key == "io_pdx_mesh"
    w.accept()  # crashed
    app.processEvents()
