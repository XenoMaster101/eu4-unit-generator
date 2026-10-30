"""Release 1.0 R2: the updater reads the latest GitHub release's installer and checksums, refuses a download that
doesn't match, and runs the installer in place."""
import hashlib
import io

import pytest

from eu4unitgen.core import updates


def release(sha):
    return {"tag_name": "v1.0.1", "html_url": "https://x/rel", "body": "fixes",
            "assets": [{"name": "eu4unitgen-1.0.1-setup.exe", "browser_download_url": "https://x/setup", "size": 5},
                       {"name": "SHA256SUMS.txt", "browser_download_url": "https://x/sums"},
                       {"name": "eu4unitgen-1.0.1-windows.zip", "browser_download_url": "https://x/zip"}]}


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
    assert u.newer and u.setup_name == "eu4unitgen-1.0.1-setup.exe" and u.sums_url and u.notes == "fixes"
    assert "in place" in u.text
    files = {"https://x/sums": f"{good}  eu4unitgen-1.0.1-setup.exe\n{'0' * 64}  other\n".encode(),
             "https://x/setup": data}
    seen = []
    out = updates.download(u, tmp_path, progress=lambda d, t: seen.append(d), opener=lambda url: Resp(files[url]))
    assert out.read_bytes() == data and seen[-1] == len(data)
    cmd = updates.install_command(out, tmp_path / "root")
    assert cmd[0] == str(out) and "/SILENT" in cmd and "/RELAUNCH=1" in cmd and f"/DIR={tmp_path / 'root'}" in cmd


def test_a_download_that_doesnt_match_its_checksum_is_refused_and_deleted(tmp_path):
    u = updates.check("me/app", "1.0.0", fetch=lambda url: release("x"))
    files = {"https://x/sums": f"{'a' * 64}  eu4unitgen-1.0.1-setup.exe\n".encode(), "https://x/setup": b"EVIL"}
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
    assert updates.built_repo() == "XenoMaster101/eu4-unit-generator" and __version__ == "1.0.2"
    assert updates.expected_sha256(f"{'b' * 64} *a.exe\n", "a.exe") == "b" * 64


def test_every_setup_step_lists_its_downloads_before_it_runs():
    from eu4unitgen.core import setup
    keys = {s.key for s in setup.steps()} - {"ai_folder"}
    assert keys <= {d.step for d in setup.DOWNLOADS}
    assert all(d.size_mb > 0 and d.licence and d.source and d.where for d in setup.DOWNLOADS)
    required = {d.step for d in setup.DOWNLOADS if d.required}
    assert required == {"installer", "blender", "io_pdx_mesh"}  # the AI is only for parts from pictures
    html = setup.downloads_html()
    assert "Needed for everything" in html and "the AI pack" in html and "DINOv3 License" in html
    assert "only to make parts" not in html  # said once, in the group's heading, not on every row
