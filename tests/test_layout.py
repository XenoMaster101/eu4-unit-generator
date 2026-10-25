"""The one-root layout (core.layout, SPEC P5): every path the installed app uses resolves inside the root."""
import json
import os
import tempfile
from pathlib import Path

import pytest

from eu4unitgen.core import config, layout, setup
from eu4unitgen.ref3d import trellis


@pytest.fixture
def rooted(tmp_path, monkeypatch):
    r = tmp_path / "EU4 Unit Generator"
    for f in layout.FOLDERS:
        (r / f).mkdir(parents=True)
    (r / "app" / layout.MARKER).write_text(json.dumps({"version": "0.1.0"}), encoding="utf-8")
    monkeypatch.setenv("EU4UNITGEN_ROOT", str(r))
    for k in ("EU4UNITGEN_CONFIG_DIR", "EU4UNITGEN_WORKSPACE", "EU4UNITGEN_AI_PYTHON", "BLENDER_PATH"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(config, "checkout_root", lambda: None)
    yield r
    tempfile.tempdir = None


def test_every_path_inside_the_root(rooted):
    cfg = config.load()
    assert config.config_dir() == rooted / "settings"
    assert cfg.work() == rooted / "work"
    assert cfg.projects() == rooted / "work" / "projects"
    assert cfg.ai_root_path() == rooted / "ai"
    assert setup.default_ai_root() == rooted / "ai"
    assert config.logs_dir() == rooted / "settings" / "logs"
    assert "ai_folder" not in [s.key for s in setup.steps()]  # the AI folder is fixed: ai\


def test_blender_and_distro_from_the_root(rooted):
    exe = layout.blender_exe()
    assert exe.parent == rooted / "tools" / "blender"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    assert config.load().blender() == exe
    assert setup.io_pdx_mesh_dir().is_relative_to(rooted / "tools" / "blender" / "portable")
    assert setup.distro() == layout.DISTRO
    layout.update_marker(wsl_distro="EU4UnitGen")
    assert config.load().wsl_base() == ["wsl.exe", "-d", "EU4UnitGen"]


def test_temp_and_caches_inside_the_root(rooted, monkeypatch):
    for k in ("TEMP", "TMP", "TMPDIR", "PIP_CACHE_DIR", "HF_HOME"):
        monkeypatch.delenv(k, raising=False)
    old = rooted / "settings" / "tmp" / "old"
    old.mkdir(parents=True)
    os.utime(old, (0, 0))
    fresh = rooted / "settings" / "tmp" / "running_job"
    fresh.mkdir()
    tmp = layout.use_root_temp()
    assert tmp == rooted / "settings" / "tmp"
    assert Path(tempfile.gettempdir()) == tmp and os.environ["TEMP"] == str(tmp)
    assert os.environ["HF_HOME"] == str(rooted / "ai" / "hf_home")
    assert not old.exists() and fresh.exists()  # a day-old leftover goes; another window's job stays


def test_plans_stay_inside_the_root(rooted):
    b = setup.plan("blender")
    assert b[0].download[0] == setup.BLENDER_ZIP and Path(b[0].download[1]).is_relative_to(rooted)
    assert not any(c.admin for c in b)  # portable: no administrator rights
    ub = setup.plan("ubuntu")[-2].argv
    assert ub[ub.index("--location") + 1] == str(rooted / "wsl") and ub[ub.index("--name") + 1] == layout.DISTRO
    w = setup.plan("weights", token="hf_x")
    assert w[1].env == {"HF_HOME": str(rooted / "ai" / "hf_home"), "HF_TOKEN": "hf_x"}


def test_wsl_commands_quote_paths_with_spaces():
    cmd = trellis.wsl_job_cmd("~/eu4unitgen_ai/trellis2_env/bin/python", Path(r"D:\EU4 Unit Generator\app\job.py"),
                              Path(r"D:\EU4 Unit Generator\work\a b.json"), Path(r"D:\EU4 Unit Generator\work\x.log"))
    assert cmd.startswith('cd /tmp && "$HOME"/eu4unitgen_ai/trellis2_env/bin/python ')
    assert "'/mnt/d/EU4 Unit Generator/app/job.py'" in cmd and "'/mnt/d/EU4 Unit Generator/work/a b.json'" in cmd
    tr = setup.plan("trellis_env", root=Path(r"D:\EU4 Unit Generator\ai"))
    assert "AI_WIN='/mnt/d/EU4 Unit Generator/ai'" in tr[-1].argv[-1]


def test_no_root_keeps_the_old_paths(monkeypatch, tmp_path):
    monkeypatch.delenv("EU4UNITGEN_ROOT", raising=False)
    monkeypatch.setattr(layout, "_checkout", lambda: None)
    assert layout.root() is None or (layout.root() / "app" / layout.MARKER).is_file()


def test_unzip_blender_checks_the_checksum(tmp_path):
    import zipfile
    z = tmp_path / "b.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("blender-4.2.23-windows-x64/blender.exe", b"x")
    with pytest.raises(SystemExit):
        setup.unzip_blender(z, tmp_path / "blender")
