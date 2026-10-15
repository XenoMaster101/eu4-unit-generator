"""blender/runner.py: version pin, job protocol, failures; a real check when Blender is installed."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from eu4unitgen.blender import runner as rmod
from eu4unitgen.blender.runner import BlenderError, BlenderRunner, addon_version, parse_version
from eu4unitgen.core.project import BLENDER_VERSION, default_blender_path

SRC = Path(__file__).resolve().parents[1] / "src" / "eu4unitgen"


def test_parse_versions():
    assert parse_version("TBBmalloc: noise\nBlender 4.2.23 LTS\n\tbuild date") == "4.2.23"
    assert parse_version("Python 3.14") is None
    assert addon_version([0, 91, 0]) == "0.91" and addon_version("0.91.0") == "0.91" and addon_version((1, 2, 3)) == "1.2.3"


class FakeBlender:
    """Stands in for subprocess.run: answers --version and runs 'jobs' by writing a result."""

    def __init__(self, version="4.2.23", result: dict | None = None, code=0, timeout=False):
        self.version, self.result, self.code, self.timeout, self.calls = version, result, code, timeout, []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        if self.timeout:
            raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
        if cmd[1:] == ["--version"]:
            return subprocess.CompletedProcess(cmd, 0, f"Blender {self.version} LTS\n", "")
        args = json.loads(Path(cmd[cmd.index("--") + 1]).read_text(encoding="utf-8"))
        if self.result is not None:
            Path(args["result_path"]).write_text(json.dumps(self.result), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, self.code, "line 1\nTraceback: boom\n", "")


@pytest.fixture
def exe(tmp_path) -> Path:
    p = tmp_path / "blender.exe"
    p.write_bytes(b"")
    return p


def test_job_protocol_and_log(monkeypatch, exe, tmp_path):
    fake = FakeBlender(result={"found": True, "enabled": True, "module": "bl_ext.user_default.io_pdx_mesh",
                               "version": [0, 91, 0]})
    monkeypatch.setattr(rmod.subprocess, "run", fake)
    r = BlenderRunner(exe, log_dir=tmp_path / "logs")
    res = r.check_addon()
    assert res["addon_version"] == "0.91" and res["blender"] == BLENDER_VERSION
    cmd = fake.calls[-1]
    assert cmd[1:5] == ["--background", "--python-exit-code", "1", "--python"]
    assert cmd[5].endswith("check_addon.py") and cmd[6] == "--" and "--factory-startup" not in cmd
    assert (tmp_path / "logs" / "check_addon.log").read_text(encoding="utf-8").startswith("line 1")


@pytest.mark.parametrize("fake,match", [
    (FakeBlender(version="4.3.0"), "pinned to"),
    (FakeBlender(result=None, code=1), "failed \\(exit 1, no result\\)"),
    (FakeBlender(result={"found": True, "enabled": False, "module": "m"}), "not enabled"),
    (FakeBlender(result={"found": True, "enabled": True, "module": "m", "version": [0, 90]}), "pinned to 0.91"),
    (FakeBlender(result={"found": False}), "isn't installed"),
    (FakeBlender(timeout=True), "timed out"),
])
def test_failures(monkeypatch, exe, fake, match):
    monkeypatch.setattr(rmod.subprocess, "run", fake)
    with pytest.raises(BlenderError, match=match):
        BlenderRunner(exe).check_addon()


def test_missing_blender_and_job(tmp_path, monkeypatch, exe):
    with pytest.raises(BlenderError, match="no Blender executable"):
        BlenderRunner(None)
    with pytest.raises(BlenderError, match="not found"):
        BlenderRunner(tmp_path / "nope.exe").version()
    monkeypatch.setattr(rmod.subprocess, "run", FakeBlender())
    with pytest.raises(BlenderError, match="no Blender job"):
        BlenderRunner(exe).run_job("does_not_exist")


def test_bpy_only_in_jobs():
    """CLAUDE.md: bpy is never imported in the app process; only Blender job scripts use it."""
    offenders = [p for p in SRC.rglob("*.py") if "jobs" not in p.parts
                 and any(line.strip().startswith(("import bpy", "from bpy", "import addon_utils"))
                         for line in p.read_text(encoding="utf-8").splitlines())]
    assert offenders == []


@pytest.mark.skipif(default_blender_path() is None or not default_blender_path().is_file(),
                    reason="pinned Blender not installed")
def test_real_blender_check(tmp_path):
    res = BlenderRunner(default_blender_path(), log_dir=tmp_path).check_addon()
    assert res["enabled"] and res["module"].endswith("io_pdx_mesh")
