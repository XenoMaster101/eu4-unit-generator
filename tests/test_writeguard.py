"""Track M runs may write only inside the project or the AI root (user rule, 2026-10-20)."""
import sys

import pytest

from eu4unitgen.ai import env as ai_env
from eu4unitgen.ai.env import AIJobRunner
from eu4unitgen.ref3d import writeguard
from eu4unitgen.ref3d.writeguard import WriteGuardError, check_writes


def test_paths_outside_the_roots_are_refused(tmp_path):
    roots = [tmp_path / "project", tmp_path / "ai"]
    check_writes({"out_dir": str(tmp_path / "project" / "a" / "b"), "image": "C:/anything/read.png"}, roots)
    check_writes({"nested": {"output": str(tmp_path / "ai" / "x.glb")}}, roots)
    with pytest.raises(WriteGuardError, match="out_dir"):
        check_writes({"out_dir": str(tmp_path / "elsewhere")}, roots)
    with pytest.raises(WriteGuardError, match="elsewhere"):  # ".." is resolved before the check
        check_writes({"out_dir": str(tmp_path / "project" / ".." / "elsewhere")}, roots)


def test_git_bash_style_path_is_refused(tmp_path):
    # the 2026-10-20 incident: "/c/Users/..." given to a Windows python lands in C:\c\Users\...
    with pytest.raises(WriteGuardError):
        check_writes({"out_dir": "/c/Users/someone/scratch"}, [tmp_path])


def test_roots_come_from_the_env_python_or_overrides(tmp_path, monkeypatch):
    monkeypatch.delenv("EU4UNITGEN_AI_ROOT", raising=False)
    py = tmp_path / "ai" / "triposg_env" / "Scripts" / "python.exe"
    assert writeguard.ai_root(py) == (tmp_path / "ai").resolve()
    monkeypatch.setenv("EU4UNITGEN_AI_ROOT", str(tmp_path / "other"))
    assert writeguard.ai_root(py) == (tmp_path / "other").resolve()
    assert writeguard.project_root() / "src" / "eu4unitgen" in writeguard.Path(writeguard.__file__).resolve().parents


def test_ai_runner_refuses_before_starting_and_keeps_temp_in_the_ai_root(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "echo.py").write_text(
        "import json, sys, pathlib\n"
        "a = json.load(open(sys.argv[1], encoding='utf-8'))\n"
        "json.dump({'tmp': str(pathlib.Path(a['result_path']).parent.parent)}, open(a['result_path'], 'w'))\n",
        encoding="utf-8")
    monkeypatch.setattr(ai_env, "JOBS_DIR", jobs)
    monkeypatch.setenv("EU4UNITGEN_AI_ROOT", str(tmp_path / "ai"))
    r = AIJobRunner(sys.executable)
    assert r.run_job("echo", {"out_dir": str(tmp_path / "ai" / "out")}).result["tmp"] == str((tmp_path / "ai" / "tmp").resolve())
    with pytest.raises(WriteGuardError):
        r.run_job("echo", {"out_dir": str(tmp_path / "nope")})
