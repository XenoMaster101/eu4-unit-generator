"""Long AI runs: checkpoints, the drive watch, and the runner's drive-lost error (2026-10-20: the
drive holding the AI environment dropped out mid-run)."""
import sys
import time

import pytest

from eu4unitgen.ai import env as ai_env
from eu4unitgen.ai.env import AIDriveLost, AIJobRunner
from eu4unitgen.ai.jobs import _resilience as R


def test_checkpoints_round_trip_and_leave_no_temp_files(tmp_path):
    ck = R.Checkpoints(tmp_path / "out")
    assert ck.load("b2_r0") is None
    ck.save("b2_r0", {"x": [1.0, 2.5], "done": False})
    ck.save("b2_r0", {"x": [3.0], "done": True})
    assert ck.load("b2_r0") == {"x": [3.0], "done": True}
    assert [p.name for p in (tmp_path / "out" / "checkpoints").iterdir()] == ["b2_r0.json"]


def test_drive_watch_raises_once_the_probe_is_gone(tmp_path):
    probe = tmp_path / "python.exe"
    probe.write_bytes(b"MZ")
    w = R.DriveWatch(probe, interval=0.05)
    w.check()
    probe.unlink()
    deadline = time.time() + 5
    while not w.lost.is_set() and time.time() < deadline:
        time.sleep(0.05)
    with pytest.raises(R.DriveLost):
        w.check()
    w.stop()


def _runner(tmp_path, monkeypatch, body):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "job.py").write_text(body, encoding="utf-8")
    monkeypatch.setattr(ai_env, "JOBS_DIR", jobs)
    monkeypatch.setenv("EU4UNITGEN_AI_ROOT", str(tmp_path / "ai"))
    return AIJobRunner(sys.executable)


def test_runner_reports_a_job_stopped_for_the_drive(tmp_path, monkeypatch):
    r = _runner(tmp_path, monkeypatch, f"print('STOPPED'); raise SystemExit({R.EXIT_DRIVE_LOST})\n")
    with pytest.raises(AIDriveLost, match="dropped out"):
        r.run_job("job", {"out_dir": str(tmp_path / "ai" / "out")})


def test_runner_blames_the_drive_when_it_is_gone_after_a_failure(tmp_path, monkeypatch):
    r = _runner(tmp_path, monkeypatch, "raise OSError(19, 'No such device')\n")
    with pytest.raises(ai_env.AIJobError) as e:
        r.run_job("job", {"out_dir": str(tmp_path / "ai" / "out")})
    assert not isinstance(e.value, AIDriveLost)  # drive fine: an ordinary failure
    calls = iter([True, False])  # readable at the start, gone afterwards
    monkeypatch.setattr(ai_env, "drive_ok", lambda p: next(calls))
    with pytest.raises(AIDriveLost):
        r.run_job("job", {"out_dir": str(tmp_path / "ai" / "out")})


def test_runner_does_not_start_without_the_drive(tmp_path, monkeypatch):
    r = _runner(tmp_path, monkeypatch, "raise SystemExit(0)\n")
    monkeypatch.setattr(ai_env, "drive_ok", lambda p: False)
    with pytest.raises(AIDriveLost, match="not started"):
        r.run_job("job", {"out_dir": str(tmp_path / "ai" / "out")})
