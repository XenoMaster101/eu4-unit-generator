"""Track C, test 15: the TripoSG runner starts a job script in another Python and reads its
result JSON (a fake job here; the real one needs the TripoSG environment and a CUDA GPU)."""
import sys
from pathlib import Path

import pytest

from eu4unitgen.ai import triposg
from eu4unitgen.ai.triposg import TripoSGError, TripoSGRunner


def test_runner_needs_its_three_paths(tmp_path, monkeypatch):
    for v in ("EU4UNITGEN_TRIPOSG_PYTHON", "EU4UNITGEN_TRIPOSG_REPO", "EU4UNITGEN_TRIPOSG_WEIGHTS"):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(TripoSGError, match="isn't set up"):
        TripoSGRunner()
    with pytest.raises(TripoSGError, match="weights folder not found"):
        TripoSGRunner(sys.executable, tmp_path, tmp_path / "nope")


def test_runner_passes_args_and_reads_the_result(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "echo.py").write_text(
        "import json, sys\n"
        "a = json.load(open(sys.argv[1], encoding='utf-8'))\n"
        "json.dump({'got': a['x'], 'repo': a['repo']}, open(a['result_path'], 'w', encoding='utf-8'))\n"
        "print('hello from the job')\n", encoding="utf-8")
    (jobs / "fails.py").write_text("raise SystemExit('no GPU')\n", encoding="utf-8")
    monkeypatch.setattr(triposg, "JOBS_DIR", jobs)
    r = TripoSGRunner(sys.executable, tmp_path, tmp_path)
    res = r.run_job("echo", {"x": 7})
    assert res.result == {"got": 7, "repo": str(tmp_path)} and "hello from the job" in res.log
    with pytest.raises(TripoSGError, match="no GPU"):
        r.run_job("fails", {})
    with pytest.raises(TripoSGError, match="no TripoSG job"):
        r.run_job("missing", {})
