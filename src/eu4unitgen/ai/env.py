"""Run a job in the separate AI environment (Track M; torch, diffusers, rembg live there, never in
the app's .venv). Like TripoSGRunner and blender.runner:

    <env>/Scripts/python.exe -I jobs/<job>.py <args.json>

The job writes its result JSON to ``args["result_path"]``. The environment's python defaults to
EU4UNITGEN_AI_PYTHON, then EU4UNITGEN_TRIPOSG_PYTHON (one environment serves both today). Extra
environment variables (model folders such as U2NET_HOME, HF_HOME; HF_HUB_OFFLINE) are passed as
given: jobs run offline on weights the user installed.

Writes are guarded (ref3d.writeguard): every output path must be inside the project or the AI root,
and the runner's own args/result files live in ``<AI root>/tmp``, not the system temp folder.

If the AI drive drops out (it did on 2026-10-20), a job stops with EXIT_DRIVE_LOST (jobs/_resilience)
or simply fails; either way, while the environment's python can't be read the runner raises
AIDriveLost. Jobs that checkpoint resume when rerun into the same out_dir.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from eu4unitgen.ai.jobs._resilience import EXIT_DRIVE_LOST, drive_ok
from eu4unitgen.ref3d.writeguard import allowed_roots, check_writes

JOBS_DIR = Path(__file__).with_name("jobs")


class AIJobError(RuntimeError):
    pass


class AIDriveLost(AIJobError):
    """The drive holding the AI environment is gone: the job stopped; rerun it once the drive is back."""


@dataclass
class AIJobResult:
    job: str
    result: dict
    log: str


class AIJobRunner:
    def __init__(self, python: Path | str | None = None, env: dict | None = None, timeout: float = 3600):
        python = python or os.environ.get("EU4UNITGEN_AI_PYTHON") or os.environ.get("EU4UNITGEN_TRIPOSG_PYTHON")
        if not python:
            raise AIJobError("the AI environment isn't set up: give its python (EU4UNITGEN_AI_PYTHON)")
        self.python = Path(python)
        if not self.python.exists():
            raise AIJobError(f"AI environment python not found at {self.python}")
        self.env = {k: str(v) for k, v in (env or {}).items()}
        self.timeout = timeout
        self.roots = allowed_roots(self.python)

    def run_job(self, job: str, args: dict, timeout: float | None = None) -> AIJobResult:
        script = JOBS_DIR / f"{job}.py"
        if not script.is_file():
            raise AIJobError(f"no AI job {job!r} ({script})")
        check_writes(args, self.roots, f"AI job {job}")
        if not drive_ok(self.python):
            raise AIDriveLost(f"AI job {job} not started: the AI environment ({self.python}) can't be read")
        tmp_root = self.roots[-1] / "tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        result = None
        with tempfile.TemporaryDirectory(prefix="eu4unitgen_ai_", dir=tmp_root, ignore_cleanup_errors=True) as tmp:
            result_path = Path(tmp) / "result.json"
            args_path = Path(tmp) / "args.json"
            args_path.write_text(json.dumps({**args, "result_path": str(result_path)}), encoding="utf-8")
            try:
                out = subprocess.run([str(self.python), "-I", str(script), str(args_path)], capture_output=True,
                                     text=True, encoding="utf-8", errors="replace", timeout=timeout or self.timeout,
                                     env={**os.environ, "HF_HUB_OFFLINE": "1", **self.env})
            except subprocess.TimeoutExpired as exc:
                raise AIJobError(f"AI job {job} timed out after {timeout or self.timeout:.0f} s") from exc
            log = out.stdout + (("\n--- stderr ---\n" + out.stderr) if out.stderr.strip() else "")
            try:
                result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
            except OSError:
                pass
        if out.returncode == EXIT_DRIVE_LOST or (out.returncode != 0 or result is None) and not drive_ok(self.python):
            tail = "\n".join(log.strip().splitlines()[-5:])
            raise AIDriveLost(f"AI job {job} stopped: the AI drive dropped out (exit {out.returncode}). Last lines:\n{tail}")
        if out.returncode != 0 or result is None:
            tail = "\n".join(log.strip().splitlines()[-15:])
            raise AIJobError(f"AI job {job} failed (exit {out.returncode}). Last lines:\n{tail}")
        return AIJobResult(job, result, log)
