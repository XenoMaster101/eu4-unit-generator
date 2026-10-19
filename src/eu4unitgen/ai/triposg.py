"""Run TripoSG (image -> 3D shape, MIT) as a subprocess in its own Python 3.10 environment.

The app never imports torch or TripoSG. ``TripoSGRunner`` starts the environment's python on
``jobs/<job>.py`` with an arguments JSON, like blender.runner does for Blender:

    <env>/Scripts/python.exe -I jobs/triposg_generate.py <args.json>

The job writes its result JSON to ``args["result_path"]``. Paths default to the environment
variables EU4UNITGEN_TRIPOSG_PYTHON (the environment's python), EU4UNITGEN_TRIPOSG_REPO (the
TripoSG source checkout) and EU4UNITGEN_TRIPOSG_WEIGHTS (the VAST-AI/TripoSG weights folder).
The user installs these; nothing is bundled.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

JOBS_DIR = Path(__file__).with_name("jobs")


class TripoSGError(RuntimeError):
    pass


@dataclass
class TripoSGResult:
    job: str
    result: dict
    log: str


class TripoSGRunner:
    def __init__(self, python: Path | str | None = None, repo: Path | str | None = None,
                 weights: Path | str | None = None, timeout: float = 1800):
        python = python or os.environ.get("EU4UNITGEN_TRIPOSG_PYTHON")
        repo = repo or os.environ.get("EU4UNITGEN_TRIPOSG_REPO")
        weights = weights or os.environ.get("EU4UNITGEN_TRIPOSG_WEIGHTS")
        if not (python and repo and weights):
            raise TripoSGError("TripoSG isn't set up: set its environment's python, source folder and weights folder")
        self.python, self.repo, self.weights = Path(python), Path(repo), Path(weights)
        for what, p in (("python", self.python), ("source folder", self.repo), ("weights folder", self.weights)):
            if not p.exists():
                raise TripoSGError(f"TripoSG {what} not found at {p}")
        self.timeout = timeout

    def run_job(self, job: str, args: dict, timeout: float | None = None) -> TripoSGResult:
        script = JOBS_DIR / f"{job}.py"
        if not script.is_file():
            raise TripoSGError(f"no TripoSG job {job!r} ({script})")
        with tempfile.TemporaryDirectory(prefix="eu4unitgen_triposg_") as tmp:
            result_path = Path(tmp) / "result.json"
            args_path = Path(tmp) / "args.json"
            args_path.write_text(json.dumps({**args, "repo": str(self.repo), "weights": str(self.weights),
                                             "result_path": str(result_path)}), encoding="utf-8")
            try:
                out = subprocess.run([str(self.python), "-I", str(script), str(args_path)], capture_output=True,
                                     text=True, encoding="utf-8", errors="replace", timeout=timeout or self.timeout)
            except subprocess.TimeoutExpired as exc:
                raise TripoSGError(f"TripoSG job {job} timed out after {timeout or self.timeout:.0f} s") from exc
            log = out.stdout + (("\n--- stderr ---\n" + out.stderr) if out.stderr.strip() else "")
            result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
        if out.returncode != 0 or result is None:
            tail = "\n".join(log.strip().splitlines()[-15:])
            raise TripoSGError(f"TripoSG job {job} failed (exit {out.returncode}). Last lines:\n{tail}")
        return TripoSGResult(job, result, log)

    def generate(self, image: Path, out_glb: Path, seed: int = 42, steps: int = 50, guidance: float = 7.0,
                 octree_depth: int = 9) -> dict:
        """Image (RGBA with the background removed) -> raw mesh (.glb). Returns the job's result."""
        return self.run_job("triposg_generate", {"image": str(image), "output": str(out_glb), "seed": seed,
                                                 "steps": steps, "guidance": guidance,
                                                 "octree_depth": octree_depth}).result
