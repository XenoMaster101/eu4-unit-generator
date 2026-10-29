"""Run Blender jobs as a subprocess (CLAUDE.md: never import bpy in the app process).

    blender --background --python-exit-code 1 --python jobs/<job>.py -- <args.json>

A job reads its arguments from the JSON file and writes its result as JSON to
``args["result_path"]``. The runner checks the pinned Blender version first, applies a
timeout, and keeps Blender's console output as the job's log. No ``--factory-startup``:
io_pdx_mesh is a user extension and must stay loaded.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from eu4unitgen.core.project import BLENDER_VERSION, IO_PDX_MESH_VERSION

JOBS_DIR = Path(__file__).with_name("jobs")
VERSION_RE = re.compile(r"^Blender (\d+\.\d+\.\d+)", re.MULTILINE)


class BlenderError(RuntimeError):
    pass


@dataclass
class JobResult:
    job: str
    returncode: int
    result: dict
    log: str
    log_path: Path | None = None


def parse_version(output: str) -> str | None:
    m = VERSION_RE.search(output)
    return m.group(1) if m else None


def addon_version(value) -> str:
    """bl_info / manifest version -> "0.91": (0, 91, 0) or "0.91.0" both drop trailing zeros."""
    parts = [str(int(p)) for p in (value if isinstance(value, (list, tuple)) else str(value).split("."))]
    while len(parts) > 2 and parts[-1] == "0":
        parts.pop()
    return ".".join(parts)


class BlenderRunner:
    def __init__(self, executable: Path | str | None, log_dir: Path | None = None, timeout: float = 300):
        if not executable:
            raise BlenderError("no Blender executable set (Settings, or $BLENDER_PATH)")
        self.executable = Path(executable)
        self.log_dir = Path(log_dir) if log_dir else None
        self.timeout = timeout
        self._checked = False

    def _run(self, args: list[str], timeout: float) -> subprocess.CompletedProcess:
        if not self.executable.is_file():
            raise BlenderError(f"Blender not found at {self.executable}")
        try:
            return subprocess.run([str(self.executable), *args], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise BlenderError(f"Blender timed out after {timeout:.0f} s ({' '.join(args[:4])} ...)") from exc
        except OSError as exc:
            raise BlenderError(f"can't start Blender at {self.executable}: {exc}") from exc

    def version(self) -> str:
        out = self._run(["--version"], 60)
        version = parse_version(out.stdout + out.stderr)
        if version is None:
            raise BlenderError(f"{self.executable} --version printed no Blender version")
        return version

    def check_version(self) -> str:
        version = self.version()
        if version != BLENDER_VERSION:
            raise BlenderError(f"Blender {version} at {self.executable}; this app is pinned to {BLENDER_VERSION} LTS")
        self._checked = True
        return version

    def run_job(self, job: str, args: dict | None = None, timeout: float | None = None) -> JobResult:
        """Run jobs/<job>.py in background Blender. Raises BlenderError if Blender fails, the
        job exits non-zero or writes no result."""
        script = JOBS_DIR / f"{job}.py"
        if not script.is_file():
            raise BlenderError(f"no Blender job {job!r} ({script})")
        from eu4unitgen.ref3d.writeguard import absolute, app_roots, check_writes
        args = absolute(args or {})  # Blender reads a relative path from the drive's root
        check_writes(args, app_roots(), f"Blender job {job}")  # nothing outside the app's folders
        if not self._checked:
            self.check_version()
        with tempfile.TemporaryDirectory(prefix="eu4unitgen_blender_") as tmp:
            tmp_path = Path(tmp)
            result_path = tmp_path / "result.json"
            args_path = tmp_path / "args.json"
            args_path.write_text(json.dumps({**(args or {}), "result_path": str(result_path)}), encoding="utf-8")
            out = self._run(["--background", "--python-exit-code", "1", "--python", str(script), "--", str(args_path)],
                            timeout or self.timeout)
            log = out.stdout + (("\n--- stderr ---\n" + out.stderr) if out.stderr.strip() else "")
            log_path = None
            if self.log_dir is not None:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                log_path = self.log_dir / f"{job}.log"
                log_path.write_text(log, encoding="utf-8")
            result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
        if out.returncode != 0 or result is None:
            tail = "\n".join(log.strip().splitlines()[-15:])
            raise BlenderError(f"Blender job {job} failed (exit {out.returncode}"
                               f"{', no result' if result is None else ''}). Last lines:\n{tail}")
        return JobResult(job, out.returncode, result, log, log_path)

    def check_addon(self) -> dict:
        """The pinned toolchain: Blender version and io_pdx_mesh enabled at the pinned version."""
        res = self.run_job("check_addon", timeout=120).result
        if not res.get("found"):
            raise BlenderError("io_pdx_mesh isn't installed in this Blender")
        if not res.get("enabled"):
            raise BlenderError(f"io_pdx_mesh ({res.get('module')}) is installed but not enabled")
        version = addon_version(res.get("version") or "0")
        if version != IO_PDX_MESH_VERSION:
            raise BlenderError(f"io_pdx_mesh {version} installed; this app is pinned to {IO_PDX_MESH_VERSION}")
        return {**res, "blender": BLENDER_VERSION, "addon_version": version}
