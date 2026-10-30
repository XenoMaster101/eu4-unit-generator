"""Publish a release on GitHub (Release 1.0 R2): the repository made if it doesn't exist (public), this branch pushed,
then a release ``v<version>`` with dist/<version>/'s files (setup.exe, the zip, SHA256SUMS.txt) and its notes.

Needs the GitHub command line, signed in (``gh auth login``). Build first: python packaging/windows/build_package.py

usage: python packaging/release.py [--dry-run]
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def gh() -> str:
    for c in (shutil.which("gh"), r"C:\Program Files\GitHub CLI\gh.exe"):
        if c and Path(c).is_file():
            return c
    raise SystemExit("the GitHub command line (gh) isn't installed")


def run(*args, check=True, capture=False):
    print("+", " ".join(str(a) for a in args))
    return subprocess.run([str(a) for a in args], check=check, cwd=ROOT, text=True,
                          capture_output=capture)


def main(dry: bool = False) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from eu4unitgen import __version__ as v
    repo = json.loads((ROOT / "src" / "eu4unitgen" / "release.json").read_text(encoding="utf-8"))["repo"]
    folder = ROOT / "dist" / v
    files = [folder / n for n in sorted(p.name for p in folder.iterdir()) if n != "RELEASE_NOTES.md"] \
        if folder.is_dir() else []
    if not any(f.name.endswith("-setup.exe") for f in files):
        raise SystemExit(f"{folder} has no setup.exe: build it first (packaging/windows/build_package.py)")
    g = gh()
    if run(g, "auth", "status", check=False, capture=True).returncode:
        raise SystemExit("gh isn't signed in: run gh auth login")
    if dry:
        print(f"would publish v{v} to {repo}:", [f.name for f in files])
        return
    if run(g, "repo", "view", repo, check=False, capture=True).returncode:
        run(g, "repo", "create", repo, "--public", "--description",
            "Makes Europa Universalis IV land units (infantry, cavalry, artillery) from pictures and 3D models")
    remotes = run("git", "remote", capture=True).stdout.split()
    if "origin" not in remotes:
        run("git", "remote", "add", "origin", f"https://github.com/{repo}.git")
    branch = run("git", "branch", "--show-current", capture=True).stdout.strip()
    run("git", "push", "-u", "origin", f"{branch}:main")
    tag = f"v{v}"
    if run(g, "release", "view", tag, "--repo", repo, check=False, capture=True).returncode == 0:
        run(g, "release", "upload", tag, *files, "--repo", repo, "--clobber")
    else:
        run(g, "release", "create", tag, *files, "--repo", repo, "--target", "main", "--title",
            f"EU4 Unit Generator {v}", "--notes-file", folder / "RELEASE_NOTES.md")
    print(f"published: https://github.com/{repo}/releases/tag/{tag}")


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
