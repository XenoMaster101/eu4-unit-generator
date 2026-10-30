"""Help -> Updates: the newest release on GitHub Releases against this version, and (Release 1.0 R2) installing it in
place. Standard library only; nothing is sent but the requests for the release and its files (no account, no machine
details).

The repository is the settings file's ``update_repo`` ("owner/name"), else the one the release was built from
(``release.json`` beside this package, written by packaging/windows/build_package.py), else ``REPO``.

Installing an update: the release's ``eu4unitgen-<v>-setup.exe`` is downloaded into the app's temp folder, its SHA-256
checked against the release's ``SHA256SUMS.txt`` (a damaged or changed download is refused), then run with the install
folder given (``/DIR``) once the app has closed: it upgrades in place (the app replaced; settings, work, the library,
Blender and the AI downloads kept) and starts the app again. Files downloaded this way carry no "from the internet"
mark, so Windows doesn't show its unsigned-download warning for them (only the first download from a browser does)."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO = ""  # fallback "owner/name" (release.json normally gives it)
API = "https://api.github.com/repos/{repo}/releases/latest"
PAGE = "https://github.com/{repo}/releases"
SETUP = re.compile(r"^eu4unitgen-[\d.]+-setup\.exe$")
SUMS = "SHA256SUMS.txt"


@dataclass
class Update:
    ok: bool  # the check worked
    newer: bool  # a newer release exists
    current: str
    latest: str = ""
    url: str = ""  # the release's page
    text: str = ""  # what to tell the user
    notes: str = ""  # the release's notes (what changed), Markdown
    setup_url: str = ""  # the installer's download address ("" when the release has none: the page only)
    setup_name: str = ""
    setup_size: int = 0  # bytes
    sums_url: str = ""


def parse_version(v: str) -> tuple[int, ...]:
    """"v0.2.0" / "0.2.0" / "0.2" -> (0, 2, 0); other text after the numbers is ignored."""
    nums = re.findall(r"\d+", (v or "").split("-", 1)[0])
    out = tuple(int(n) for n in nums[:3])
    return out + (0,) * (3 - len(out))


def _agent() -> str:
    from eu4unitgen import __version__
    return f"eu4unitgen/{__version__}"


def _fetch(url: str, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": _agent()})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - a fixed https address
        return json.loads(r.read().decode("utf-8"))


def built_repo() -> str:
    """The repository this copy was released from (release.json), or ""."""
    try:
        return str(json.loads((Path(__file__).resolve().parents[1] / "release.json").read_text(encoding="utf-8"))
                   .get("repo") or "")
    except (OSError, ValueError):
        return ""


def repo() -> str:
    from eu4unitgen.core import config
    return (config.load().update_repo or built_repo() or REPO).strip()


def check(repository: str | None = None, current: str | None = None, fetch=_fetch) -> Update:
    from eu4unitgen import __version__
    current = current or __version__
    repository = repository if repository is not None else repo()
    if not repository:
        return Update(False, False, current, text=f"You have version {current}. This copy wasn't built from a "
                                                   "release, so it doesn't know where updates come from. (Set "
                                                   "update_repo in the settings file to \"owner/name\".)")
    page = PAGE.format(repo=repository)
    try:
        rel = fetch(API.format(repo=repository))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return Update(False, False, current, url=page,
                          text=f"You have version {current}. No release has been published yet.")
        return Update(False, False, current, url=page,
                      text=f"Couldn't check for updates (GitHub answered {exc.code}). Try later.")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return Update(False, False, current, text=f"Couldn't reach GitHub to check for updates ({exc}). Are you online?")
    latest = str(rel.get("tag_name") or rel.get("name") or "")
    url = rel.get("html_url") or page
    assets = {a.get("name", ""): a for a in rel.get("assets") or []}
    setup = next((a for n, a in assets.items() if SETUP.match(n)), None)
    sums = assets.get(SUMS)
    extra = dict(notes=str(rel.get("body") or ""),
                 setup_url=(setup or {}).get("browser_download_url", ""), setup_name=(setup or {}).get("name", ""),
                 setup_size=int((setup or {}).get("size") or 0), sums_url=(sums or {}).get("browser_download_url", ""))
    if parse_version(latest) > parse_version(current):
        how = ("Install it from here: it updates this copy in place (your units, settings, library and downloads "
               "stay)." if extra["setup_url"] and extra["sums_url"] else "Open its page to download it.")
        return Update(True, True, current, latest, url, f"Version {latest.lstrip('v')} is out (you have {current}). "
                      + how, **extra)
    return Update(True, False, current, latest, url, f"You have the newest version ({current}).", **extra)


def expected_sha256(sums_text: str, name: str) -> str | None:
    """The SHA-256 a ``sha256sum``-style list gives ``name`` ("<hex>  <name>" or "<hex> *<name>")."""
    for line in sums_text.splitlines():
        m = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$", line)
        if m and m.group(2) == name:
            return m.group(1).lower()
    return None


def _open(url: str, timeout: float = 30.0):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": _agent()}), timeout=timeout)  # noqa: S310


def download(u: Update, dest: Path | None = None, progress=None, opener=_open) -> Path:
    """The release's installer downloaded into ``dest`` (default: the app's temp folder) and checked against the
    release's SHA256SUMS.txt; ``progress(done_bytes, total_bytes)``. A file that doesn't match is deleted and refused
    (ValueError)."""
    if not (u.setup_url and u.sums_url):
        raise ValueError("this release has no installer or no checksums file: download it from its page")
    with opener(u.sums_url) as r:
        want = expected_sha256(r.read().decode("utf-8", "replace"), u.setup_name)
    if not want:
        raise ValueError(f"{SUMS} doesn't list {u.setup_name}: not installed")
    if dest is None:
        from eu4unitgen.core import config
        dest = config.config_dir() / "tmp" / "updates"
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / u.setup_name
    part = out.with_suffix(".part")
    h = hashlib.sha256()
    got = 0
    with opener(u.setup_url) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or u.setup_size or 0) if hasattr(r, "headers") else u.setup_size
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            got += len(chunk)
            if progress:
                progress(got, total)
    if h.hexdigest() != want:
        part.unlink(missing_ok=True)
        raise ValueError(f"the download of {u.setup_name} doesn't match its checksum (damaged or changed): not installed")
    part.replace(out)
    return out


def install_command(setup: Path, root: Path | None) -> list[str]:
    """The installer run for an in-place update: its progress window only, into this install's folder, the app
    started again afterwards."""
    cmd = [str(setup), "/SILENT", "/NOCANCEL", "/RELAUNCH=1"]
    if root:
        cmd.append(f"/DIR={root}")
    return cmd


def launch(setup: Path) -> None:
    """Start the installer detached (it waits for this app to close: the app quits right after)."""
    from eu4unitgen.core import layout
    if sys.platform != "win32":
        raise OSError("updates install themselves on Windows only: download the new version from its page")
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen(install_command(setup, layout.root()), creationflags=flags, close_fds=True)  # noqa: S603


MUTEX = "eu4unitgen_running"  # held while the app runs; the installer waits for it to go (setup.iss)


def hold_running_mutex():
    """A named mutex while the app runs (Windows), so an installer can wait for it to close. -> the handle (keep it)."""
    if sys.platform != "win32":
        return None
    import ctypes
    try:
        return ctypes.windll.kernel32.CreateMutexW(None, False, MUTEX)
    except (AttributeError, OSError):
        return None
