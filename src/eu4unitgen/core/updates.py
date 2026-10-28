"""Help -> Check for updates: the newest release on GitHub Releases against this version. Standard library only;
nothing is sent but the request for the release list (no account, no machine details).

The repository is the settings file's ``update_repo`` ("owner/name"), else ``REPO``. While neither is set the check
says so instead of guessing an address."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

REPO = ""  # "owner/name" once the project has a public GitHub repository with Releases
API = "https://api.github.com/repos/{repo}/releases/latest"
PAGE = "https://github.com/{repo}/releases"


@dataclass
class Update:
    ok: bool  # the check worked
    newer: bool  # a newer release exists
    current: str
    latest: str = ""
    url: str = ""  # the release's page (its download)
    text: str = ""  # what to tell the user


def parse_version(v: str) -> tuple[int, ...]:
    """"v0.2.0" / "0.2.0" / "0.2" -> (0, 2, 0); other text after the numbers is ignored."""
    nums = re.findall(r"\d+", (v or "").split("-", 1)[0])
    out = tuple(int(n) for n in nums[:3])
    return out + (0,) * (3 - len(out))


def _fetch(url: str, timeout: float = 10.0) -> dict:
    from eu4unitgen import __version__
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": f"eu4unitgen/{__version__}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - a fixed https address
        return json.loads(r.read().decode("utf-8"))


def repo() -> str:
    from eu4unitgen.core import config
    return (config.load().update_repo or REPO).strip()


def check(repository: str | None = None, current: str | None = None, fetch=_fetch) -> Update:
    from eu4unitgen import __version__
    current = current or __version__
    repository = repository if repository is not None else repo()
    if not repository:
        return Update(False, False, current, text=f"You have version {current}. Update checks aren't set up yet: the "
                                                   "app has no public download page. (Set update_repo in the settings "
                                                   "file once there is one.)")
    try:
        rel = fetch(API.format(repo=repository))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return Update(False, False, current, url=PAGE.format(repo=repository),
                          text=f"You have version {current}. No release has been published yet.")
        return Update(False, False, current, text=f"Couldn't check for updates (GitHub answered {exc.code}). Try later.")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return Update(False, False, current, text=f"Couldn't reach GitHub to check for updates ({exc}). Are you online?")
    latest = str(rel.get("tag_name") or rel.get("name") or "")
    url = rel.get("html_url") or PAGE.format(repo=repository)
    if parse_version(latest) > parse_version(current):
        return Update(True, True, current, latest, url,
                      f"Version {latest.lstrip('v')} is out (you have {current}). Open its page to download it; your "
                      "units and settings stay where they are.")
    return Update(True, False, current, latest, url, f"You have the newest version ({current}).")
