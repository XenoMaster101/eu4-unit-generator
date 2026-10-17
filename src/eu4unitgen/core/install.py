"""Find the EU4 install and the DLC the user owns, without asking (SPEC Track R, GUI usability U2).

Install: ``$EU4_PATH``, else every Steam library (``steamapps/libraryfolders.vdf``) of the Steam
install (Windows registry ``HKCU\\Software\\Valve\\Steam`` SteamPath, the default folders on
Windows and Linux) holding app 236850.

Owned DLC: Steam's ``appmanifest_236850.acf`` lists the DLC depots it installed for this account
(``InstalledDepots`` entries with a ``dlcappid``); each ``dlc/<folder>/*.dlc`` names its
``steam_id``. A folder counts as owned when its steam_id is installed. UNVERIFIED that Steam only
installs owned DLC depots: on the user's install all 122 folders matched, so the GUI uses this
list only to fill an empty DLC list, never to replace the user's ticks. Other stores: unknown (empty).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

EU4_APP = "236850"
_KV = re.compile(r'"((?:[^"\\]|\\.)*)"|([{}])')


class VDFError(ValueError):
    pass


def parse_vdf(text: str) -> dict:
    """Valve's KeyValues text (libraryfolders.vdf, appmanifest .acf): nested dicts of strings."""
    tokens = [m.group(1) if m.group(1) is not None else m.group(2) for m in _KV.finditer(text)]
    root: dict = {}
    stack = [root]
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "}":
            if len(stack) == 1:
                raise VDFError("unbalanced }")
            stack.pop()
            i += 1
            continue
        if i + 1 >= len(tokens):
            raise VDFError(f"key {t!r} has no value")
        nxt = tokens[i + 1]
        key = t.replace("\\\\", "\\")
        if nxt == "{":
            child: dict = {}
            stack[-1][key] = child
            stack.append(child)
        else:
            stack[-1][key] = nxt.replace("\\\\", "\\")
        i += 2
    return root


def steam_roots() -> list[Path]:
    """Where Steam may be installed (existing folders only)."""
    out: list[Path] = []
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:
                out.append(Path(winreg.QueryValueEx(key, "SteamPath")[0]))
        except OSError:
            pass
        for env in ("ProgramFiles(x86)", "ProgramFiles"):
            if os.environ.get(env):
                out.append(Path(os.environ[env]) / "Steam")
    else:
        home = Path.home()
        out += [home / ".steam" / "steam", home / ".local" / "share" / "Steam",
                home / ".var" / "app" / "com.valvesoftware.Steam" / ".local" / "share" / "Steam"]
    seen, found = set(), []
    for p in out:
        key = str(p).lower() if os.name == "nt" else str(p)
        if key not in seen and p.is_dir():
            seen.add(key)
            found.append(p)
    return found


def steam_libraries(roots: list[Path] | None = None) -> list[Path]:
    """Every Steam library folder (holding ``steamapps``) the roots know of."""
    libs: list[Path] = []
    for root in roots if roots is not None else steam_roots():
        libs.append(root)
        vdf = root / "steamapps" / "libraryfolders.vdf"
        try:
            doc = parse_vdf(vdf.read_text(encoding="utf-8", errors="replace"))
        except (OSError, VDFError):
            continue
        for entry in (doc.get("libraryfolders") or {}).values():
            if isinstance(entry, dict) and entry.get("path"):
                libs.append(Path(entry["path"]))
    out, seen = [], set()
    for lib in libs:
        key = str(lib).lower() if os.name == "nt" else str(lib)
        if key not in seen and (lib / "steamapps").is_dir():
            seen.add(key)
            out.append(lib)
    return out


def find_eu4(roots: list[Path] | None = None) -> Path | None:
    """The EU4 install folder, or None."""
    if os.environ.get("EU4_PATH") and Path(os.environ["EU4_PATH"]).is_dir():
        return Path(os.environ["EU4_PATH"])
    for lib in steam_libraries(roots):
        manifest = lib / "steamapps" / f"appmanifest_{EU4_APP}.acf"
        folder = "Europa Universalis IV"
        try:
            folder = parse_vdf(manifest.read_text(encoding="utf-8", errors="replace"))["AppState"].get("installdir",
                                                                                                    folder)
        except (OSError, VDFError, KeyError):
            pass
        game = lib / "steamapps" / "common" / folder
        if (game / "dlc").is_dir() or (game / "gfx").is_dir():
            return game
    return None


def _manifest_for(eu4: Path) -> Path | None:
    steamapps = eu4.parent.parent  # <library>/steamapps/common/<game>
    m = steamapps / f"appmanifest_{EU4_APP}.acf"
    return m if m.is_file() else None


def owned_dlc(eu4: Path) -> tuple[list[str], str]:
    """(dlc folder names the Steam account owns, a note on where that came from)."""
    manifest = _manifest_for(Path(eu4))
    if manifest is None:
        return [], "no Steam manifest next to the install: tick your DLC by hand"
    try:
        doc = parse_vdf(manifest.read_text(encoding="utf-8", errors="replace"))
    except (OSError, VDFError) as exc:
        return [], f"can't read {manifest.name} ({exc}): tick your DLC by hand"
    depots = (doc.get("AppState") or {}).get("InstalledDepots") or {}
    installed = {d["dlcappid"] for d in depots.values() if isinstance(d, dict) and d.get("dlcappid")}
    owned = []
    for dlc in sorted((Path(eu4) / "dlc").glob("*/*.dlc")):
        try:
            m = re.search(r'steam_id\s*=\s*"?(\d+)"?', dlc.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if m and m.group(1) in installed:
            owned.append(dlc.parent.name)
    return owned, f"{len(owned)} owned DLC found through Steam ({manifest.name})"
