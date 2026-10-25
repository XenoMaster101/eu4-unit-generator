"""Project settings: where the mod and the game live, and project-wide generation options."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from eu4unitgen.core.naming import Naming

DEFAULT_ENTITY_VERSION = 20  # SPEC §3: project-wide version on every generated entity
DEFAULT_TAG_PACK_ORDER = 110  # SPEC §3: beats vanilla (0-70) and DLC (up to 100)

# Pinned Blender toolchain (SPEC §8). Blender only ever runs as a subprocess.
BLENDER_VERSION = "4.2.23"  # LTS
IO_PDX_MESH_VERSION = "0.91"
WINDOWS_BLENDER_PATH = Path(r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe")


@dataclass
class Project:
    mod_root: Path
    eu4_path: Path | None = None
    prefix: str = "hr_"
    entity_version: int = DEFAULT_ENTITY_VERSION
    tag_pack_order: int = DEFAULT_TAG_PACK_ORDER
    languages: list[str] = field(default_factory=lambda: ["english"])
    include_dlc: bool = True  # index DLC zips for collision checks
    owned_dlc: list[str] = field(default_factory=list)  # dlc folder names; the dlc/ folder doesn't prove ownership
    # Read the mod's existing files when generating and validating: its copies of shared files
    # (mil.txt, graphicalculturetype.txt) and name/version collisions with its entities.
    # Off until the mod folder exists. Export's ownership checks run regardless.
    check_mod_collisions: bool = False

    def __post_init__(self) -> None:
        self.mod_root = Path(self.mod_root)
        if self.eu4_path is None:  # the settings file (EU4_PATH wins over it: core.config)
            from eu4unitgen.core import config
            self.eu4_path = config.load().eu4()
        elif self.eu4_path is not None:
            self.eu4_path = Path(self.eu4_path)
        self.naming = Naming(self.prefix)

    def mod_files(self):
        """The mod folder as a read-only layer, or None when mod checks are off or it doesn't exist."""
        from eu4unitgen.library.scanner import GameFiles

        if not self.check_mod_collisions or not self.mod_root.is_dir():
            return None
        return GameFiles(self.mod_root, include_dlc=False)


def default_blender_path() -> Path | None:
    """The pinned Blender 4.2 executable: $BLENDER_PATH, else the standard Windows install
    location, else ``blender`` on PATH (Linux). None if nothing is found off Windows."""
    if os.environ.get("BLENDER_PATH"):
        return Path(os.environ["BLENDER_PATH"])
    if os.name == "nt":
        return WINDOWS_BLENDER_PATH
    import shutil

    found = shutil.which("blender")
    return Path(found) if found else None


def documents_dir() -> Path:
    """The user's Documents folder (on Windows often redirected, e.g. to OneDrive: asked of the shell)."""
    docs = Path.home() / "Documents"
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
            if ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf) == 0:  # CSIDL_PERSONAL
                docs = Path(buf.value)
        except (AttributeError, OSError):
            pass
    elif os.environ.get("XDG_DOCUMENTS_DIR"):
        docs = Path(os.environ["XDG_DOCUMENTS_DIR"])
    return docs


def default_mod_dir() -> Path:
    """The EU4 user mod folder: Documents/Paradox Interactive/Europa Universalis IV/mod.

    On Windows "Documents" is often redirected (e.g. to OneDrive), so ask the shell
    for it instead of assuming ~/Documents.
    """
    docs = Path.home() / "Documents"
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
            # CSIDL_PERSONAL = 5 (My Documents); SHGFP_TYPE_CURRENT = 0
            if ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf) == 0:
                docs = Path(buf.value)
        except (AttributeError, OSError):
            pass
    else:
        xdg = os.environ.get("XDG_DOCUMENTS_DIR")
        if xdg:
            docs = Path(xdg)
        # Steam/Proton EU4 on Linux uses ~/.local/share/Paradox Interactive.
        native = Path.home() / ".local" / "share" / "Paradox Interactive" / "Europa Universalis IV" / "mod"
        if native.is_dir():
            return native
    return docs / "Paradox Interactive" / "Europa Universalis IV" / "mod"
