"""Project settings: where the mod and the game live, and project-wide generation options."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from eu4unitgen.core.naming import Naming

DEFAULT_ENTITY_VERSION = 20  # SPEC §3: project-wide version on every generated entity
DEFAULT_TAG_PACK_ORDER = 110  # SPEC §3: beats vanilla (0-70) and DLC (up to 100)


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
        if self.eu4_path is None and os.environ.get("EU4_PATH"):
            self.eu4_path = Path(os.environ["EU4_PATH"])
        elif self.eu4_path is not None:
            self.eu4_path = Path(self.eu4_path)
        self.naming = Naming(self.prefix)

    def mod_files(self):
        """The mod folder as a read-only layer, or None when mod checks are off or it doesn't exist."""
        from eu4unitgen.library.scanner import GameFiles

        if not self.check_mod_collisions or not self.mod_root.is_dir():
            return None
        return GameFiles(self.mod_root, include_dlc=False)


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
