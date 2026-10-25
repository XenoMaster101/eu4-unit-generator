"""One settings file for every path the app and its pipelines use (SPEC Track P3.3).

``settings.json`` in the config folder: ``%APPDATA%\\eu4unitgen`` on Windows, ``$XDG_CONFIG_HOME/eu4unitgen``
(``~/.config/eu4unitgen``) on Linux; ``EU4UNITGEN_CONFIG_DIR`` points elsewhere (a fresh-user test, the tests).
Environment variables still win over the file for the paths they name (``EU4_PATH``, ``BLENDER_PATH``,
``EU4UNITGEN_AI_PYTHON``, ``EU4UNITGEN_WORKSPACE``), so scripts and the test suite keep working unchanged.

Keys (all optional; an unset one falls back to its default or detection):
  eu4_path      the EU4 install (read at run time, never copied: rule 1)
  mod_dir       the EU4 user mod folder ("Add to mod" writes there; default: Documents/Paradox Interactive/...)
  workspace     your work (default: the checkout when run from one, else ``Documents\\eu4unitgen``):
                ``references`` (your images, copied in), ``projects`` (one folder per unit), ``result sheets``,
                ``exported mods``; a checkout keeps its own layout (``assets_in``, ``build``)
  ai_python     the Windows AI environment's python (cut-out and SAM 2); its AI root holds the weights
  ai_root       the folder holding the weights and the TRELLIS.2 checkout (default: from ai_python)
  wsl_distro    the WSL distribution TRELLIS.2 runs in ("" = the default one)
  wsl_python    the TRELLIS.2 environment's python inside WSL
  blender_path  Blender 4.2 LTS (only ever run as a subprocess)
  owned_dlc     DLC folder names the user owns
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

FILE = "settings.json"
DEFAULT_WSL_PYTHON = "~/eu4unitgen_ai/trellis2_env/bin/python"
ENV = {"eu4_path": "EU4_PATH", "blender_path": "BLENDER_PATH", "ai_python": "EU4UNITGEN_AI_PYTHON",
       "workspace": "EU4UNITGEN_WORKSPACE"}


def config_dir() -> Path:
    env = os.environ.get("EU4UNITGEN_CONFIG_DIR")
    if env:
        return Path(env)
    if os.name == "nt":
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "eu4unitgen"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "eu4unitgen"


def checkout_root() -> Path | None:
    """The source checkout this package runs from (a ``pyproject.toml`` two levels over ``src/``), or None when
    installed."""
    root = Path(__file__).resolve().parents[3]
    return root if (root / "pyproject.toml").is_file() and (root / "src" / "eu4unitgen").is_dir() else None


def default_workspace() -> Path:
    root = checkout_root()
    if root is not None:
        return root
    from eu4unitgen.core.project import documents_dir
    return documents_dir() / "eu4unitgen"


def logs_dir() -> Path:
    """Logs live beside the settings: ``%APPDATA%\\eu4unitgen\\logs``."""
    return config_dir() / "logs"


@dataclass
class Config:
    eu4_path: str = ""
    mod_dir: str = ""
    workspace: str = ""
    ai_python: str = ""
    ai_root: str = ""
    wsl_distro: str = ""
    wsl_python: str = DEFAULT_WSL_PYTHON
    blender_path: str = ""
    owned_dlc: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ resolved paths
    def _env(self, key: str) -> str:
        return os.environ.get(ENV.get(key, ""), "") or getattr(self, key)

    def eu4(self) -> Path | None:
        v = self._env("eu4_path")
        return Path(v) if v else None

    def mods(self) -> Path:
        from eu4unitgen.core.project import default_mod_dir
        return Path(self.mod_dir) if self.mod_dir else default_mod_dir()

    def work(self) -> Path:
        v = self._env("workspace")
        return Path(v) if v else default_workspace()

    def _layout(self) -> dict:
        """The workspace's folders: a checkout's own (assets_in, build) or the installed app's (Documents)."""
        w = self.work()
        if checkout_root() is not None and w == checkout_root():
            return {"projects": w / "assets_in", "references": w / "assets_in" / "new_refs",
                    "sheets": w / "build" / "result_sheets", "exports": w / "build" / "add_to_mod"}
        return {"projects": w / "projects", "references": w / "references", "sheets": w / "result sheets",
                "exports": w / "exported mods"}

    def projects(self) -> Path:
        return self._layout()["projects"]

    def references(self) -> Path:
        return self._layout()["references"]

    def sheets(self) -> Path:
        return self._layout()["sheets"]

    def exports(self) -> Path:
        return self._layout()["exports"]

    def ai(self) -> str:
        return self._env("ai_python")

    def ai_root_path(self) -> Path | None:
        if self.ai_root:
            return Path(self.ai_root)
        from eu4unitgen.ref3d.writeguard import ai_root
        return ai_root(self.ai()) if self.ai() else None

    def blender(self) -> Path | None:
        v = self._env("blender_path")
        if v:
            return Path(v)
        from eu4unitgen.core.project import default_blender_path
        return default_blender_path()

    def wsl_base(self) -> list[str]:
        """``wsl.exe`` with the distribution, for ``subprocess`` (``-e bash -lc <cmd>`` follows)."""
        return ["wsl.exe"] + (["-d", self.wsl_distro] if self.wsl_distro else [])


def load() -> Config:
    p = config_dir() / FILE
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    known = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in raw.items() if k in known})


def save(cfg: Config) -> Path:
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = d / FILE
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(cfg), indent=1), encoding="utf-8")
    os.replace(tmp, p)
    return p


def update(**values) -> Config:
    cfg = load()
    for k, v in values.items():
        if not hasattr(cfg, k):
            raise KeyError(k)
        setattr(cfg, k, v)
    save(cfg)
    return cfg
