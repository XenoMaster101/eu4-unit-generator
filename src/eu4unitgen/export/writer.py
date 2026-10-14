"""Export a generated set into the mod folder: dry-run diff first, then atomic writes + manifest.

Non-destructive (CLAUDE.md rule 5):
  * a file is written only if it doesn't exist yet, or the manifest says the tool wrote it
    and it hasn't been edited since (its hash still matches);
  * any other existing file is BLOCKED and left alone; for shared files (mil.txt,
    graphicalculturetype.txt) INTEGRATION.md says what to add by hand;
  * files from an earlier run that this run no longer produces are reported as STALE and
    never deleted;
  * validator errors block the whole export.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from eu4unitgen import __version__
from eu4unitgen.core.generator import PATCHED, GeneratedFile, GeneratedSet
from eu4unitgen.core.naming import MANIFEST_FILE
from eu4unitgen.library.scanner import GameFiles

CREATE, UPDATE, UNCHANGED, BLOCKED = "create", "update", "unchanged", "blocked"
MANIFEST_SCHEMA = 1


class ExportError(RuntimeError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unedited(path: str, current: bytes, recorded: str | None) -> bool:
    """``current`` is still what the tool wrote (hash ``recorded``). One allowance: the EU4
    launcher rewrites launcher descriptors (``<mod dir>/<name>.mod``) and drops the final
    newline (seen 2026-10-14); that alone doesn't count as an edit."""
    if sha256(current) == recorded:
        return True
    if "/" not in path and path.endswith(".mod"):
        return sha256(current.rstrip(b"\r\n") + b"\n") == recorded
    return False


def _decode(data: bytes) -> str:
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


@dataclass
class PlannedFile:
    path: str
    action: str
    data: bytes
    role: str
    reason: str = ""
    diff: str = ""


@dataclass
class ExportPlan:
    mod_root: Path  # export root: every planned path is relative to it
    set_name: str
    manifest_rel: str = MANIFEST_FILE
    files: list[PlannedFile] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)  # in the manifest for this set, not generated now
    validation_errors: list[str] = field(default_factory=list)
    manifest: dict = field(default_factory=dict)  # what the manifest will be after apply()

    def by(self, action: str) -> list[PlannedFile]:
        return [f for f in self.files if f.action == action]

    @property
    def can_apply(self) -> bool:
        return not self.validation_errors and not self.by(BLOCKED)

    def summary(self) -> str:
        counts = ", ".join(f"{len(self.by(a))} {a}" for a in (CREATE, UPDATE, UNCHANGED, BLOCKED))
        return f"{counts}, {len(self.stale)} stale" + ("" if self.can_apply else "  -> export blocked")

    def diff_text(self) -> str:
        parts = [f"# eu4unitgen plan: {self.set_name} -> {self.mod_root}", f"# {self.summary()}"]
        parts += [f"# VALIDATION ERROR: {e}" for e in self.validation_errors]
        for f in self.files:
            note = f" ({f.reason})" if f.reason else ""
            parts.append(f"# {f.action.upper():9} {f.path}{note}")
        parts += [f"# STALE     {p} (generated earlier, not now; left in place)" for p in self.stale]
        parts.append("")
        parts += [f.diff for f in self.files if f.diff]
        return "\n".join(parts)


def read_manifest(mod_root: Path, manifest_rel: str = MANIFEST_FILE) -> dict:
    path = Path(mod_root) / manifest_rel
    if not path.is_file():
        return {"schema": MANIFEST_SCHEMA, "tool": "eu4unitgen", "files": {}}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ExportError(f"manifest {path} is unreadable ({exc}); refusing to guess what the tool owns") from exc
    if doc.get("schema") != MANIFEST_SCHEMA or not isinstance(doc.get("files"), dict):
        raise ExportError(f"manifest {path} has an unknown format")
    return doc


def _diff(path: str, old: bytes | None, new: bytes, old_label: str) -> str:
    a = _decode(old).splitlines(keepends=True) if old is not None else []
    b = _decode(new).splitlines(keepends=True)
    lines = difflib.unified_diff(a, b, fromfile=old_label, tofile=f"b/{path}", n=3)
    return "".join(ln if ln.endswith("\n") else ln + "\n\\ No newline at end of file\n" for ln in lines)


def plan_export(mod_root: Path, generated: GeneratedSet, game: GameFiles | None = None,
                validation_errors: list[str] | None = None, manifest_rel: str = MANIFEST_FILE) -> ExportPlan:
    """Work out what an export would do. Reads the target folder; writes nothing.
    Ownership is recorded per unit set, so several sets can share files like mil.txt."""
    mod_root = Path(mod_root)
    owner = generated.nation.set_name
    owned = read_manifest(mod_root, manifest_rel)["files"]
    plan = ExportPlan(mod_root, owner, manifest_rel, validation_errors=list(validation_errors or []))
    new_entries = {}

    for gf in sorted(generated.files.values(), key=lambda f: f.path):
        target = mod_root / gf.path
        current = target.read_bytes() if target.is_file() else None
        entry = owned.get(gf.path)
        pf = PlannedFile(gf.path, CREATE, gf.data, gf.role)
        if current is None:
            # New file. For a patched copy of a vanilla file, diff against vanilla so the
            # dry run shows only the lines the tool adds.
            src = gf.game_path or gf.path
            base = game.read(src) if (game is not None and gf.role == PATCHED and gf.based_on == "vanilla") else None
            if base is not None:
                pf.reason = "new in mod: vanilla copy + additions"
                pf.diff = _diff(gf.path, base, gf.data, f"vanilla/{src}")
            else:
                pf.diff = _diff(gf.path, None, gf.data, "/dev/null")
        elif entry is None:
            pf.action, pf.reason = BLOCKED, "exists and was not generated by eu4unitgen"
            if gf.role == PATCHED:
                pf.reason += "; merge by hand using INTEGRATION.md"
            pf.diff = _diff(gf.path, current, gf.data, f"a/{gf.path}")
        elif not unedited(gf.path, current, entry.get("sha256")):
            pf.action, pf.reason = BLOCKED, "edited since eu4unitgen wrote it"
            pf.diff = _diff(gf.path, current, gf.data, f"a/{gf.path}")
        elif current == gf.data:
            pf.action = UNCHANGED
        else:
            pf.action = UPDATE
            pf.diff = _diff(gf.path, current, gf.data, f"a/{gf.path}")
        plan.files.append(pf)
        if pf.action != BLOCKED:
            owners = set(entry.get("sets", [])) if entry else set()
            new_entries[gf.path] = {"sha256": sha256(gf.data), "role": gf.role, "sets": sorted(owners | {owner})}

    plan.stale = sorted(p for p, e in owned.items() if owner in e.get("sets", []) and p not in generated.files)
    files = {p: e for p, e in owned.items() if p not in new_entries}
    files.update(new_entries)
    plan.manifest = {"schema": MANIFEST_SCHEMA, "tool": "eu4unitgen", "tool_version": __version__,
                     "updated": None, "files": dict(sorted(files.items()))}
    return plan


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def refuse_mod_folder(root: Path, mod_dirs: list[Path]) -> None:
    """The drop-in package never goes into a mod. Refuse ``root`` if it is inside an EU4 mod
    directory, or it (or a folder above it) looks like a mod: has a descriptor.mod, or a
    sibling <name>.mod descriptor."""
    root = Path(root).resolve()
    for mod_dir in mod_dirs:
        try:
            root.relative_to(Path(mod_dir).resolve())
        except ValueError:
            continue
        raise ExportError(f"{root} is inside the EU4 mod directory {mod_dir}; packages are written to an "
                          "output folder and copied into a mod by hand")
    for folder in (root, *root.parents):
        sibling = folder.with_name(folder.name + ".mod") if folder.name else None  # drive roots have no name
        if (folder / "descriptor.mod").is_file() or (sibling is not None and sibling.is_file()):
            raise ExportError(f"{root} is inside a mod ({folder}); pick an output folder outside any mod")


def plan_package(out_root: Path, generated: GeneratedSet, game: GameFiles | None = None,
                 validation_errors: list[str] | None = None, mod_dirs: list[Path] | None = None) -> ExportPlan:
    """Plan the drop-in package at ``out_root/<set>/``: new files + INTEGRATION.md, no patched
    copies of game files. The manifest sits at ``out_root/.eu4unitgen/<set>.manifest.json``."""
    from eu4unitgen.core.naming import Naming

    refuse_mod_folder(out_root, mod_dirs or [])
    set_name = generated.nation.set_name
    folder = Naming.package_dir(set_name)
    wrapped = GeneratedSet(generated.nation, notes=list(generated.notes), integration=generated.integration)
    for f in generated.package_files():
        wrapped.add(GeneratedFile(f"{folder}/{f.path}", f.data, f.role, f.based_on, game_path=f.game_path or f.path))
    return plan_export(out_root, wrapped, game, validation_errors, manifest_rel=Naming.package_manifest(set_name))


def check_writable(root: Path) -> None:
    """Fail before the first write, not halfway through, when the target can't be written.

    On Windows, Defender's Controlled Folder Access blocks unknown programs (python.exe) from
    creating files under Documents; the OS reports that as a missing file or access denied."""
    try:
        root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=root, prefix=".eu4unitgen_probe.", suffix=".tmp")
        os.close(fd)
        os.unlink(tmp)
    except OSError as exc:
        hint = ""
        if os.name == "nt":
            hint = ("\nOn Windows this is usually Controlled Folder Access (Windows Security > Ransomware "
                    "protection). Allow python.exe there, or use --stage-dir and copy the result in Explorer.")
        raise ExportError(f"can't write to {root}: {exc.strerror or exc}{hint}") from exc


def stage_plan(plan: ExportPlan, stage_root: Path, require_empty: bool = True) -> ExportPlan:
    """The same export written under ``stage_root`` instead, for copying into place by hand.
    The stage folder must be empty: staging never overwrites anything. ``require_empty=False``
    stages a further plan into a folder this run already staged into; its files must not
    exist yet (apply_export refuses any that do)."""
    stage_root = Path(stage_root)
    if require_empty and stage_root.exists() and any(stage_root.iterdir()):
        raise ExportError(f"stage folder {stage_root} isn't empty; empty it or pick another")
    if not plan.can_apply:
        raise ExportError(f"export blocked: {plan.summary()}")
    files = [PlannedFile(f.path, CREATE, f.data, f.role, f.reason, f.diff) for f in plan.files]
    return ExportPlan(stage_root, plan.set_name, plan.manifest_rel, files, list(plan.stale),
                      list(plan.validation_errors), dict(plan.manifest))


def apply_export(plan: ExportPlan) -> list[str]:
    """Write the planned files and the manifest. Refuses if anything is blocked or invalid.
    Re-checks each target right before writing, in case the folder changed since planning."""
    if not plan.can_apply:
        raise ExportError(f"export blocked: {plan.summary()}")
    check_writable(plan.mod_root)
    fresh = read_manifest(plan.mod_root, plan.manifest_rel)["files"]
    written = []
    for pf in plan.files:
        if pf.action not in (CREATE, UPDATE):
            continue
        target = plan.mod_root / pf.path
        if pf.action == CREATE and target.exists():
            raise ExportError(f"{pf.path} appeared after the dry run; plan again")
        if pf.action == UPDATE and not unedited(pf.path, target.read_bytes(), fresh.get(pf.path, {}).get("sha256")):
            raise ExportError(f"{pf.path} changed after the dry run; plan again")
        _atomic_write(target, pf.data)
        written.append(pf.path)
    manifest = dict(plan.manifest, updated=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    _atomic_write(plan.mod_root / plan.manifest_rel, (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    return written
