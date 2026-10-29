"""Errors in plain words (Track P2.8): what happened and what to do; the technical text only behind "Details".

``explain(text, stage)`` -> {"what", "todo", "details"}: the first rule whose pattern matches the error text wins.
"""
from __future__ import annotations

import re

STAGE_WORDS = {"import": "preparing your 3D model", "cutout": "cutting the subject out of the image", "trellis": "making the 3D model",
               "build": "building the game model", "finish": "applying your edits", "preview": "staging the result",
               "add": "adding the unit to your mod", "setup": "setting up"}

RULES = [
    (r"out of (GPU|CUDA) memory|OutOfMemoryError|CUDA out of memory",
     "The graphics card ran out of memory.",
     "Close EU4, games and browsers (they use the graphics card), then press Resume."),
    (r"MemoryError|Cannot allocate memory|killed|Killed",
     "The computer ran out of RAM.",
     "Close programs you don't need (the pre-run check lists the largest), then press Resume."),
    (r"TRELLIS\.2 failed",
     "The 3D model step (TRELLIS.2, inside WSL) stopped with an error.",
     "Open Setup and press \"Check again\": it tests WSL and the TRELLIS.2 environment. Then press Resume."),
    (r"wsl(\.exe)?.*(not found|cannot find)|WSL isn't installed|The Windows Subsystem for Linux",
     "WSL isn't available.",
     "Open Setup and press \"Install WSL\" (it needs administrator rights and a restart)."),
    (r"set the EU4 install|EU4_PATH",
     "The app doesn't know where EU4 is installed.",
     "Open Settings and choose the folder that holds eu4.exe."),
    (r"Blender|blender\.exe",
     "Blender couldn't be started.",
     "Install Blender 4.2 LTS (Setup shows the link) or set its path in Settings, then press Resume."),
    (r"not inside an AI root|AI environment|ai_python",
     "The AI environment wasn't found.",
     "Open Setup and set up the AI environment, or point Settings to its python.exe."),
    (r"WriteGuardError|would write outside the allowed folders",
     "A step tried to write outside the app's folders, so it was stopped.",
     "Check Settings: the workspace folder must exist and be writable."),
    (r"Controlled Folder Access|PermissionError|Access is denied|Operation not permitted",
     "Windows didn't allow the app to write there.",
     "The app staged the files instead; follow the copy steps it shows (or allow python.exe in Windows Security "
     "> Ransomware protection)."),
    (r"No space left|disk full|ENOSPC",
     "The disk is full.",
     "Free some space, then press Resume."),
    (r"FileNotFoundError|No such file",
     "A file the run needs is missing (moved, deleted or on a drive that dropped out).",
     "Reconnect the drive or put the file back, then press Resume."),
    (r"exists already|pick a new name",
     "That name is already used by another run.",
     "Pick another name."),
    (r"staging the preview failed|validation|ERROR\s+\w+:",
     "The result didn't pass the checks the game needs.",
     "Open Details to see which check; press Resume after any edit. If it repeats, report it with the details."),
]


def explain(text: str, stage: str | None = None) -> dict:
    """The plain-words version of an error (module doc)."""
    t = text or ""
    for pat, what, todo in RULES:
        if re.search(pat, t, re.I):
            return {"what": what, "todo": todo, "details": t}
    where = STAGE_WORDS.get(stage or "", "this step")
    return {"what": f"Something went wrong while {where}.",
            "todo": "Press Resume to try again. If it happens again, open Details and send the text with your report.",
            "details": t}
