"""Blender job (runs inside Blender only): is io_pdx_mesh installed and enabled, and which version?

Usage: blender --background --python check_addon.py -- args.json
Writes {"found", "enabled", "module", "version", "blender"} to args["result_path"].
On Blender 4.2 the addon is an extension module named like ``bl_ext.user_default.io_pdx_mesh``.
"""
import json
import sys

import addon_utils  # noqa: E402 - Blender module
import bpy  # noqa: E402 - Blender module


def main() -> None:
    args = json.loads(open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8").read())
    result = {"found": False, "enabled": False, "module": None, "version": None,
              "blender": ".".join(str(v) for v in bpy.app.version)}
    for mod in addon_utils.modules():
        if mod.__name__.split(".")[-1] == "io_pdx_mesh":
            _default, loaded = addon_utils.check(mod.__name__)
            result.update(found=True, enabled=bool(loaded), module=mod.__name__,
                          version=list(mod.bl_info.get("version", ())))
            break
    with open(args["result_path"], "w", encoding="utf-8") as f:
        json.dump(result, f)


main()
