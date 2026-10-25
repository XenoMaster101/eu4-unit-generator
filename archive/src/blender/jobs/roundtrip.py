"""Blender job: Phase 3 step 1 export round-trip (runs inside Blender only).

Imports vanilla meshes in place (read from the install, nothing copied), reports what Blender
sees, then builds a trivial procedural box with a PDX material and exports it with io_pdx_mesh.
args: {"imports": [abs .mesh paths], "out_mesh": abs path, "result_path": ...}
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bmesh  # noqa: E402 - Blender module
import bpy  # noqa: E402 - Blender module

import _pdx  # noqa: E402


def main() -> None:
    a = _pdx.args()
    ie = _pdx.io_module()
    imports = []
    for path in a.get("imports", []):
        _pdx.clear_scene()
        ie.import_meshfile(path)
        meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        rigs = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
        info = {"path": path, "meshes": [o.name for o in meshes],
                "verts": sum(len(o.data.vertices) for o in meshes),
                "materials": sorted({m.get(_pdx.PDX_SHADER, "") for o in meshes for m in o.data.materials if m}),
                "bones": [b.name for r in rigs for b in r.data.bones]}
        if rigs and "head" in rigs[0].data.bones:  # Blender (Z-up) world position of the head joint
            info["head_world"] = [round(v, 4) for v in (rigs[0].matrix_world @ rigs[0].data.bones["head"].head_local)]
        imports.append(info)

    _pdx.clear_scene()
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bm.verts.ensure_lookup_table()
    mesh = bpy.data.meshes.new("hr_roundtrip_boxShape")
    bm.to_mesh(mesh)
    bm.free()
    mesh.uv_layers.new(name="map1")
    obj = bpy.data.objects.new("hr_roundtrip_box", mesh)
    bpy.context.scene.collection.objects.link(obj)
    mesh.materials.append(_pdx.pdx_material("hr_roundtrip_mat", "PdxMeshStandard",
                                            {"diff": "hr_roundtrip_diffuse.dds", "n": "hr_roundtrip_normal.dds",
                                             "spec": "hr_roundtrip_spec.dds"}))
    ie.export_meshfile(a["out_mesh"], exp_mesh=True, exp_skel=False, exp_locs=False)
    _pdx.write_result(a, {"imports": imports, "exported": a["out_mesh"],
                          "exported_exists": os.path.isfile(a["out_mesh"])})


main()
