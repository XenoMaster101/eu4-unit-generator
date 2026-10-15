"""Shared helpers for Blender jobs (runs inside Blender only).

io_pdx_mesh 0.91 on Blender 4.2 is an extension module (``bl_ext.<repo>.io_pdx_mesh``); its
import/export functions live in ``pdx_blender.blender_import_export``. Materials it exports
carry the custom property ``shader`` and take texture *file names* from the image nodes on
the Principled BSDF's Base Color (diff), Roughness (spec) and Normal (n) inputs. Blender is
Z-up; the addon converts to the mesh files' Y-up on import/export.
"""
import importlib
import json
import sys

import addon_utils  # noqa: E402 - Blender module
import bpy  # noqa: E402 - Blender module

PDX_SHADER = "shader"


def args() -> dict:
    return json.loads(open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8").read())


def write_result(a: dict, result: dict) -> None:
    with open(a["result_path"], "w", encoding="utf-8") as f:
        json.dump(result, f)


def io_module():
    """io_pdx_mesh's blender_import_export module (the addon must be enabled)."""
    for mod in addon_utils.modules():
        if mod.__name__.split(".")[-1] == "io_pdx_mesh":
            return importlib.import_module(mod.__name__ + ".pdx_blender.blender_import_export")
    raise RuntimeError("io_pdx_mesh not installed")


def clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials, bpy.data.images):
        for item in list(coll):
            coll.remove(item)


def pdx_material(name: str, shader: str, textures: dict) -> "bpy.types.Material":
    """A material io_pdx_mesh exports: ``textures`` maps diff/spec/n to file names (bare)."""
    mat = bpy.data.materials.new(name)
    mat[PDX_SHADER] = shader
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    for slot, socket in (("diff", "Base Color"), ("spec", "Roughness"), ("n", "Normal")):
        if not textures.get(slot):
            continue
        img = bpy.data.images.new(textures[slot], 4, 4)
        img.filepath = "//" + textures[slot]
        node = nodes.new("ShaderNodeTexImage")
        node.image = img
        if slot == "n":
            nmap = nodes.new("ShaderNodeNormalMap")
            links.new(node.outputs["Color"], nmap.inputs["Color"])
            links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
        else:
            links.new(node.outputs["Color"], bsdf.inputs[socket])
    return mat
