"""Blender job: render a plain model file (.glb/.gltf/.obj/.fbx, or a .mesh through io_pdx_mesh)
from named views, for checking imported and generated geometry (Track C, test 15).

args: {"model": path, "out_dir": path, "name": str, "size": 512, "texture": diffuse image | null,
       "views": {"front": [azimuth_deg, elevation_deg], ...}, "result_path": path}
Azimuth 0 looks along +Y (Blender's front view: the camera at -Y), 90 from +X (right side).
Workbench, studio light, object colours by random (pieces) or a single grey.
Result: {"images": {view: path}, "triangles": n, "objects": n, "dimensions": [x, y, z]}.
"""
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).parent))
from _pdx import args as job_args, clear_scene, io_module, write_result  # noqa: E402


def _import(path: Path) -> list:
    before = set(bpy.data.objects)
    ext = path.suffix.lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif ext == ".mesh":
        io_module().import_meshfile(str(path))
    else:
        raise SystemExit(f"can't import {path}")
    return [o for o in bpy.data.objects if o not in before]


def main() -> None:
    args = job_args()
    clear_scene()
    objs = [o for o in _import(Path(args["model"])) if o.type == "MESH"]
    if not objs:
        raise SystemExit("no mesh in the model")
    pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    centre, extent = (lo + hi) / 2, max((hi - lo).length, 1e-6)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = args.get("colour", "SINGLE")
    if args.get("texture"):  # a diffuse image on the model's UVs
        mat = bpy.data.materials.new("tex")
        mat.use_nodes = True
        node = mat.node_tree.nodes.new("ShaderNodeTexImage")
        node.image = bpy.data.images.load(args["texture"])
        mat.node_tree.links.new(node.outputs["Color"], mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
        for o in objs:
            o.data.materials.clear()
            o.data.materials.append(mat)
        scene.display.shading.color_type = "TEXTURE"
    scene.display.shading.single_color = (0.62, 0.62, 0.64)
    scene.display.shading.show_cavity = True
    scene.render.resolution_x = scene.render.resolution_y = int(args.get("size", 512))
    scene.render.film_transparent = False
    world = bpy.data.worlds.new("w")
    scene.world = world
    scene.display.shading.background_type = "VIEWPORT"
    scene.display.shading.background_color = (0.85, 0.85, 0.87)
    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = extent * 1.05
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    out_dir = Path(args["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    images = {}
    for view, (az, el) in args["views"].items():
        a, e = math.radians(az), math.radians(el)
        direction = Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))
        cam.location = centre + direction * extent * 2
        cam.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
        path = out_dir / f"{args['name']}_{view}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        images[view] = str(path)
    tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in objs)
    write_result(args, {"images": images, "triangles": tris, "objects": len(objs), "dimensions": list(hi - lo)})


main()
