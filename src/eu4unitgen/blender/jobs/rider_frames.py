"""Blender job (M23): a hand node's frame in the seated rider pose, for placing a strapped shield.

The base rider body (a .mesh) is posed at frame 1 of a rider animation (io_pdx_mesh, as render_unit and
check_deform do). A static part on ``node`` is drawn as the node's posed matrix times its PDX coordinates (io_pdx_mesh
swaps y and z: render_unit's S), so for any world direction or point the job gives the PDX vector in the part's own
frame that lands there. Reported, in that frame:
  * ``elbow``: where the forearm's other end (``elbow`` bone) is: a shield strapped to the forearm sits between it
    and the origin (the hand);
  * ``out``: the rider's outward side (away from the spine, horizontal, on the node's side);
  * ``up``: world up;
  * ``forward``: the rider's forward (horizontal).

The node's frame is the game's (render_unit's): the posed bone matrix times the offset from Blender's bone rest
to the bind pose the .mesh records (``bind``: the skeleton summary's 12 numbers). On western_generic_01_type1 the
offset is the identity (checked for Left_hand_node, test 33's round); applied anyway, as render_unit does.

args: {"mesh", "anim", "node", "elbow", "shoulder", "bind", "result_path"}
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402 - Blender module
from mathutils import Matrix, Vector  # noqa: E402 - Blender module

import _pdx  # noqa: E402
import check_deform as cd  # noqa: E402

S = Matrix(((1, 0, 0), (0, 0, 1), (0, 1, 0)))
S4 = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))


def main():
    a = _pdx.args()
    _pdx.clear_scene()
    ie = _pdx.io_module()
    before = set(bpy.data.objects)
    ie.import_meshfile(a["mesh"])
    rig = next(o for o in bpy.data.objects if o not in before and o.type == "ARMATURE")
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    cd.import_anim(ie, rig, a["anim"])
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    pb = rig.pose.bones
    world = lambda name: rig.matrix_world @ pb[name].matrix  # noqa: E731
    P = world(a["node"])
    if a.get("bind"):  # as render_unit.place: the bone's rest -> the .mesh's bind pose, the scale taken out
        b = a["bind"]
        rest_local = S4 @ Matrix([b[0:4], b[4:8], b[8:12], [0, 0, 0, 1]]) @ S4
        P = P @ (rig.data.bones[a["node"]].matrix_local.inverted() @ rest_local)
        loc, rot, _ = P.decompose()
        P = Matrix.Translation(loc) @ rot.to_matrix().to_4x4()
    inv = P.inverted()
    rot_inv = inv.to_3x3()
    to_part = lambda v: S @ v  # Blender (the node's) local -> the part's PDX coordinates (S is its own inverse)  # noqa: E731
    elbow = to_part((inv @ world(a["elbow"]).translation))
    spine, shoulder = world("mid_spine").translation, world(a["shoulder"]).translation
    out = shoulder - spine
    out.z = 0.0
    out.normalize()
    up = Vector((0.0, 0.0, 1.0))
    # forward (the rider faces PDX -Z, Blender -Y): out x up on the left, up x out on the right (checked in the
    # gallery: with the other order a sword hung "back" pointed at the horse's head)
    fwd = out.cross(up) if a["node"].startswith("Left") else up.cross(out)
    res = {"elbow": list(elbow), "out": list(to_part(rot_inv @ out).normalized()), "up": list(to_part(rot_inv @ up).normalized()),
           "forward": list(to_part(rot_inv @ fwd).normalized()), "node_world": list(P.translation)}
    _pdx.write_result(a, res)


main()
