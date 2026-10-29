"""The known fixes (Track P1.3): every lesson from the horse and rider tracks (SPEC M15-M24, tests 15-35) as a
general rule, the case that taught it, where it lives, and the test that checks it.

``test`` is a pytest node ("tests/<file>.py::<name>", a parametrised case in brackets) or a measure of the regression
set ("regress:<metric>", ref3d.regress.METRICS): the three reference models are rebuilt after every pipeline change
and a metric worse than the last good run is flagged. tests/test_known_fixes.py checks that every test and every
``where`` exists. ``python -m eu4unitgen.core.known_fixes`` prints the SPEC table.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fix:
    key: str
    rule: str
    case: str
    where: str  # "<path under src/eu4unitgen>:<symbol>"
    tests: tuple[str, ...]
    area: str = "horse"


J = "blender/jobs/"
FIXES: tuple[Fix, ...] = (
    # ---------------------------------------------------------------- the model, any subject
    Fix("weld", "TRELLIS.2's mesh is welded at its UV seams before anything else, and only its largest piece is kept",
        "M21: the mesh comes split at its seams; a collapse on it stalls at 36-61k triangles",
        J + "fresh_horse.py:merge_seams", ("regress:tris",), "model"),
    Fix("facing", "the model is turned to face -Z: its long axis, the end that reaches higher holds the head",
        "M21: TRELLIS.2 picks its own facing (the cataphract 90 degrees off horse 2)",
        J + "fresh_horse.py:face_forward", ("regress:iou_side",), "model"),
    Fix("solid", "a closed solid before decimation: flood fill (closing 0.2 kept only where thick), voxel remesh, relaxed",
        "M21: the cataphract's open surface became a hollow shell that collapsed to lace",
        J + "fresh_horse.py:solid_fill", ("regress:iou_front34", "regress:tris"), "model"),
    Fix("decimate", "quadric collapse straight to the budget (2,800 body triangles), degenerate edges dissolved; the close "
        "retried coarser if it stalls", "M21: an even remesh lost the silhouette; collapses stalled on sheets",
        J + "fresh_horse.py:collapse_to", ("regress:tris", "regress:flips_idle"), "model"),
    Fix("uv_regions", "region UV islands (legs, head, trunk, main panels, tail), borders straightened, panels at 2x density",
        "M21: Smart UV's many islands packed at 45-59%", J + "fresh_horse.py:region_uv", ("regress:uv_used",), "model"),
    Fix("colour_from_raw", "the colour is baked from the raw model itself (selected to active), misses and gutters filled "
        "from the nearest reached texel", "M21 v001: a projected/painted colour read blurry",
        J + "bake_colour.py:main", ("regress:bake_reached",), "model"),
    Fix("colour_matched", "the bake's colour matched in Lab to the raw model's lit render, lightness half way, spread capped",
        "test 26: the raw base colour read orange and flat next to the approved raw render",
        J + "bake_colour.py:colour_transfer", ("regress:bake_reached", "regress:silver_cap"), "model"),
    Fix("metal_cap", "bright metal is held under the raw model's own: silver at 0.85 of its luminance",
        "test 26: the scale rendered nearly white; test 12: bright metals read pale in game",
        J + "bake_colour.py:silver_mask", ("regress:silver_cap",), "model"),
    Fix("gloss_classes", "gloss by colour class in the specular alpha: silver 14, gold 12, cloth and coat matte 5",
        "M21: TRELLIS.2's metallic map covered coat and hair alike (0.65 on horse 2)",
        J + "bake_colour.py:gloss_map", ("regress:gloss_gold", "regress:gloss_silver"), "model"),
    Fix("cloth_any_colour", "the cloth is the subject's own dominant saturated non-gold colour, not a fixed red; a horse's "
        "coat is told apart from it", "Track P1 audit: the country mask, head band, emblem and crest only worked on red",
        J + "_colour.py:cloth_and_coat", ("tests/test_blender_rules.py::test_rule[cloth_hue_blue]",
                                          "tests/test_blender_rules.py::test_rule[coat_kept_apart_from_cloth]",
                                          "tests/test_blender_rules.py::test_rule[red_cloth_window_skips_copper]",
                                          "regress:cloth_primary", "regress:mask_share"), "model"),
    Fix("country_colour", "with the option on, the cloth is the country colour's mask (a grey times the primary); the set's "
        "tags without colours of their own get the cloth's colour", "test 28: Byzantium took its map purple",
        J + "bake_colour.py:cloth_mask", ("tests/test_trellis_pipeline.py::test_horse_unit_colours_fill_tags_without_their_own",),
        "model"),
    Fix("emblem_decal", "emblems are crisp decals: a mask painted over the blurred bake (horse panels) or traced from the "
        "reference's crop and projected from its view (shields)", "M21: the caparison's crosses were blurred",
        "ref3d/part_gen.py:project_emblem", ("tests/test_trellis_pipeline.py::test_tetragrammatic_decal",
                                             "tests/test_part_gen.py::test_two_colours_traces_the_emblem",
                                             "tests/test_part_gen.py::test_project_emblem_paints_front_faces_only",
                                             "regress:emblem_texels"), "model"),
    Fix("unseen_filled", "what the reference hides is filled from matching visible parts: texels from the nearest reached "
        "ones, a part's hidden bits from its mirror side inside its hull", "M20: hidden areas were smeared",
        "ref3d/part_gen.py:fill_occlusion", ("tests/test_part_gen.py::test_fill_occlusion_fills_a_bite_from_the_mirror_side",),
        "model"),
    Fix("palette", "parts take the unit's palette: their colours moved onto the horse's red, gold and steel",
        "test 34: the shield must use the caparison's red and the chanfron's gold",
        "core/palette.py:classes", ("tests/test_part_gen.py::test_palette_classes_by_specular",
                                    "tests/test_part_gen.py::test_finish_texture_moves_colours_onto_the_palette_and_masks_the_cloth"),
        "model"),
    # ---------------------------------------------------------------- the horse
    Fix("scale_seat", "one uniform scale (proportions kept): the croup, then the seat at vanilla's (rays at Saddle_Node)",
        "M21: the rider sits on Saddle_Node", J + "fresh_horse.py:direct_fit", ("regress:seat_gap",)),
    Fix("slim", "a horse whose side silhouette is over 1.15x vanilla's is slimmed about the seat, legs stretched to the ground",
        "test 26: 1.44-1.65x vanilla's slim horse", J + "fresh_horse.py:slim", ("regress:side_ratio",)),
    Fix("head_before_slim", "the head is found before the slim and its poll carried through",
        "test 26: after it the cataphract's chest was taken for its head", J + "fresh_horse.py:direct",
        ("regress:iou_side",)),
    Fix("minimal_pose", "minimal posing: only the lower legs slide under vanilla's heels; head and neck keep their shape "
        "(put on vanilla's poll only for the weights)", "test 25 v001: the reshape bent the neck into a goose curve",
        J + "fresh_horse.py:direct_pose", ("regress:iou_front34", "regress:iou_side")),
    Fix("head_droop", "the fitted head pitch is kept (no repose onto vanilla's 40 degrees)",
        "M20: the game head drooped, long and thin", J + "fresh_horse.py:head_shift", ("regress:iou_side",)),
    Fix("tail", "the tail is found by its own colour, cut off and replaced by our tube on vanilla's tail bones, painted "
        "from it; the body's tail weights cleared", "M21: the strands tore in idle; the rump swung with the tail",
        J + "fresh_horse.py:hair", ("regress:tail", "regress:flips_idle")),
    Fix("mane_crest", "the mane's voxels are smoothed hard into one solid crest; an opening on head and neck removes flaps",
        "test 25: mane slabs and flaps", J + "fresh_horse.py:smooth_region", ("regress:flips_moving",)),
    Fix("head_band", "with a chanfron, a clean band of the head's cloth behind its rear edge",
        "test 26: the red blobs by the eyes", J + "bake_colour.py:head_band", ("regress:head_band",)),
    Fix("thin_marks", "tassels and a horse's plume too small for geometry are painted marks where the reference has them",
        "test 27: the tassels sat on the ears like hairpins", J + "bake_colour.py:head_marks", ("regress:head_marks",)),
    Fix("reins", "thin parts are rebuilt, not lost: the reins as straps from the bit to vanilla's Rem bone, coloured "
        "from the raw model's, when it shows them", "M22's known limit: the reins were lost to the budget",
        J + "_reins.py:build", ("tests/test_blender_rules.py::test_rule[reins_found_when_the_raw_shows_them]",
                                "tests/test_blender_rules.py::test_rule[reins_weights_head_to_rem]", "regress:reins")),
    Fix("chain_weights", "chain weights along vanilla's bones; the tail along its own chain", "M21",
        J + "fresh_horse.py:transfer_weights", ("regress:flips_attack",)),
    # ---------------------------------------------------------------- the rider
    Fix("split_weapons", "the lance and shield are split off the body (the column beside it; the plate on the other side)",
        "M23: the weapons came fused to the hands", J + "fresh_rider.py:split", ("regress:parts",), "rider"),
    Fix("helmet_scale", "the rider is scaled by its helmet's top (not the crest's)", "M23: the crest was taken for the head",
        J + "fresh_rider.py:place", ("regress:ref_iou",), "rider"),
    Fix("rider_slim", "x and z slimmed towards vanilla's rider (at least 0.75); arms shift whole",
        "test 29: bulkier than vanilla's rider", J + "fresh_rider.py:slim_body", ("regress:group_flips_torso",), "rider"),
    Fix("arms_turned", "each arm is turned about its shoulder onto its hand node, not slid",
        "M23: sliding the hands 4.6 and 5.9 units stretched the arms", J + "fresh_rider.py:swing_arm",
        ("regress:max_idle", "regress:group_flips_arms"), "rider"),
    Fix("cloth_weights", "the cloak on the spine by height, the skirt halves on the legs, a slot between the legs, the cloth "
        "under the arms cut", "M23: one skirt block tore when the seated legs spread; the cloak hung from the shield arm",
        J + "fresh_rider.py:rider_weights", ("regress:group_flips_legs", "regress:flips_moving"), "rider"),
    Fix("collar_cut", "the collar is cut round the helmet so the head shows clean steel",
        "test 30: the head read as a hooded blob", J + "fresh_rider.py:main", ("regress:ref_iou",), "rider"),
    Fix("helmet_rigid", "the helmet is pinned to the head bone", "test 35: the dome was 70% on the head; the finial floated",
        "ref3d/trellis.py:pin_head", ("tests/test_pin_head.py::test_pin_head", "regress:group_flips_head"), "rider"),
    Fix("legs_clear", "the rider's legs move out by how much wider its horse is than vanilla at the seated thighs' height",
        "Track P1 audit: the cataphract's caparison swallowed 23% of the rider's lower body (vanilla: 9%)",
        "ref3d/trellis.py:leg_spread", ("tests/test_rider_rules.py::test_leg_spread_is_the_extra_width_at_the_same_height",
                                        "tests/test_rider_rules.py::test_spread_legs_moves_only_leg_weighted_vertices_outward",
                                        "tests/test_quality.py::test_legs_clear_and_verdict"), "rider"),
    Fix("shield_forearm", "the shield is strapped to the forearm (Left_hand_node in the seated pose: half way to the elbow, "
        "out 1.2, up 0.3), face out, top up", "test 29: on the rein hand it sat over the horse's neck",
        "ref3d/trellis.py:strap", ("tests/test_rider_rules.py::test_shield_strapped_half_way_to_the_elbow_out_and_up",), "rider"),
    Fix("anim_scale", "head parts are scaled by the rider animation's 0.805", "test 35: tier 1 (scaled) approved in game",
        "ref3d/trellis.py:plume_part", ("tests/test_rider_rules.py::test_head_part_scaled_for_the_rider_animation",), "rider"),
    # ---------------------------------------------------------------- parts
    Fix("parts_from_reference", "parts are never placeholders: each is cut out of the reference (SAM 2), made by TRELLIS.2 "
        "from its crop, decimated, baked (colour and normal), attached to its bone, and checked against the crop; else "
        "procedural, still traced and coloured from the crop", "the user, 2026-10-24 (after test 35)",
        "ref3d/trellis.py:shield_from_reference", ("tests/test_part_gen.py::test_check_scores_a_matching_part_high_and_a_wrong_one_low",
                                                   "regress:part_iou_shield", "regress:part_de_shield"), "parts"),
    Fix("locate_parts", "each part is found in the reference by projecting the model's split part onto it (mirrored or not)",
        "Track P1.2", "ref3d/part_gen.py:locate", ("tests/test_part_gen.py::test_locate_projects_parts_onto_the_reference",
                                                    "regress:locate_fit"), "parts"),
    Fix("occluders", "what covers a part's edge (a gripping hand) is trimmed by colour, and leaning crops are stood upright",
        "Track P1.2: the cataphract's shield crop came with the gauntlet", "ref3d/part_gen.py:trim_foreign",
        ("tests/test_part_gen.py::test_deskew_stands_a_leaning_part_upright",), "parts"),
    Fix("flat_part_front", "a flat part faces along its thinnest axis, its painted side out, its wide end up as in the crop",
        "Track P1.2: the first TRELLIS.2 shield came out edge-on", J + "_part_view.py:flat_view",
        ("tests/test_blender_rules.py::test_rule[flat_part_faces_its_painted_side]",
         "tests/test_blender_rules.py::test_rule[flat_part_wide_end_up]"), "parts"),
    Fix("shield_outline", "a procedural shield's outline, emblem and colours come from its crop (the kite and cross are "
        "only the default)", "test 32: the reference's kite; Track P1: any shield", "core/gear.py:kite_shield",
        ("tests/test_part_gen.py::test_outline_is_the_crop_silhouette", "tests/test_gear.py::test_kite_shield_shape_and_mask"),
        "parts"),
    Fix("shield_design", "the shield belongs to the design: palette colours, worn and shaded, studs all round, 12% darker",
        "test 34: a clean sticker", "ref3d/trellis.py:shield_from_reference",
        ("tests/test_gear.py::test_kite_shield_worn_and_darker",
         "tests/test_part_gen.py::test_finish_texture_moves_colours_onto_the_palette_and_masks_the_cloth"), "parts"),
    Fix("horsehair", "plumes, crests and manes are flat horsehair blades with a jagged top and strand texture, coloured "
        "from their crop", "test 32: a round tube read as a sausage", "core/gear.py:horsehair_blade",
        ("tests/test_gear.py::test_horsehair_blade_flat_and_jagged", "tests/test_gear.py::test_strand_texture_streaks_along_u",
         "tests/test_part_gen.py::test_hair_colours_dark_and_light"), "parts"),
    Fix("plume_separate", "small decorations keep their attachment point and stay separate: the plume rises from a gold "
        "finial, a rigid head part clear of helmet, mail and cloak", "tests 29 and 31: merged into the helmet",
        "ref3d/trellis.py:plume_part", ("tests/test_rider_rules.py::test_plume_takes_the_units_gold_and_a_ridged_normal",
                                        "regress:parts"), "parts"),
    Fix("crest_any_colour", "the crest is found by its own colour over the helmet, whatever it is",
        "Track P1 audit: a fixed red", J + "_colour.py:crest_colour", ("tests/test_blender_rules.py::test_rule[crest_any_colour]",),
        "parts"),
    Fix("helmet_gold", "the helmet's gold (finial, brow band, nasal, ribs) is built where the reference's helmet shows it, "
        "in the unit's gold", "test 34: the reference's tall finial and bands", "ref3d/part_gen.py:read_helmet",
        ("tests/test_part_gen.py::test_read_helmet_finds_its_gold",), "parts"),
    Fix("part_normals", "every part carries a normal map (baked from its model, or from its height)",
        "M24's known limit: the parts' normal maps were flat", "ref3d/trellis.py:process_rider",
        ("tests/test_rider_rules.py::test_sword_is_textured_not_flat", "tests/test_gear.py::test_normal_from_height_signs"),
        "parts"),
    Fix("candidate_pick", "of TRELLIS.2's two candidates the one built scores best on outline, colour (median Lab "
        "distance to the reference) and worst stretch once fitted to the game skeleton (a shape-only trial build)",
        "the user, 2026-10-28: horse 2's seed 42 and 43 tied on outline; the darker mane was closer to the reference",
        "ref3d/trellis.py:pick_score", ("tests/test_app_flow.py::test_pick_weighs_outline_colour_and_stretch",
                                         "tests/test_app_flow.py::test_colour_distance_reads_the_subjects_colours"),
        "model"),
    Fix("foot_parts", "a soldier on foot holds its hand parts in the base game's convention for its stance, not the "
        "seated pose they were split off in: the weapon's point along +z (muslim_spear's), the shield's face along "
        "+x with its back at the hand (indian_generic_01_type1_shield's)",
        "the cataphract on foot, 2026-10-28: the lance hung point down and the kite sat behind the body",
        "ref3d/trellis.py:foot_part", ("tests/test_app_flow.py::test_a_soldiers_hand_parts_follow_the_base_game_on_foot",),
        "parts"),
    Fix("shield_is_a_plate", "a shield is split off only when it is a broad flat plate (its thinnest spread under 0.3 of "
        "its width, its width over 0.035 of the height); an arm held out is left on the body",
        "the English soldier, 2026-10-28: his left forearm was cut off as a shield (0.571 flat, 0.022 broad; the "
        "cataphract's kite: 0.089, 0.062), so he took the spear-and-shield stance and crouched",
        "blender/jobs/fresh_rider.py:split", ("regress:parts", "tests/test_app_flow.py::test_the_regression_set_expects_no_shield_and_a_bare_coat"), "parts"),
    Fix("blade_by_place", "a polearm's broad head (within 0.09 of the height round the shaft, from 0.08 H over the grip, "
        "on the weapon's side) goes with the weapon, by place: it is often a separate piece",
        "the English soldier's poleaxe, 2026-10-28: only the shaft was kept",
        "blender/jobs/fresh_rider.py:split", ("regress:parts", "tests/test_app_flow.py::test_the_regression_set_expects_no_shield_and_a_bare_coat"), "parts"),
    Fix("coat_not_cloth", "a horse's coat is never its cloth: texels within 22 (Lab) of the coat's colour (the head's "
        "median) stay out of the country mask",
        "the bay horse, 2026-10-28: its reddish coat fell in the red cloth's hue window; 40% of it was masked and its "
        "game version came out grey",
        "blender/jobs/bake_colour.py:cloth_mask", ("regress:mask_share", "tests/test_app_flow.py::test_the_regression_set_expects_no_shield_and_a_bare_coat"), "colour"),
)


def markdown() -> str:
    lines = ["| Rule | The case that taught it | Where | Checked by |", "|---|---|---|---|"]
    for f in FIXES:
        tests = "; ".join(t.split("::")[-1] if "::" in t else t for t in f.tests)
        lines.append(f"| {f.rule} | {f.case} | `{f.where}` | {tests} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(markdown())
