"""Track P1.2: parts from the reference (ref3d/part_gen.py) and the unit's palette (core/palette.py)."""
import json
import math

from PIL import Image, ImageDraw

from eu4unitgen.core import palette
from eu4unitgen.ref3d import part_gen

RED, GOLD = (140, 30, 28), (200, 160, 70)


def _kite_crop(size=256, bite=False, emblem=True):
    """A synthetic shield crop: a red almond, a gold cross on it, optionally a bite out of its top right."""
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.polygon([(128, 236), (70, 120), (80, 40), (128, 20), (176, 40), (186, 120)], fill=RED + (255,))
    if emblem:
        d.rectangle([120, 50, 136, 200], fill=GOLD + (255,))
        d.rectangle([90, 95, 166, 111], fill=GOLD + (255,))
    if bite:
        d.ellipse([150, 10, 200, 60], fill=(0, 0, 0, 0))
    return im


def test_lab_round_trip_and_snap():
    for c in [(0, 0, 0), (255, 255, 255), RED, GOLD, (10, 200, 90)]:
        back = palette.rgb(palette.lab(c))
        assert max(abs(a - b) for a, b in zip(back, c)) <= 1
    pal = [(94, 26, 23), (147, 116, 80)]
    assert palette.snap((100, 30, 30), pal) == (94, 26, 23)
    assert palette.snap((20, 90, 200), pal) == (20, 90, 200)  # too far from every palette colour: kept


def test_palette_classes_by_specular(tmp_path):
    d = Image.new("RGB", (64, 64), (90, 81, 74))
    s = Image.new("RGBA", (64, 64), (0, 0, 0, 5))
    d.paste((147, 116, 80), (0, 0, 16, 64))
    s.paste((0, 0, 0, 12), (0, 0, 16, 64))  # gold
    d.paste((120, 115, 113), (16, 0, 24, 64))
    s.paste((0, 0, 0, 14), (16, 0, 24, 64))  # silver
    d.paste((90, 90, 90), (24, 0, 48, 64))
    s.paste((255, 0, 0, 5), (24, 0, 48, 64))  # the country mask: a grey times the primary
    d.save(tmp_path / "d.png")
    s.save(tmp_path / "s.png")
    p = palette.classes(tmp_path / "d.png", tmp_path / "s.png", primary=(255, 77, 70), size=64)
    assert p["gold"]["colour"] == (147, 116, 80) and p["silver"]["colour"] == (120, 115, 113)
    assert p["cloth"]["colour"] == (90, 27, 24)
    assert p["matte"][0]["colour"] == (90, 81, 74)


def test_locate_projects_parts_onto_the_reference():
    # a figure: a body column and a shield block on the figure's left (+x); the reference shows it mirrored
    pts = {"all": [], "shield": []}
    for i in range(40):
        for j in range(10):
            pts["all"].append([-1 + j * 0.2, i * 0.4, 0.0])
    for i in range(10):
        for j in range(6):
            p = [3 + j * 0.2, 6 + i * 0.4, 0.0]
            pts["all"].append(p)
            pts["shield"].append(p)
    fg = Image.new("L", (200, 200), 0)
    d = ImageDraw.Draw(fg)
    # the body at x 80-100 over the full height, the shield on the image's left (mirrored), mid height
    d.rectangle([80, 20, 100, 180], fill=255)
    d.rectangle([20, 84, 32, 124], fill=255)
    loc = part_gen.locate(pts, fg, keys=("shield",))
    assert loc["mirror"] is True
    x0, y0, x1, y1 = loc["shield"]["box"]
    cx, cy = loc["shield"]["points"][0]
    assert 15 < cx < 40 and 80 < cy < 130
    assert x0 < 25 and x1 > 30
    assert loc["shield"]["hint"].getbbox() is not None


def test_fill_occlusion_fills_a_bite_from_the_mirror_side():
    full = _kite_crop()
    bitten = _kite_crop(bite=True)
    filled, added = part_gen.fill_occlusion(bitten)
    a_full, a_bit, a_fill = (im.getchannel("A").histogram()[255] for im in (full, bitten, filled))
    assert a_bit < a_full and added > 0
    assert a_fill > a_bit + 0.25 * (a_full - a_bit) and a_fill <= a_full  # the bite inside the hull back
    # an unbitten symmetric part gains almost nothing
    assert part_gen.fill_occlusion(full)[1] < 0.02


def test_two_colours_traces_the_emblem():
    f, e, mask = part_gen.two_colours(_kite_crop())
    assert palette.delta_e(f, RED) < 10 and palette.delta_e(e, GOLD) < 10
    mp = mask.load()
    assert mp[128, 150] == 255 and mp[100, 103] == 255  # on the cross
    assert mp[100, 160] == 0  # the field


def test_outline_is_the_crop_silhouette():
    ol = part_gen.outline(_kite_crop(emblem=False), n=32)
    ys = [y for _, y in ol]
    xs = [x for x, _ in ol]
    assert min(ys) < -0.45 and max(ys) > 0.45  # height 1 about the centre
    assert 0.25 < (max(xs) - min(xs)) < 0.7  # the almond's width over its height (116 / 216)


def test_check_scores_a_matching_part_high_and_a_wrong_one_low():
    crop = _kite_crop()
    good = part_gen.check(crop, crop)
    assert good["ok"] and good["iou"] > 0.99 and good["colour_de"] < 1
    wrong = Image.new("RGBA", (256, 256))
    ImageDraw.Draw(wrong).ellipse([28, 28, 228, 228], fill=(30, 60, 200, 255))
    bad = part_gen.check(wrong, crop)
    assert not bad["ok"]


def test_draw_front_fills_triangles_from_the_texture():
    geo = {"p": [-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0], "tri": [0, 1, 2, 0, 2, 3], "u0": [0.5, 0.5] * 4}
    tex = Image.new("RGB", (8, 8), GOLD)
    im = part_gen.draw_front(geo, tex, 64)
    assert im.getpixel((32, 32)) == GOLD + (255,)
    assert im.getchannel("A").getbbox() is not None


def test_finish_texture_moves_colours_onto_the_palette_and_masks_the_cloth(tmp_path):
    d = Image.new("RGB", (32, 32), (150, 40, 40))
    d.paste((210, 170, 80), (0, 0, 16, 32))
    s = Image.new("RGBA", (32, 32), (0, 0, 0, 5))
    d.save(tmp_path / "d.png")
    s.save(tmp_path / "s.png")
    cloth, gold = (94, 26, 23), (147, 116, 80)
    r = part_gen.finish_texture(tmp_path / "d.png", tmp_path / "s.png", [((150, 40, 40), cloth), ((210, 170, 80), gold)],
                                (255, 77, 70), cloth, 0.0, tmp_path / "out")
    out = Image.open(r["diffuse"]).convert("RGB")
    sp = Image.open(r["specular"])
    assert palette.delta_e(out.getpixel((4, 4)), gold) < 4
    g = out.getpixel((28, 4))
    assert g[0] == g[1] == g[2]  # the cloth: a grey on the mask
    assert sp.getpixel((28, 4))[0] == 255 and sp.getpixel((4, 4))[0] == 0
    assert 0.4 < r["mask_share"] < 0.6


def test_project_emblem_paints_front_faces_only(tmp_path):
    # a quad facing +z (front) and the same quad facing -z (back), on separate halves of the uv square
    p = [-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0] * 2
    uv = [0, 1, 0.5, 1, 0.5, 0, 0, 0] + [0.5, 1, 1, 1, 1, 0, 0.5, 0]
    tri = [0, 1, 2, 0, 2, 3, 4 + 0, 4 + 2, 4 + 1, 4 + 0, 4 + 3, 4 + 2]
    (tmp_path / "g.json").write_text(json.dumps({"p": p, "tri": tri, "u0": uv}), encoding="utf-8")
    Image.new("RGBA", (64, 64), RED + (255,)).save(tmp_path / "d.png")
    em = Image.new("L", (100, 100), 255)  # everything is emblem
    inner = Image.new("L", (100, 100), 255)
    r = part_gen.project_emblem(tmp_path / "g.json", tmp_path / "d.png", em, inner, (0, 0, 100, 100), 0, GOLD, RED,
                                tmp_path / "o.png")
    out = Image.open(tmp_path / "o.png").convert("RGB")
    assert out.getpixel((16, 32)) == GOLD  # the front quad's half
    assert out.getpixel((48, 32)) == RED  # the back quad's half
    assert r["emblem_texels"] > 500


def test_deskew_stands_a_leaning_part_upright():
    im = Image.new("RGBA", (200, 200))
    ImageDraw.Draw(im).rectangle([90, 30, 110, 170], fill=RED + (255,))
    leaning = im.rotate(12, resample=Image.BICUBIC)
    up, turn = part_gen.deskew(leaning)
    assert abs(abs(turn) - 12) < 2
    x0, y0, x1, y1 = up.getchannel("A").point(lambda v: 255 if v > 127 else 0).getbbox()
    assert (x1 - x0) < 30 and (y1 - y0) > 120


def test_read_helmet_finds_its_gold():
    im = Image.new("RGBA", (128, 160), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([14, 20, 114, 150], fill=(150, 150, 155, 255))  # steel
    d.rectangle([54, 0, 74, 30], fill=GOLD + (255,))  # finial
    d.rectangle([14, 100, 114, 110], fill=GOLD + (255,))  # brow band
    d.rectangle([60, 110, 68, 150], fill=GOLD + (255,))  # nasal
    r = part_gen.read_helmet(im)
    assert r["finial"] and r["brow"] and r["nasal"] and not r["ribs"]
    plain = Image.new("RGBA", (128, 160), (0, 0, 0, 0))
    ImageDraw.Draw(plain).ellipse([14, 20, 114, 150], fill=(150, 150, 155, 255))
    r2 = part_gen.read_helmet(plain)
    assert not (r2["finial"] or r2["brow"] or r2["nasal"] or r2["ribs"])


def test_hair_colours_dark_and_light():
    im = Image.new("RGBA", (64, 64), (80, 12, 20, 255))
    d = ImageDraw.Draw(im)
    for y in range(0, 64, 4):
        d.line([(0, y), (63, y)], fill=(160, 40, 55, 255))
    dark, light = part_gen.hair_colours(im)
    assert palette.lum(dark) < palette.lum(light)
    assert palette.delta_e(dark, (80, 12, 20)) < 5 and palette.delta_e(light, (160, 40, 55)) < 5


def test_drop_specks():
    m = Image.new("L", (50, 50), 0)
    d = ImageDraw.Draw(m)
    d.rectangle([10, 10, 40, 40], fill=255)
    d.point((2, 2), fill=255)
    out = part_gen.drop_specks(m, 0.01)
    assert out.getpixel((2, 2)) == 0 and out.getpixel((20, 20)) == 255
    assert math.isclose(out.histogram()[255], 31 * 31)


def test_emblem_match_tells_a_crisp_emblem_from_a_broken_one():
    crisp = _kite_crop()
    assert part_gen.emblem_match(crisp, crisp) > 0.9
    broken = _kite_crop(emblem=False)
    d = ImageDraw.Draw(broken)
    for y in range(50, 200, 30):  # the cross in scattered blobs
        d.ellipse([110 + (y % 40), y, 126 + (y % 40), y + 10], fill=GOLD + (255,))
    assert part_gen.emblem_match(broken, crisp) < part_gen.EMBLEM_MIN
