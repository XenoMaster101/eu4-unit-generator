"""F5: a reference image's palette per horizontal band becomes suggested finishes and colours."""
from PIL import Image, ImageDraw

from eu4unitgen.core.prompt import Interpretation
from eu4unitgen.core.reference import band_colours, finish_for, suggestions


def _soldier(bg=(250, 250, 250, 255), torso=(225, 215, 190), legs=(190, 140, 115), crest=(180, 20, 25)):
    """A made-up front view: crest, head, torso, legs on a plain background."""
    img = Image.new("RGBA", (200, 400), bg)
    d = ImageDraw.Draw(img)
    d.rectangle((80, 40, 120, 110), fill=crest)          # head band: the crest and helmet
    d.rectangle((60, 110, 140, 240), fill=torso)         # torso
    d.rectangle((70, 240, 130, 380), fill=legs)          # legs
    return img


def test_finishes_by_colour():
    assert finish_for((226, 214, 184)) == "linen"
    assert finish_for((140, 97, 90)) == "skin"
    assert finish_for((150, 105, 40)) == "bronze"
    assert finish_for((130, 130, 135)) == "iron"
    assert finish_for((40, 70, 160)) == "colour"


def test_bands_skip_the_background_and_suggest_chips():
    cols = band_colours(_soldier())
    assert cols["torso"] == (225, 215, 190) and cols["legs"] == (190, 140, 115) and cols["head"] == (180, 20, 25)
    items = suggestions(_soldier())
    by = {(i.kind, i.area): i for i in items}
    assert by[("finish", "torso")].value == "linen" and by[("finish", "legs")].value == "skin"
    assert by[("crest_colour", None)].colour == (180, 20, 25)
    assert by[("unit_colour", None)].colour == (180, 20, 25)
    look = Interpretation("", items).look()
    assert look.tier(4)["body"]["torso"] == "linen" and look.crest_colour == [180, 20, 25]


def test_transparent_background_and_a_coloured_tunic():
    items = suggestions(_soldier(bg=(0, 0, 0, 0), torso=(40, 70, 160), crest=(120, 90, 50)))
    by = {(i.kind, i.area): i for i in items}
    assert by[("finish", "torso")].value == "colour" and by[("finish", "torso")].colour == (40, 70, 160)
    assert ("crest_colour", None) not in by  # a bronze-looking head band is the helmet, not a crest
    assert suggestions(Image.new("RGB", (50, 50), (9, 9, 9))) == []  # nothing but background
