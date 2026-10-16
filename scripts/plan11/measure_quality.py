"""Plan: "match vanilla quality". Measure texture style and texel density of base-game unit meshes
against our generated parts (read-only; the install is read in place; prints numbers only).

  texel density: sqrt(sum UV area in texels / sum 3D surface area), texels per mesh unit (the
                 pdxmesh scale is 0.5 for bodies, weapons and our parts alike)
  normal map:    mean tilt of the tangent-space normal from flat (degrees); 0 = flat
  detail:        mean std dev of luminance in 8x8 blocks (fine painted detail), over covered texels
  shading:       std dev of luminance after a 16 px blur (broad painted light/shadow, AO) over covered texels
  value range:   5th..95th percentile luminance over covered texels
  gloss:         spec alpha mean (vanilla's gloss channel)

usage: python measure_quality.py <staged mod folder>
"""
import math
import os
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageStat

from eu4unitgen.library.scanner import GameFiles, _walk, parse_pdx_mesh, summarize_mesh
from eu4unitgen.textures.dds import read_dds


def mesh_info(data):
    t = parse_pdx_mesh(data)
    m = next(n for n in _walk(t) if n["name"] == "mesh")["props"]
    return m, summarize_mesh(t)["shapes"][0]


def density(m, size):
    p, u, tri = m["p"], m["u0"], m["tri"]
    area3 = areauv = 0.0
    for k in range(0, len(tri), 3):
        a, b, c = tri[k:k + 3]
        e1 = [p[b * 3 + j] - p[a * 3 + j] for j in range(3)]
        e2 = [p[c * 3 + j] - p[a * 3 + j] for j in range(3)]
        cr = (e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0])
        area3 += 0.5 * math.sqrt(sum(x * x for x in cr))
        ux = [(u[i * 2] * size[0], u[i * 2 + 1] * size[1]) for i in (a, b, c)]
        areauv += 0.5 * abs((ux[1][0] - ux[0][0]) * (ux[2][1] - ux[0][1]) - (ux[2][0] - ux[0][0]) * (ux[1][1] - ux[0][1]))
    return math.sqrt(areauv / area3) if area3 else 0.0


def coverage_mask(m, size):
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    u, tri = m["u0"], m["tri"]
    for k in range(0, len(tri), 3):
        d.polygon([(u[i * 2] * size[0], u[i * 2 + 1] * size[1]) for i in tri[k:k + 3]], fill=255)
    return mask


def style(diff, norm, spec, mask):
    lum = diff.convert("L")
    w, h = lum.size
    px = list(lum.getdata())
    mk = list(mask.getdata())
    covered = [v for v, k in zip(px, mk) if k]
    covered.sort()
    q = lambda f: covered[int(f * (len(covered) - 1))] if covered else 0  # noqa: E731
    blocks = []
    for y in range(0, h - 7, 8):
        for x in range(0, w - 7, 8):
            if mask.crop((x, y, x + 8, y + 8)).getextrema()[0] == 255:
                blocks.append(ImageStat.Stat(lum.crop((x, y, x + 8, y + 8))).stddev[0])
    blur = lum.filter(ImageFilter.GaussianBlur(16 * w / 512))
    bl = [v for v, k in zip(blur.getdata(), mk) if k]
    mean = sum(bl) / len(bl) if bl else 0
    shading = math.sqrt(sum((v - mean) ** 2 for v in bl) / len(bl)) if bl else 0
    tilts = []
    if norm is not None:
        n = norm.convert("RGB").resize(lum.size)
        for (r, g, b), k in zip(n.getdata(), mk):
            if k:
                x, y = r / 127.5 - 1, g / 127.5 - 1
                tilts.append(math.degrees(math.asin(min(1.0, math.hypot(x, y)))))
    gloss = None
    if spec is not None:
        a = [px_[3] for px_, k in zip(spec.convert("RGBA").resize(lum.size).getdata(), mk) if k]
        gloss = sum(a) / len(a) if a else None
    return {"detail": sum(blocks) / len(blocks) if blocks else 0, "shading": shading, "range": (q(0.05), q(0.95)),
            "normal_tilt": sum(tilts) / len(tilts) if tilts else 0, "gloss": gloss}


def tex(read, folder, name):
    data = read(f"{folder}/{name}") if name else None
    if data is None:
        return None
    try:
        return Image.open(BytesIO(data))
    except OSError:
        return read_dds(data)


def report(label, read, file):
    m, sh = mesh_info(read(file))
    folder = file.rsplit("/", 1)[0]
    d = tex(read, folder, sh["diffuse"])
    n = tex(read, folder, sh["normal"])
    s = tex(read, folder, sh["specular"])
    mask = coverage_mask(m, d.size)
    st = style(d, n, s, mask)
    print(f"{label:34} tex {d.size[0]:4}  tris {len(m['tri']) // 3:5}  density {density(m, d.size):6.1f} tx/unit  "
          f"detail {st['detail']:5.1f}  shading {st['shading']:5.1f}  range {st['range'][0]:3}-{st['range'][1]:3}  "
          f"normal tilt {st['normal_tilt']:5.1f} deg  gloss {st['gloss'] if st['gloss'] is None else round(st['gloss']):>4}")


def main():
    g = GameFiles(Path(os.environ["EU4_PATH"]), include_dlc=False)
    stage = Path(sys.argv[1])
    print("BASE GAME")
    for name in ("western_generic_01_type1", "african_generic_03_type1", "western_halbard", "western_musket",
                 "cavalry_saber", "indian_generic_01_type1_shield", "south_american_generic_01_type1_shield"):
        report(name, g.read, f"gfx/models/Units/{name}.mesh")
    print("OURS (test 11)")
    read = lambda rel: (stage / rel).read_bytes() if (stage / rel).is_file() else None  # noqa: E731
    for name in ("dory_1", "hoplon_1", "greek_helmet_1"):
        report(name, read, f"gfx/models/units/hr_macedonian_test/hr_macedonian_test_{name}.mesh")


if __name__ == "__main__":
    main()
