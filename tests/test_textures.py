"""Phase 2: DDS writer, hue shift, meshsettings retextures, the texture test mod and its rules.
Synthetic images and the fake install only; real-install checks run when EU4_PATH is set."""
from __future__ import annotations

import io
import os
import struct
from pathlib import Path

import pytest
from PIL import Image

from eu4unitgen.core.generator import GeneratedFile, GenerationError
from eu4unitgen.core.nation import Nation
from eu4unitgen.core.retexture import BESIDE_MESH, OWN_DIR, Retexture, retextured_mesh
from eu4unitgen.export.testmod import build_texture_test, texture_checklist
from eu4unitgen.export.writer import apply_export, stage_plan
from eu4unitgen.library.scanner import GameFiles, build_index, read_dds_header
from eu4unitgen.pdxscript.parser import parse_bytes
from eu4unitgen.textures.dds import DDSError, DDSFormat, check_matches, full_mip_count, mip_chain, read_dds, write_dds
from eu4unitgen.textures.recolour import hue_shift
from eu4unitgen.validate import validate

T = "eu4unitgen_test_textures"


# --------------------------------------------------------------------------- dds

def _gradient(w: int, h: int, alpha: int = 255) -> Image.Image:
    im = Image.new("RGBA", (w, h))
    im.putdata([(x * 255 // max(1, w - 1), y * 255 // max(1, h - 1), 90, alpha) for y in range(h) for x in range(w)])
    return im


@pytest.mark.parametrize("fmt,w,h", [("DXT1", 64, 64), ("DXT5", 64, 32), ("DXT1", 512, 256)])
def test_write_dds_full_mip_chain_vanilla_header(fmt, w, h):
    target = DDSFormat(fmt, w, h, full_mip_count(w, h))
    data = write_dds(_gradient(w, h), target)
    hdr = read_dds_header(data)
    assert hdr == {"width": w, "height": h, "mips": full_mip_count(w, h), "format": fmt, "dx10_header": False}
    assert check_matches(data, target) == []
    flags, = struct.unpack_from("<I", data, 8)
    caps, = struct.unpack_from("<I", data, 108)
    assert flags == 0xA1007 and caps == 0x401008  # what vanilla unit textures carry
    block = 8 if fmt == "DXT1" else 16
    levels = [(max(1, w >> i), max(1, h >> i)) for i in range(full_mip_count(w, h))]
    assert len(data) == 128 + sum(max(1, (a + 3) // 4) * max(1, (b + 3) // 4) * block for a, b in levels)
    back = read_dds(data)
    assert back.size == (w, h)
    assert abs(back.getpixel((w - 1, 0))[0] - 255) < 16  # decodes to roughly the input


def test_write_dds_resizes_to_the_base_size():
    data = write_dds(_gradient(128, 128), DDSFormat("DXT1", 32, 32, 6))
    assert read_dds_header(data)["width"] == 32


def test_mips_keep_colour_where_alpha_is_zero():
    """Spec maps hold country-colour masks in areas with alpha (gloss) 0 (SPEC §4)."""
    levels = mip_chain(Image.new("RGBA", (16, 16), (200, 100, 50, 0)), 5)
    assert levels[-1].getpixel((0, 0)) == (200, 100, 50, 0)


def test_dds_errors():
    with pytest.raises(DDSError):
        write_dds(_gradient(8, 8), DDSFormat("DXT3", 8, 8, 1))
    with pytest.raises(DDSError):
        write_dds(_gradient(8, 8), DDSFormat("DXT1", 8, 8, 9))
    with pytest.raises(DDSError):
        DDSFormat.of({"format": "DX10", "width": 8, "height": 8, "mips": 1, "dx10_header": True})
    good = write_dds(_gradient(16, 16), DDSFormat("DXT5", 16, 16, 5))
    assert check_matches(good, DDSFormat("DXT1", 16, 16, 5)) == ["format DXT5, base is DXT1"]
    assert check_matches(good, DDSFormat("DXT5", 32, 16, 5))[0].startswith("size 16x16")
    assert check_matches(good, DDSFormat("DXT5", 16, 16, 1)) == ["5 mip levels, base has 1"]
    assert check_matches(good[:-8], DDSFormat("DXT5", 16, 16, 5))[0].startswith("truncated")


def test_hue_shift():
    out = hue_shift(Image.new("RGBA", (2, 2), (255, 0, 0, 77)), 120)
    r, g, b, a = out.getpixel((0, 0))
    assert g > 240 and r < 15 and b < 15 and a == 77


@pytest.mark.skipif(not os.environ.get("EU4_PATH"), reason="needs a real EU4 install (EU4_PATH)")
def test_vanilla_textures_round_trip_to_the_same_format():
    files = GameFiles(Path(os.environ["EU4_PATH"]), include_dlc=False)
    try:
        for rel in ("gfx/models/Units/western_generic_01_type1_diffuse.dds",
                    "gfx/models/Units/western_generic_01_type1_spec.dds"):
            data = files.read(rel)
            target = DDSFormat.of(read_dds_header(data))
            out = write_dds(read_dds(data), target)
            assert len(out) == len(data) and check_matches(out, target) == []
    finally:
        files.close()


# --------------------------------------------------------------------------- retexture on the fake install

def _real_dds(fmt: str = "DXT1", colour=(200, 60, 40)) -> bytes:
    return write_dds(Image.new("RGBA", (64, 64), colour + (255,)), DDSFormat(fmt, 64, 64, 7))


@pytest.fixture
def tex_install(fake_install: Path) -> Path:
    """The fake install with decodable diffuse maps and the shader file meshsettings names."""
    units = fake_install / "gfx/models/Units"
    for tex in ("g1_d", "g2_d", "h_d", "c_d"):
        (units / f"{tex}.dds").write_bytes(_real_dds())
    (fake_install / "gfx/FX").mkdir(parents=True, exist_ok=True)
    (fake_install / "gfx/FX/pdxmesh.shader").write_bytes(b"# made-up stand-in\n")
    return fake_install


@pytest.fixture
def tex_game(tex_install):
    files = GameFiles(tex_install)
    yield files, build_index(files)
    files.close()


def test_retextured_mesh_beside_and_own_dir(project, tex_game):
    files, idx = tex_game
    shift = lambda im: hue_shift(im, 120)  # noqa: E731
    mesh, made = retextured_mesh(project.naming, "mac", Retexture("infantry", 1, "generic_01", BESIDE_MESH, shift),
                                 files, idx)
    assert mesh["name"] == "hr_mac_infantry_1_mesh" and mesh["file"] == "gfx/models/Units/generic_01.mesh"
    assert mesh["settings"][0] == {"shape": "bodyShape", "index": 0, "diffuse": "hr_mac_infantry_1_diffuse.dds",
                                   "normal": "g_n.dds", "specular": "g_s.dds", "shader": "PdxMeshColor",
                                   "shader_file": "gfx/FX/pdxmesh.shader"}
    assert [f.path for f in made] == ["gfx/models/Units/hr_mac_infantry_1_diffuse.dds"]
    assert made[0].derived_from == "gfx/models/Units/g1_d.dds"
    assert read_dds(made[0].data).getpixel((0, 0))[1] > 150  # red-ish base shifted to green

    mesh, made = retextured_mesh(project.naming, "mac", Retexture("horse", 2, "horse_3", OWN_DIR, shift), files, idx)
    assert made[0].path == "gfx/models/units/hr_mac/hr_mac_horse_2_diffuse.dds"
    s = mesh["settings"][0]
    assert s["diffuse"] == made[0].path and s["normal"] == "gfx/models/Units/h_n.dds"


def test_retexture_refuses_non_base_meshes(project, tex_game):
    files, idx = tex_game
    with pytest.raises(GenerationError, match="isn't a base-game pdxmesh"):
        retextured_mesh(project.naming, "mac", Retexture("infantry", 1, "nope", BESIDE_MESH, lambda im: im), files, idx)
    with pytest.raises(GenerationError):
        Retexture("rider", 1, "generic_01", BESIDE_MESH, lambda im: im)


@pytest.fixture
def texture_test(project, tex_game, tmp_path):
    files, idx = tex_game
    nation = Nation("AAA", "hellenicgfx", set_name="macedonian_test", adjective="Macedonian Test")
    mod_dir = tmp_path / "eu4_user" / "mod"
    mod_dir.mkdir(parents=True)
    return build_texture_test(project, nation, files, idx, mod_dir, {2: 2, 3: 3, 4: 4})


def test_texture_test_mod_is_clean_and_complete(texture_test):
    tm = texture_test.mod
    assert tm.report.errors == [] and tm.report.warnings == [], "\n".join(map(str, tm.report.issues))
    paths = {f.path for f in tm.plan.files}
    for p in (f"{T}.mod", f"{T}/interface/assets/hr_macedonian_test.gfx",
              f"{T}/gfx/models/Units/hr_macedonian_test_infantry_1_diffuse.dds",
              f"{T}/gfx/models/Units/hr_macedonian_test_cannon_1_diffuse.dds",
              f"{T}/gfx/models/units/hr_macedonian_test/hr_macedonian_test_horse_2_diffuse.dds"):
        assert p in paths, p
    gfx = parse_bytes(tm.package.files["interface/assets/hr_macedonian_test.gfx"].data)
    assert gfx.warnings == []
    meshes = {p.get_str("name"): p for p in gfx.get_block("objectTypes").get_all("pdxmesh")}
    assert sorted(meshes) == sorted(f"hr_macedonian_test_{r}_{t}_mesh" for r in ("infantry", "horse", "cannon")
                                    for t in (1, 2))
    ms = meshes["hr_macedonian_test_horse_2_mesh"].get_block("meshsettings")
    assert ms.get_str("texture_diffuse") == "gfx/models/units/hr_macedonian_test/hr_macedonian_test_horse_2_diffuse.dds"
    assert meshes["hr_macedonian_test_horse_2_mesh"].get_str("file") == "gfx/models/Units/horse_3.mesh"

    ents = {e.get_str("name"): e for e in parse_bytes(
        tm.package.files["gfx/entities/hr_macedonian_test_units.asset"].data).get_all("entity")}
    assert ents["AAA_INFANTRY_1"].get_str("pdxmesh") == "hr_macedonian_test_infantry_1_mesh"
    assert ents["AAA_CAVALRY_1_rider"].get_str("pdxmesh") == "hr_macedonian_test_infantry_1_mesh"
    assert ents["AAA_CAVALRY_2"].get_str("pdxmesh") == "hr_macedonian_test_horse_2_mesh"
    assert ents["AAA_INFANTRY_3"].get_str("pdxmesh") == "generic_03"  # control tiers untouched
    cannon = ents["AAA_ARTILLERY_1_cannon"]
    assert cannon.get_str("pdxmesh") == "hr_macedonian_test_cannon_1_mesh"
    attack = next(s for s in cannon.get_all("state") if s.get_str("name") == "attack")
    assert attack.get_block("event").get_str("node") == "cannon_muzzle"  # muzzle event restated
    gunman = ents["AAA_ARTILLERY_1"].get_block("attach")
    assert gunman.get_str("cannon_node") == "AAA_ARTILLERY_1_cannon"
    assert "AAA_ARTILLERY_3_cannon" not in ents

    text = texture_checklist(texture_test, {2: 2, 3: 3, 4: 4})
    assert "VARIANT A = tier 1" in text and "VARIANT B = tier 2" in text and T in text


def test_texture_test_stages_binary_files(texture_test, tmp_path):
    stage = tmp_path / "stage"
    written = apply_export(stage_plan(texture_test.mod.plan, stage))
    dds = stage / T / "gfx/models/Units/hr_macedonian_test_infantry_1_diffuse.dds"
    assert dds.is_file() and read_dds_header(dds.read_bytes())["format"] == "DXT1"
    assert any(p.endswith(".dds") for p in written)
    assert "Binary " in texture_test.mod.plan.diff_text()


# --------------------------------------------------------------------------- validator rules

def _revalidate(project, tex_game, test, change):
    files, idx = tex_game
    pkg = test.mod.package
    change(pkg)
    return validate(project, test.nation, pkg, files, idx)


def test_dds_format_rule(project, tex_game, texture_test):
    rel = "gfx/models/Units/hr_macedonian_test_infantry_1_diffuse.dds"

    def tamper(pkg):
        f = pkg.files[rel]
        pkg.files[rel] = GeneratedFile(f.path, write_dds(Image.new("RGBA", (64, 64)), DDSFormat("DXT5", 64, 64, 7)),
                                       f.role, derived_from=f.derived_from)
        pkg.files["gfx/models/units/x.dds"] = GeneratedFile("gfx/models/units/x.dds", f.data, f.role,
                                                            derived_from="gfx/models/Units/nothere.dds")
    report = _revalidate(project, tex_game, texture_test, tamper)
    msgs = [i.message for i in report.errors if i.rule == "dds_format"]
    assert any("format DXT5, base is DXT1" in m for m in msgs)
    assert any("missing from the install" in m for m in msgs)
    assert any(i.rule == "mesh_settings" and "isn't named" in i.message for i in report.warnings)


def test_mesh_settings_rule(project, tex_game, texture_test):
    rel = "interface/assets/hr_macedonian_test.gfx"

    def tamper(pkg):
        f = pkg.files[rel]
        text = f.data.decode().replace('name = "bodyShape"', 'name = "wrongShape"', 1)
        text = text.replace('"hr_macedonian_test_cannon_1_diffuse.dds"', '"missing.dds"')
        pkg.files[rel] = GeneratedFile(f.path, text.encode(), f.role)
    report = _revalidate(project, tex_game, texture_test, tamper)
    msgs = [i.message for i in report.errors if i.rule == "mesh_settings"]
    assert any("not 'wrongShape'" in m for m in msgs)
    assert any("missing.dds" in m and "doesn't exist" in m for m in msgs)
    assert any(i.rule == "missing_nodes" for i in report.errors)
