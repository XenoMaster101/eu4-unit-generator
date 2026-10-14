"""A synthetic EU4 install with a complete 4-tier westerngfx land-unit chain.

Structure mirrors vanilla 1.37.5 (clone chains, attach nodes, animation ids, a DLC
pack + version override) but every byte is made up here: no Paradox content.
"""
from __future__ import annotations

import struct
import zipfile
from pathlib import Path


# --------------------------------------------------------------------------- binary builders

def _prop(name: str, dtype: str, vals: list) -> bytes:
    out = b"!" + bytes([len(name)]) + name.encode() + dtype.encode() + struct.pack("<i", len(vals))
    if dtype == "i":
        return out + struct.pack(f"<{len(vals)}i", *vals)
    if dtype == "f":
        return out + struct.pack(f"<{len(vals)}f", *vals)
    for s in vals:
        raw = s.encode() + b"\0"
        out += struct.pack("<i", len(raw)) + raw
    return out


def _obj(depth: int, name: str) -> bytes:
    return b"[" * depth + name.encode() + b"\0"


def make_mesh(shader: str, diff: str, normal: str, spec: str, bones=(), locators=None) -> bytes:
    out = b"@@b@" + _prop("pdxasset", "i", [1, 0]) + _obj(1, "object") + _obj(2, "bodyShape")
    out += _obj(3, "mesh") + _prop("p", "f", [0.0, 1.0, 2.0])
    out += _obj(4, "material") + _prop("shader", "s", [shader]) + _prop("diff", "s", [diff])
    out += _prop("n", "s", [normal]) + _prop("spec", "s", [spec])
    if bones:
        out += _obj(3, "skeleton")
        for i, bone in enumerate(bones):
            out += _obj(4, bone) + _prop("ix", "i", [i])
    if locators:
        out += _obj(1, "locator")
        for name, parent in locators.items():
            out += _obj(2, name) + _prop("p", "f", [0.0, 0.0, 0.0])
            if parent:
                out += _prop("pa", "s", [parent])
    return out


def make_dds(width: int, height: int, mips: int, fourcc: bytes = b"DXT1") -> bytes:
    hdr = bytearray(128)
    hdr[0:4] = b"DDS "
    struct.pack_into("<I", hdr, 4, 124)
    struct.pack_into("<2I", hdr, 12, height, width)
    struct.pack_into("<I", hdr, 28, mips)
    struct.pack_into("<I", hdr, 80, 0x4)
    hdr[84:88] = fourcc
    if fourcc == b"DX10":
        hdr += struct.pack("<I", 71) + bytes(16)
    return bytes(hdr)


# --------------------------------------------------------------------------- script text

INF_BONES = ["root", "Left_hand_node", "Right_hand_node", "attach_back"]
HORSE_BONES = ["Root", "Saddle_Node"]
STATES4 = "\n".join(f'\tstate = {{ name = "{s}" animation = "{s}" }}' for s in ("idle", "walk", "attack", "retreat"))
HORSE_STATES = ('\tstate = { name = "idle" animation = "horse_idle" }\n\tstate = { name = "walk" animation = "horse_walk" }\n'
                '\tstate = { name = "attack" animation = "horse_attack" }\n\tstate = { name = "retreat" animation = "horse_retreat" }')
HORSE_ROOT_STATES = HORSE_STATES.replace('animation = "horse_retreat"', 'animation = "horse_walk"')


def _western_units() -> str:
    parts = []
    for t in range(1, 5):
        node, weapon = ("Right_hand_node", "w_halbard") if t == 1 else ("Left_hand_node", "w_musket")
        parts.append(f'entity = {{\n\tname = "westerngfx_INFANTRY_{t}"\n\tpdxmesh = "generic_0{t}"\n{STATES4}\n'
                     f'\t#state = {{ name = "death" animation = "death" }}\n\tattach = {{ {node} = "{weapon}" }}\n}}')
    parts.append(f'entity = {{\n\tname = "westerngfx_CAVALRY_1_rider"\n\tpdxmesh = "generic_01"\n{HORSE_STATES}\n'
                 f'\tattach = {{ Right_hand_node = "w_saber" }}\n}}')
    for t in range(2, 5):
        parts.append(f'entity = {{ clone = "westerngfx_CAVALRY_1_rider" name = "westerngfx_CAVALRY_{t}_rider" pdxmesh = "generic_0{t}" }}')
    parts.append(f'entity = {{\n\tname = "westerngfx_CAVALRY_1"\n\tpdxmesh = "horse"\n{HORSE_ROOT_STATES}\n'
                 f'\tattach = {{ name = "rider" Saddle_Node = "westerngfx_CAVALRY_1_rider" }}\n}}')
    parts.append('entity = { clone = "westerngfx_CAVALRY_1" name = "westerngfx_CAVALRY_2" '
                 'attach = { name = "rider" Saddle_Node = "westerngfx_CAVALRY_2_rider" } }')
    for t in (3, 4):
        parts.append(f'entity = {{ clone = "westerngfx_CAVALRY_1" name = "westerngfx_CAVALRY_{t}" pdxmesh = "horse_3" '
                     f'attach = {{ name = "rider" Saddle_Node = "westerngfx_CAVALRY_{t}_rider" }} }}')
    parts.append('entity = {\n\tname = "westerngfx_ARTILLERY_1"\n\tpdxmesh = "artillery_base"\n'
                 '\tstate = { name = "idle" animation = "artillery_base_idle" }\n'
                 '\tstate = { name = "walk" animation = "artillery_base_moving" }\n'
                 '\tstate = { name = "attack" animation = "artillery_base_attack" }\n'
                 '\tstate = { name = "retreat" animation = "artillery_base_moving" }\n'
                 '\tattach = { name = "gunman" infantry_node = "westerngfx_INFANTRY_1" cannon_node = "westerngfx_ARTILLERY_1_cannon" }\n}')
    for t in range(2, 5):
        parts.append(f'entity = {{ clone = "westerngfx_ARTILLERY_1" name = "westerngfx_ARTILLERY_{t}_generic" }}')
        parts.append(f'entity = {{ clone = "westerngfx_ARTILLERY_{t}_generic" name = "westerngfx_ARTILLERY_{t}" '
                     f'attach = {{ name = "gunman" infantry_node = "westerngfx_INFANTRY_{t}" cannon_node = "westerngfx_ARTILLERY_{t}_cannon" }} }}')
    for t in range(1, 5):
        parts.append(f'entity = {{\n\tname = "westerngfx_ARTILLERY_{t}_cannon"\n\tpdxmesh = "cannon"\n'
                     '\tstate = { name = "attack" animation = "artillery_attack" event = { time = 0.0 node = "cannon_muzzle" particle = "p_cannon" } }\n'
                     '\tstate = { name = "idle" animation = "artillery_idle" }\n}')
    # A base-game tag set, used to check pack-order shadowing and collisions.
    parts.append('entity = { clone = "westerngfx_INFANTRY_1" name = "TST_INFANTRY_1" }')
    return "\n\n".join(parts) + "\n"


WEAPONS = """
entity = { name = "w_halbard" pdxmesh = "halbard" }
entity = { name = "w_saber" pdxmesh = "halbard" }
entity = {
\tname = "w_musket"
\tpdxmesh = "musket"
\tstate = { name = "attack" event = { time = 0.90 node = "muzzle" particle = "p_musket" } }
}
"""

INF_ANIMS = " ".join(f'animation = {{ id = "{i}" type = "{t}" }}' for i, t in [
    ("idle", "inf_idle_animation"), ("walk", "inf_walk_animation"), ("attack", "inf_attack_animation"),
    ("death", "inf_idle_animation"), ("retreat", "inf_walk_animation"),
    ("horse_idle", "rider_idle_animation"), ("horse_walk", "rider_idle_animation"),
    ("horse_attack", "rider_idle_animation"), ("horse_death", "rider_idle_animation"),
    ("horse_retreat", "rider_idle_animation"), ("artillery_attack", "inf_attack_animation")])
HORSE_ANIMS = " ".join(f'animation = {{ id = "{i}" type = "horse_idle_animation" }}'
                       for i in ("horse_idle", "horse_walk", "horse_attack", "horse_retreat"))


def _gfx() -> str:
    lines = [f'\tpdxmesh = {{ name = "generic_0{t}" file = "gfx/models/Units/generic_0{t}.mesh" {INF_ANIMS} scale = 0.5 }}'
             for t in range(1, 5)]
    lines += [
        f'\tpdxmesh = {{ name = "horse" file = "gfx/models/Units/horse.mesh" {HORSE_ANIMS} scale = 0.48 }}',
        f'\tpdxmesh = {{ name = "horse_3" file = "gfx/models/Units/horse_3.mesh" {HORSE_ANIMS} scale = 0.48 }}',
        '\tpdxmesh = { name = "artillery_base" file = "gfx/models/Units/artillery_base.mesh" '
        'animation = { id = "artillery_base_idle" type = "base_idle_animation" } '
        'animation = { id = "artillery_base_moving" type = "base_idle_animation" } '
        'animation = { id = "artillery_base_attack" type = "base_idle_animation" } scale = 1.0 }',
        '\tpdxmesh = { name = "cannon" file = "gfx/models/Units/cannon.mesh" '
        'animation = { id = "artillery_attack" type = "cannon_attack_animation" } '
        'animation = { id = "artillery_idle" type = "cannon_attack_animation" } scale = 1.1 }',
        '\tpdxmesh = { name = "halbard" file = "gfx/models/Units/halbard.mesh" scale = 0.5 }',
        '\tpdxmesh = { name = "musket" file = "gfx/models/Units/musket.mesh" scale = 0.5 }',
    ]
    return "objectTypes = {\n" + "\n".join(lines) + "\n}\n"


ANIMS = "\n".join(f'animation = {{ name = "{n}_animation" file = "{n}.anim" }}' for n in (
    "inf_idle", "inf_walk", "inf_attack", "rider_idle", "horse_idle", "base_idle", "cannon_attack"))

PACKS = """
westerngfx_sprite_pack = {
\tcountry_trigger = { always = no }
\tgraphical_cultures = { westerngfx }
\tsprite_infantry = { 0 = westerngfx_INFANTRY_1 1 = westerngfx_INFANTRY_2 2 = westerngfx_INFANTRY_3 3 = westerngfx_INFANTRY_4 }
\tsprite_cavalry = { 0 = westerngfx_CAVALRY_1 1 = westerngfx_CAVALRY_2 2 = westerngfx_CAVALRY_3 3 = westerngfx_CAVALRY_4 }
\tsprite_artillery = { 0 = westerngfx_ARTILLERY_1 1 = westerngfx_ARTILLERY_2 2 = westerngfx_ARTILLERY_3 3 = westerngfx_ARTILLERY_4 }
}
"""

MIL = "monarch_power = MIL\n\n" + "\n".join(
    "technology = {\n\t# Tech %d\n\tyear = %d\n%s%s}\n" % (
        n, 1350 + 10 * n,
        {0: "\tsprite_level = 1\n", 3: "\tsprite_level = 2\n", 5: "\tsprite_level = 3\n", 6: "\tsprite_level = 4\n"}.get(n, ""),
        {1: "\tenable = w_medieval_infantry\n", 2: "\tenable = w_cannon\n"}.get(n, ""))
    for n in range(8))

GRAPHICAL_CULTURES = "westerngfx\neasterngfx\n"

DLC_ENTITIES = 'entity = { name = "TST_INFANTRY_1" clone = "westerngfx_INFANTRY_2" version = 4 }\n'
DLC_PACK = "dlc999_tst_sprite_pack = {\n\tcountry_trigger = { primary_culture = testish }\n\tsprite_infantry = { 0 = TST_INFANTRY_1 }\n}\n"
DLC_ORDER = "dlc999_tst_sprite_pack = 100\n"
DLC_GROUP_PACK = ("dlc999_grp_sprite_pack = {\n\tcountry_trigger = { OR = { culture_group = other_group tag = ZZZ } }\n"
                  "\tsprite_infantry = { 0 = westerngfx_INFANTRY_1 }\n}\n")

# 1444 history: AAA is the only tag nothing covers. TST: culture pack + TST_* entity.
# BBB: culture-group pack. CCC: owns land only before and after 1444. DDD: no land.
COUNTRY_TAGS = ('TST = "countries/Testland.txt"\nAAA = "countries/Alpha.txt"\n'
                'BBB = "countries/Beta.txt"\nCCC = "countries/Gamma.txt"\n')
COUNTRY = ("#Country file\ngraphical_culture = westerngfx\n\ncolor = {{ {c} }}\n\nhistorical_units = {{\n"
           "\tw_medieval_infantry\n}}\n\nmonarch_names = {{\n\t\"Alpha #0\" = 10\n}}\n")
CULTURES = ("test_group = {\n\tgraphical_culture = westerngfx\n\tmale_names = { A }\n"
            "\ttestish = { primary = TST male_names = { B } }\n\talphan = { male_names = { C } }\n}\n"
            "other_group = {\n\tbetan = { male_names = { D } }\n}\n")
HISTORY = {
    "TST - Testland.txt": "primary_culture = testish\ntechnology_group = western\n",
    "AAA - Alpha.txt": ("government = monarchy\nadd_government_reform = feudalism_reform\nprimary_culture = betan\n"
                        "technology_group = western\n1400.1.1 = { primary_culture = alphan }\n"
                        "1500.1.1 = { primary_culture = betan }\n"),
    "BBB - Beta.txt": "primary_culture = betan\ntechnology_group = eastern\n",
    "CCC - Gamma.txt": "primary_culture = alphan\n",
}
PROVINCES = {
    "1 - One.txt": "owner = TST\ncontroller = TST\n",
    "2 - Two.txt": "owner = AAA\n1450.1.1 = { owner = TST }\n",
    "3 - Three.txt": "1400.1.1 = { owner = CCC }\n1440.1.1 = { owner = BBB }\n",
    "4 - Four.txt": "1500.1.1 = { owner = CCC }\n",
}


def _write(root: Path, rel: str, data: str | bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode("latin-1") if isinstance(data, str) else data)


def build_fake_install(root: Path) -> Path:
    _write(root, "eu4_branch.txt", "release_9.99\n")
    _write(root, "eu4_rev.txt", "fake")
    _write(root, "common/graphicalculturetype.txt", GRAPHICAL_CULTURES)
    _write(root, "common/technologies/mil.txt", MIL)
    _write(root, "common/units/w_medieval_infantry.txt", "type = infantry\nunit_type = western\nmaneuver = 1\n")
    _write(root, "gfx/entities/western_units.asset", _western_units())
    _write(root, "gfx/entities/weapons.asset", WEAPONS)
    _write(root, "interface/assets/units.gfx", _gfx())
    _write(root, "gfx/models/Units/animation/animation.asset", ANIMS)
    for n in ("inf_idle", "inf_walk", "inf_attack", "rider_idle", "horse_idle", "base_idle", "cannon_attack"):
        _write(root, f"gfx/models/Units/animation/{n}.anim", b"")
    _write(root, "gfx/sprite_packs/+00_base_sprites.txt", PACKS)
    _write(root, "gfx/sprite_packs/+01_tst.txt",
           "tst_base_sprite_pack = {\n\tcountry_trigger = { tag = TST }\n\tsprite_infantry = { 0 = TST_INFANTRY_1 }\n}\n")
    _write(root, "gfx/sprite_packs_order/00_sprite_pack_orders.txt", "# higher wins\n")
    _write(root, "common/country_tags/00_countries.txt", COUNTRY_TAGS)
    for name, c in (("Testland", "1 2 3"), ("Alpha", "10 20 30"), ("Beta", "4 5 6"), ("Gamma", "7 8 9")):
        _write(root, f"common/countries/{name}.txt", COUNTRY.format(c=c))
    _write(root, "common/cultures/00_cultures.txt", CULTURES)
    for name, text in HISTORY.items():
        _write(root, f"history/countries/{name}", text)
    for name, text in PROVINCES.items():
        _write(root, f"history/provinces/{name}", text)

    units = "gfx/models/Units"
    for t in range(1, 5):
        _write(root, f"{units}/generic_0{t}.mesh", make_mesh("PdxMeshColor", f"g{t}_d.dds", "g_n.dds", "g_s.dds", INF_BONES))
    for name in ("horse", "horse_3"):
        _write(root, f"{units}/{name}.mesh", make_mesh("PdxMeshColor", "h_d.dds", "h_n.dds", "h_s.dds", HORSE_BONES))
    _write(root, f"{units}/artillery_base.mesh", make_mesh("PdxMeshStandard", "nospec.dds", "nonormal.dds", "nospec.dds",
                                                            ["root", "infantry_node", "cannon_node"]))
    _write(root, f"{units}/cannon.mesh", make_mesh("PdxMeshStandard", "c_d.dds", "c_n.dds", "c_s.dds",
                                                    ["Root", "Cannon"], {"cannon_muzzle": "Cannon"}))
    _write(root, f"{units}/halbard.mesh", make_mesh("PdxMeshStandard", "w_d.dds", "w_n.dds", "w_s.dds"))
    _write(root, f"{units}/musket.mesh", make_mesh("PdxMeshStandard", "w_d.dds", "w_n.dds", "w_s.dds", (),
                                                    {"muzzle": None, "trigger": None}))
    for tex in ("g1_d", "g2_d", "g3_d", "g4_d", "g_n", "h_d", "h_n", "c_d", "c_n", "w_d", "w_n"):
        _write(root, f"{units}/{tex}.dds", make_dds(512, 512, 10, b"DXT1"))
    for tex in ("g_s", "h_s", "c_s", "w_s", "nospec", "nonormal"):
        _write(root, f"{units}/{tex}.dds", make_dds(512, 512, 10, b"DXT5"))

    zpath = root / "dlc" / "dlc999_test" / "dlc999.zip"
    zpath.parent.mkdir(parents=True)
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("gfx/entities/dlc_tst_units.asset", DLC_ENTITIES)
        zf.writestr("gfx/sprite_packs/999_sprites.txt", DLC_PACK + DLC_GROUP_PACK)
        zf.writestr("gfx/sprite_packs_order/999_order.txt", DLC_ORDER)
    return root


# --------------------------------------------------------------------------- decodable textures (Phase 2)

def synthetic_diffuse(size: int = 64):
    """Left half skin-toned, right half neutral grey (made up)."""
    from PIL import Image
    im = Image.new("RGBA", (size, size), (128, 128, 128, 255))
    im.paste((210, 150, 120, 255), (0, 0, size // 2, size))
    return im


def synthetic_spec(size: int = 64):
    """Right half: top quarter-band R (primary), lower band G (secondary); a gloss 40 'metal'
    strip across the top rows of the grey half; everything else 0 (made up)."""
    from PIL import Image
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    h = size // 2
    im.paste((255, 0, 0, 2), (h, 8, size, h))
    im.paste((0, 255, 0, 2), (h, h, size, size))
    im.paste((0, 0, 0, 40), (h, 0, size, 8))
    return im


def make_textures_decodable(root: Path) -> Path:
    """Replace the header-only DDS files the retexture bases use with real (tiny) ones, and add
    the shader file meshsettings names."""
    from eu4unitgen.textures.dds import DDSFormat, write_dds
    units = root / "gfx/models/Units"
    for tex in ("g1_d", "g2_d", "g3_d", "g4_d", "h_d", "c_d"):
        (units / f"{tex}.dds").write_bytes(write_dds(synthetic_diffuse(), DDSFormat("DXT1", 64, 64, 7)))
    for tex in ("g_s", "h_s", "c_s"):
        (units / f"{tex}.dds").write_bytes(write_dds(synthetic_spec(), DDSFormat("DXT5", 64, 64, 7)))
    (root / "gfx/FX").mkdir(parents=True, exist_ok=True)
    (root / "gfx/FX/pdxmesh.shader").write_bytes(b"# made-up stand-in\n")
    return root
