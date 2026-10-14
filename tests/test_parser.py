"""Tests for pdxscript/parser.py.

Synthetic snippets cover the grammar and edits. The `vanilla` tests read the user's
install in place (EU4_PATH) and check the lossless round-trip on every script file,
plus a canonical-writer round-trip on the unit-related folders.
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pytest

from eu4unitgen.pdxscript.parser import (
    Block, Pair, Scalar, block, compact, format_entries, pair, parse, parse_bytes,
)

SAMPLE = """# header comment
entity = {
\tname = "DAN_INFANTRY_1"   # trailing
\tpdxmesh = "DAN_INFANTRY_1"
\t#state = { name = "death" animation = "death" }
\tstate = { name = "idle"\t\tanimation = "idle" }
\tattach = { Right_hand_node = "western_halbard" }
}

westerngfx_sprite_pack = {
\tcountry_trigger = { always = no }
\tgraphical_cultures = { westerngfx }
\tsprite_infantry = { 0 = westerngfx_INFANTRY_1 1 = westerngfx_INFANTRY_2 }
}
check = { mil_tech >= 12 ADM != 3 a == b c ?= d e < 1 f > 2 g <= 3 }
"""


# --------------------------------------------------------------------------- grammar

def test_lossless_round_trip_of_sample():
    assert parse(SAMPLE).dumps() == SAMPLE


def test_structure_and_queries():
    doc = parse(SAMPLE)
    ent = doc.get_block("entity")
    assert ent.get_str("name") == "DAN_INFANTRY_1"
    assert [s.get_str("name") for s in ent.get_all("state")] == ["idle"]  # comment is not a state
    assert ent.get_block("attach").scalar_map() == {"Right_hand_node": "western_halbard"}
    pack = doc.get_block("westerngfx_sprite_pack")
    assert pack.get_block("graphical_cultures").bare() == ["westerngfx"]
    assert pack.get_block("sprite_infantry").scalar_map() == {"0": "westerngfx_INFANTRY_1", "1": "westerngfx_INFANTRY_2"}
    assert compact(pack.get_block("country_trigger")) == "always = no"


def test_all_operators():
    ops = [p.op for p in parse(SAMPLE).get_block("check").pairs()]
    assert ops == [">=", "!=", "==", "?=", "<", ">", "<="]


def test_quoted_flag_kept():
    ent = parse(SAMPLE).get_block("entity")
    name = ent.get("name")
    assert isinstance(name, Scalar) and name.quoted and name.value == "DAN_INFANTRY_1"


def test_hyphenated_and_symbol_words():
    doc = parse("dlc085_cnn_2-4_sprite_pack = 50 a = -0.5 b = @var c = 1.0.0")
    assert doc.scalar_map() == {"dlc085_cnn_2-4_sprite_pack": "50", "a": "-0.5", "b": "@var", "c": "1.0.0"}


def test_no_spaces_around_operator():
    assert parse("a=b c={d=e}").get_block("c").scalar_map() == {"d": "e"}
    assert parse("a!=b").pairs().__next__().op == "!="


def test_escaped_quote_inside_string():
    text = 'desc = "say \\"hi\\""'
    doc = parse(text)
    assert doc.get_str("desc") == 'say \\"hi\\"' and doc.dumps() == text


def test_hash_inside_string_is_not_a_comment():
    doc = parse('a = "x # y" b = 1')
    assert doc.scalar_map() == {"a": "x # y", "b": "1"}


def test_bare_nested_blocks():
    doc = parse("color = { { 1 2 } { 3 4 } }")
    inner = doc.get_block("color").entries
    assert all(isinstance(b, Block) for b in inner) and inner[1].bare() == ["3", "4"]


@pytest.mark.parametrize("text, warning", [
    ("a = 1 } b = 2", "unmatched"),
    ("a = { b = 1", "end of file"),
    ("a = }", "no value"),
    ("= 3", "no key"),
    ('a = "unterminated', None),
])
def test_malformed_input_round_trips_and_warns(text, warning):
    doc = parse(text)
    assert doc.dumps() == text
    if warning:
        assert any(warning in w for w in doc.warnings)


def test_stray_brace_keeps_following_entries():
    doc = parse("a = 1 } b = 2")
    assert doc.scalar_map() == {"a": "1", "b": "2"}


def test_semantic_equality_ignores_layout_and_comments():
    assert parse("a = { b = 1 }") == parse("# c\na={\n\tb=1 # x\n}\n")
    assert parse("a = { b = 1 }") != parse("a = { b = 2 }")
    assert parse('a = "x"') != parse("a = x")


# --------------------------------------------------------------------------- bytes + encodings

def test_bom_utf8_round_trip():
    data = b"\xef\xbb\xbfl_english:\n hr_x:0 \"Caf\xc3\xa9\"\n"
    doc = parse_bytes(data)
    assert doc.bom and doc.to_bytes() == data


def test_windows_1252_bytes_round_trip():
    data = b'name = "M\xfcnchen"\r\nx = \x93q\x94\r\n'  # cp1252 umlaut + smart quotes, CRLF
    doc = parse_bytes(data)
    assert doc.to_bytes() == data and doc.get_str("name") == "München"


# --------------------------------------------------------------------------- edits + writer

def test_edit_keeps_untouched_text():
    src = "technology = {\n\t# Tech 1\n\tyear = 1390\n\n\tenable = western_medieval_infantry\n}\n\ntechnology = {\n\tyear = 1420\n}\n"
    doc = parse(src)
    tech1 = doc.get_all("technology")[0]
    anchor = list(tech1.pairs("enable"))[-1]
    tech1.insert_after(anchor, pair("enable", "hr_hellenic_phalanx"))
    assert doc.dumps() == src.replace(
        "enable = western_medieval_infantry\n", "enable = western_medieval_infantry\n\tenable = hr_hellenic_phalanx\n")


def test_append_to_parsed_block_and_document():
    doc = parse("a = {\n\tb = 1\n}\n")
    doc.get_block("a").append(pair("c", block(pair("d", "x"))))
    doc.append(pair("e", "f"))
    out = doc.dumps()
    assert parse(out) == parse("a = { b = 1 c = { d = x } } e = f")
    assert "\tc = { d = x }" in out


def test_added_lines_follow_crlf_source():
    doc = parse("westerngfx\r\neasterngfx\r\n")
    doc.append(Scalar("hellenicgfx"))
    assert doc.dumps() == "westerngfx\r\neasterngfx\r\nhellenicgfx\r\n"


def test_remove_entry():
    doc = parse("a = 1\nb = 2\n")
    doc.remove(next(doc.pairs("a")))
    assert doc.scalar_map() == {"b": "2"}


def test_canonical_writer():
    doc = parse(SAMPLE)
    out = format_entries([next(doc.pairs("entity"))])
    assert out == (
        "entity = {\n"
        '\tname = "DAN_INFANTRY_1"\n'
        '\tpdxmesh = "DAN_INFANTRY_1"\n'
        '\tstate = { name = "idle" animation = "idle" }\n'
        '\tattach = { Right_hand_node = "western_halbard" }\n'
        "}\n")


def test_canonical_writer_reparses_equal():
    doc = parse(SAMPLE)
    assert parse(format_entries(doc.entries)) == doc


def test_built_nodes_write_valid_script():
    ent = pair("entity", block(pair("name", "MAC_INFANTRY_1", quoted=True), pair("version", "20")))
    text = format_entries([ent])
    assert text == 'entity = { name = "MAC_INFANTRY_1" version = 20 }\n'
    assert isinstance(next(parse(text).pairs()), Pair)


# --------------------------------------------------------------------------- vanilla corpus

EU4 = os.environ.get("EU4_PATH")
needs_eu4 = pytest.mark.skipif(not EU4, reason="EU4_PATH not set")
SCRIPT_EXTS = {".txt", ".asset", ".gfx", ".gui"}
NOT_SCRIPT = {"localisation", "patchnotes", "licenses"}  # yml and prose
UNIT_DIRS = ("gfx/entities/", "interface/assets/", "gfx/sprite_packs/", "gfx/sprite_packs_order/",
             "common/units/", "common/technologies/", "common/countries/", "gfx/models/units/animation/")


def _vanilla_scripts():
    root = Path(EU4)
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.suffix.lower() in SCRIPT_EXTS and p.is_file() and not NOT_SCRIPT & set(rel.parts):
            yield rel.as_posix(), p.read_bytes()
    for z in sorted((root / "dlc").glob("*/*.zip")):
        with zipfile.ZipFile(z) as zf:
            for m in sorted(zf.namelist()):
                if Path(m).suffix.lower() in SCRIPT_EXTS:
                    yield f"{z.parent.name}:{m}", zf.read(m)


@pytest.mark.vanilla
@needs_eu4
def test_vanilla_lossless_round_trip():
    failures, count = [], 0
    for name, data in _vanilla_scripts():
        count += 1
        if parse_bytes(data).to_bytes() != data:
            failures.append(name)
    assert count > 5000, f"only {count} script files found under EU4_PATH"
    assert failures == []


@pytest.mark.vanilla
@needs_eu4
def test_vanilla_canonical_round_trip_unit_files():
    count = 0
    for name, data in _vanilla_scripts():
        rel = name.split(":", 1)[-1].lower()
        if not rel.startswith(UNIT_DIRS):
            continue
        count += 1
        doc = parse_bytes(data)
        assert parse(format_entries(doc.entries)) == doc, name
    assert count > 300
