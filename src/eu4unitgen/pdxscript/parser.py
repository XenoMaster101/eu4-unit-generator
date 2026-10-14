"""Clausewitz (Paradox script) parser and writer.

Two guarantees:

* **Lossless round-trip.** ``parse(text).dumps() == text`` for any input. Whitespace
  and comments are kept as "trivia" on the token that follows them, so a file can be
  read, edited in a few places, and written back with everything else untouched
  (used for patching ``mil.txt``).
* **Canonical writer.** ``format_entries()`` renders nodes with tab indentation and
  no comments, for nodes built in code.

Grammar (what the vanilla corpus actually uses):

    document := entry*
    entry    := scalar OP value      # key = value, key >= 3, key = { ... }
              | scalar               # bare list item: { westerngfx easterngfx }
              | block                # bare block: { { 1 2 } }
    value    := scalar | block
    block    := '{' entry* '}'
    OP       := = == != < > <= >= ?=

A stray ``}`` with no open block (vanilla has a few) becomes a ``Stray`` entry: it is
written back losslessly and dropped by the canonical writer. ``Document.warnings``
records it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, Union

OPERATORS = ("=", "==", "!=", "<", ">", "<=", ">=", "?=")
BOM = "﻿"

_TOKEN = re.compile(
    r"""
    (?P<trivia>(?:\s+|\#[^\n]*)+)
  | (?P<string>"(?:\\.|[^"\\])*"?)
  | (?P<open>\{)
  | (?P<close>\})
  | (?P<op>==|!=|<=|>=|\?=|[<>=])
  | (?P<word>(?:[^\s={}"\#<>!?]|[!?](?!=))+)
    """,
    re.VERBOSE | re.DOTALL,
)


# --------------------------------------------------------------------------- tokens + nodes

@dataclass
class Token:
    kind: str  # string | open | close | op | word
    text: str
    pre: str = ""  # whitespace and comments before this token
    line: int = 0


@dataclass(eq=False)
class Scalar:
    """A bare word or quoted string. ``value`` is unquoted (escapes kept as written)."""
    value: str
    quoted: bool = False
    tok: Token | None = None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Scalar):
            return (self.value, self.quoted) == (other.value, other.quoted)
        return NotImplemented

    def __repr__(self) -> str:
        return f"Scalar({self.text!r})"

    @property
    def text(self) -> str:
        if self.tok is not None and _scalar(self.tok) == self:
            return self.tok.text  # as written, e.g. an unterminated string at EOF
        return f'"{self.value}"' if self.quoted else self.value


@dataclass(eq=False)
class Block:
    entries: list[Entry] = field(default_factory=list)
    open_tok: Token | None = None
    close_tok: Token | None = None  # None while unterminated (EOF inside the block)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Block):
            return _semantic(self.entries) == _semantic(other.entries)
        return NotImplemented

    # ---- queries
    def pairs(self, key: str | None = None) -> Iterator[Pair]:
        for e in self.entries:
            if isinstance(e, Pair) and (key is None or e.key.value == key):
                yield e

    def get_all(self, key: str) -> list[Value]:
        """Values of every ``key = ...`` entry, in order (repeated keys are normal)."""
        return [p.value for p in self.pairs(key) if p.value is not None]

    def get(self, key: str, default: Value | None = None) -> Value | None:
        found = self.get_all(key)
        return found[0] if found else default

    def get_str(self, key: str, default: str | None = None) -> str | None:
        """Value of the first ``key = scalar`` as a str; blocks are ignored."""
        for v in self.get_all(key):
            if isinstance(v, Scalar):
                return v.value
        return default

    def get_block(self, key: str) -> Block | None:
        for v in self.get_all(key):
            if isinstance(v, Block):
                return v
        return None

    def bare(self) -> list[str]:
        """Bare list items: ``{ westerngfx easterngfx }`` -> ["westerngfx", "easterngfx"]."""
        return [e.value for e in self.entries if isinstance(e, Scalar)]

    def scalar_map(self) -> dict[str, str]:
        """``key = scalar`` entries as a dict (last one wins), e.g. ``{ 0 = A 1 = B }``."""
        return {p.key.value: p.value.value for p in self.pairs() if isinstance(p.value, Scalar)}

    # ---- edits (new nodes have no tokens and are laid out by the writer)
    def append(self, entry: Entry) -> Entry:
        self.entries.append(entry)
        return entry

    def insert_after(self, anchor: Entry, entry: Entry) -> Entry:
        self.entries.insert(self.entries.index(anchor) + 1, entry)
        return entry

    def remove(self, entry: Entry) -> None:
        self.entries.remove(entry)


@dataclass(eq=False)
class Pair:
    key: Scalar
    op: str = "="
    value: Value | None = None  # None only for malformed input like "a = }"
    op_tok: Token | None = None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Pair):
            return (self.key, self.op, self.value) == (other.key, other.op, other.value)
        return NotImplemented

    def __repr__(self) -> str:
        return f"Pair({self.key.text} {self.op} {self.value!r})"


@dataclass(eq=False)
class Stray:
    """An unmatched ``}``. Kept for lossless output; ignored everywhere else."""
    tok: Token

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Stray)


Value = Union[Scalar, Block]
Entry = Union[Pair, Scalar, Block, Stray]


@dataclass(eq=False)
class Document(Block):
    tail: str = ""  # trivia after the last token
    bom: bool = False
    encoding: str = "utf-8"
    newline: str = "\n"  # used for lines added in code; "\r\n" when the source used CRLF
    warnings: list[str] = field(default_factory=list)

    def dumps(self) -> str:
        """Lossless text: identical to the input for unedited nodes."""
        out: list[str] = [BOM] if self.bom else []
        _emit_entries(self.entries, out, 0, self.newline)
        out.append(self.tail)
        return "".join(out)

    def to_bytes(self) -> bytes:
        text = self.dumps()
        if self.bom:
            return BOM.encode("utf-8") + text[1:].encode("utf-8")
        return text.encode(self.encoding)


def pair(key: str, value: str | Value, op: str = "=", quoted: bool = False) -> Pair:
    """Build a new entry: ``pair("enable", "hr_phalanx")``, ``pair("name", "X", quoted=True)``."""
    if isinstance(value, str):
        value = Scalar(value, quoted)
    return Pair(Scalar(key), op, value)


def block(*entries: Entry) -> Block:
    return Block(list(entries))


# --------------------------------------------------------------------------- parsing

class ParseError(ValueError):
    pass


def tokenize(text: str) -> tuple[list[Token], str]:
    """Return (tokens, trailing trivia). Every character of ``text`` lands in exactly one place."""
    tokens: list[Token] = []
    pre, pos, line = "", 0, 1
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None or m.end() == pos:  # pragma: no cover - the word class accepts any leftover char
            raise ParseError(f"cannot tokenize at line {line}: {text[pos:pos + 20]!r}")
        kind, chunk = m.lastgroup, m.group()
        if kind == "trivia":
            pre += chunk
        else:
            line += pre.count("\n")
            tokens.append(Token(kind, chunk, pre, line))
            line += chunk.count("\n")
            pre = ""
        pos = m.end()
    return tokens, pre


def _scalar(tok: Token) -> Scalar:
    if tok.kind == "string":
        closed = len(tok.text) >= 2 and tok.text.endswith('"')
        return Scalar(tok.text[1:-1] if closed else tok.text[1:], True, tok)
    return Scalar(tok.text, False, tok)


def parse(text: str) -> Document:
    """Parse script text. Never raises on malformed input; problems go to ``warnings``."""
    doc = Document()
    if text.startswith(BOM):
        doc.bom, text = True, text[1:]
    if "\r\n" in text:
        doc.newline = "\r\n"
    tokens, doc.tail = tokenize(text)
    pos = 0

    def entries(depth: int, into: list[Entry]) -> Token | None:
        """Fill ``into`` until the matching close brace; return it (None at EOF)."""
        nonlocal pos
        while pos < len(tokens):
            tok = tokens[pos]
            pos += 1
            if tok.kind == "close":
                if depth > 0:
                    return tok
                doc.warnings.append(f"line {tok.line}: unmatched '}}'")
                into.append(Stray(tok))
            elif tok.kind == "open":
                into.append(_block(tok, depth))
            elif tok.kind == "op":
                doc.warnings.append(f"line {tok.line}: operator {tok.text!r} with no key")
                into.append(Pair(Scalar("", False, None), tok.text, _value(depth), tok))
            elif pos < len(tokens) and tokens[pos].kind == "op":
                op_tok = tokens[pos]
                pos += 1
                into.append(Pair(_scalar(tok), op_tok.text, _value(depth, op_tok), op_tok))
            else:
                into.append(_scalar(tok))
        if depth > 0:
            doc.warnings.append(f"end of file inside a block opened at depth {depth}")
        return None

    def _block(open_tok: Token, depth: int) -> Block:
        blk = Block(open_tok=open_tok)
        blk.close_tok = entries(depth + 1, blk.entries)
        return blk

    def _value(depth: int, op_tok: Token | None = None) -> Value | None:
        nonlocal pos
        if pos >= len(tokens) or tokens[pos].kind in ("close", "op"):
            line = op_tok.line if op_tok else "?"
            doc.warnings.append(f"line {line}: operator with no value")
            return None
        tok = tokens[pos]
        pos += 1
        return _block(tok, depth) if tok.kind == "open" else _scalar(tok)

    entries(0, doc.entries)
    return doc


def decode(data: bytes) -> tuple[str, str]:
    """(text, encoding). UTF-8 when there is a BOM, else latin-1, which maps every byte
    one-to-one so a re-encode gives back the same bytes. Game scripts are Windows-1252;
    identifiers are ASCII either way."""
    if data.startswith(b"\xef\xbb\xbf"):
        return BOM + data[3:].decode("utf-8"), "utf-8"
    return data.decode("latin-1"), "latin-1"


def parse_bytes(data: bytes) -> Document:
    text, enc = decode(data)
    doc = parse(text)
    doc.encoding = enc
    return doc


# --------------------------------------------------------------------------- writing

def _pre(tok: Token | None, default: str) -> str:
    return tok.pre if tok is not None else default


def _nl(depth: int, nl: str = "\n") -> str:
    return nl + "\t" * depth


def _inline(blk: Block) -> bool:
    """Short blocks of scalars stay on one line: ``{ 120 30 140 }``, ``{ tag = MAC }``."""
    if not all(isinstance(e, Scalar) or (isinstance(e, Pair) and isinstance(e.value, Scalar)) for e in blk.entries):
        return False
    return len(blk.entries) <= 8


def _emit_entries(entries: list[Entry], out: list[str], depth: int, nl: str, inline: bool = False) -> None:
    for i, e in enumerate(entries):
        if inline:
            default = " "
        elif depth == 0:
            default = "" if i == 0 else nl
        else:
            default = _nl(depth, nl)
        _emit_entry(e, out, depth, default, nl)


def _emit_entry(e: Entry, out: list[str], depth: int, default_pre: str, nl: str) -> None:
    if isinstance(e, Stray):
        out.append(e.tok.pre + e.tok.text)
    elif isinstance(e, Scalar):
        out.append(_pre(e.tok, default_pre) + e.text)
    elif isinstance(e, Block):
        _emit_block(e, out, depth, default_pre, nl)
    else:
        if e.key.tok is not None or e.op_tok is None:  # parsed "= x" with no key has neither
            out.append(_pre(e.key.tok, default_pre) + e.key.text)
        out.append(_pre(e.op_tok, " ") + e.op)
        if isinstance(e.value, Block):
            _emit_block(e.value, out, depth, " ", nl)
        elif e.value is not None:
            out.append(_pre(e.value.tok, " ") + e.value.text)


def _emit_block(blk: Block, out: list[str], depth: int, default_pre: str, nl: str) -> None:
    out.append(_pre(blk.open_tok, default_pre) + "{")
    inline = blk.open_tok is None and _inline(blk)
    _emit_entries(blk.entries, out, depth + 1, nl, inline)
    if blk.close_tok is not None:
        out.append(blk.close_tok.pre + "}")
    elif blk.open_tok is None:  # built in code
        out.append(" }" if inline else _nl(depth, nl) + "}")
    # else: parsed but unterminated at EOF; nothing to write back


def format_entries(entries: list[Entry], depth: int = 0) -> str:
    """Canonical text for nodes, ignoring original layout and comments."""
    return "".join(_canon_entry(e, depth) for e in entries if not isinstance(e, Stray))


def _canon_entry(e: Entry, depth: int) -> str:
    ind = "\t" * depth
    if isinstance(e, Scalar):
        return f"{ind}{e.text}\n"
    if isinstance(e, Block):
        return f"{ind}{_canon_block(e, depth)}\n"
    val = "" if e.value is None else (_canon_block(e.value, depth) if isinstance(e.value, Block) else e.value.text)
    return f"{ind}{e.key.text} {e.op} {val}\n"


def _canon_block(blk: Block, depth: int) -> str:
    entries = [e for e in blk.entries if not isinstance(e, Stray)]
    if not entries:
        return "{ }"
    if _inline(blk):
        parts = [e.text if isinstance(e, Scalar) else f"{e.key.text} {e.op} {e.value.text}" for e in entries]
        return "{ " + " ".join(parts) + " }"
    return "{\n" + format_entries(entries, depth + 1) + "\t" * depth + "}"


def compact(blk: Block) -> str:
    """One-line rendering of a block's contents: ``tag = DAN``, ``OR = { tag = POL tag = PLC }``."""
    parts = []
    for e in blk.entries:
        if isinstance(e, Stray):
            continue
        if isinstance(e, Scalar):
            parts.append(e.text)
        elif isinstance(e, Block):
            parts.append("{ " + compact(e) + " }")
        else:
            val = "" if e.value is None else ("{ " + compact(e.value) + " }" if isinstance(e.value, Block) else e.value.text)
            parts.append(f"{e.key.text} {e.op} {val}")
    return " ".join(parts)


def _semantic(entries: list[Entry]) -> list:
    return [e for e in entries if not isinstance(e, Stray)]
