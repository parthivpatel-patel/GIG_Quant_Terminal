"""Normalize caller-authored HTML into cited visible-text passages.

Each passage is one block element (paragraph, heading, list item) or one table
row. Its citation is the exact byte range of that block in the original HTML,
and re-normalizing those bytes must reproduce the passage text. Scripts, styles
and markup never become text; nothing in a document is executed or fetched.
"""
from __future__ import annotations

import hashlib
import re
from html.parser import HTMLParser

NORMALIZATION = "HTML_VISIBLE_BLOCK_TEXT_V1"
MAX_BYTES, MAX_PASSAGES = 262144, 512
BLOCKS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "caption", "blockquote", "pre", "dt", "dd"}
CELLS = {"td", "th"}
HIDDEN = {"script", "style", "template", "noscript", "head"}
BREAKERS = {"div", "section", "article", "table", "thead", "tbody", "tfoot", "ul", "ol", "dl", "body", "html", "header",
            "footer", "main", "nav", "aside", "hr", "form", "figure"}
CELL_SEPARATOR = " | "


class _Blocks(HTMLParser):
    """Collect (text, start_char, end_char) for each visible block, in document order."""

    def __init__(self, text: str):
        super().__init__(convert_charrefs=True)
        self.text = text
        self.lines = [0]+[match.end() for match in re.finditer("\n", text)]
        self.blocks: list[tuple[str, int, int]] = []
        self.hidden = 0
        self.kind: str | None = None  # open block tag; "" is bare text outside any block
        self.cells: list[list[str]] = []
        self.start = 0

    def _offset(self) -> int:
        line, column = self.getpos()
        return self.lines[line-1]+column

    def _close(self, end: int):
        if self.kind is not None:
            cells = [" ".join("".join(cell).split()) for cell in self.cells]
            body = CELL_SEPARATOR.join(cells).strip() if len(cells) > 1 else cells[0]
            if body.replace("|", "").strip():
                self.blocks.append((body, self.start, end))
        self.kind = None

    def _open(self, kind: str):
        self._close(self._offset())
        self.kind, self.cells, self.start = kind, [[]], self._offset()

    def handle_starttag(self, tag, attrs):
        if tag in HIDDEN:
            self.hidden += 1
        elif self.hidden:
            return
        elif self.kind == "tr" and tag != "tr":
            if tag in CELLS:
                # The first cell reuses the row's initial buffer only when nothing preceded it.
                if self.cells != [[]]:
                    self.cells.append([])
                self.cells[-1].append("")
            elif tag == "br":
                self.cells[-1].append(" ")
        elif tag in BLOCKS:
            self._open(tag)
        elif tag in BREAKERS:
            self._close(self._offset())
        elif tag == "br" and self.kind is not None:
            self.cells[-1].append(" ")

    def handle_endtag(self, tag):
        if tag in HIDDEN:
            self.hidden = max(0, self.hidden-1)
        elif self.hidden:
            return
        elif self.kind == tag and tag in BLOCKS:
            self._close(self.text.index(">", self._offset())+1)
        elif tag in BREAKERS or (tag == "table" and self.kind == "tr"):
            self._close(self._offset())

    def handle_data(self, data):
        if self.hidden:
            return
        if self.kind is None:
            if not data.strip():
                return
            self._open("")
        self.cells[-1].append(data)


def visible_blocks(text: str) -> list[tuple[str, int, int]]:
    parser = _Blocks(text)
    parser.feed(text)
    parser.close()
    parser._close(len(text))
    return parser.blocks


def html_document(payload):
    """Return (raw, metadata, passages) for one caller-authored HTML document."""
    if (not isinstance(payload, dict) or set(payload) != {"source_id", "title", "html", "license"}
            or not isinstance(payload["source_id"], str) or not re.fullmatch(r"[a-z0-9_-]{1,80}", payload["source_id"])
            or not isinstance(payload["title"], str) or not 1 <= len(payload["title"]) <= 256
            or not isinstance(payload["html"], str) or not payload["html"].strip()
            or payload["license"] != "GIG-AUTHORED"):
        raise ValueError("explicit bounded authored HTML and authorship attestation required")
    text = payload["html"]
    raw = text.encode("utf-8")
    if len(raw) > MAX_BYTES:
        raise ValueError("authored HTML budget exceeded")
    # Character offsets become byte offsets through one cumulative pass.
    width = [0]
    for character in text:
        width.append(width[-1]+len(character.encode("utf-8")))
    passages = []
    for body, start, end in visible_blocks(text):
        end = min(max(end, start+1), len(text))
        passage = {"text": body, "source_start": width[start], "source_end": width[end], "observation_date": None}
        if renormalize(raw[passage["source_start"]:passage["source_end"]]) != body:
            raise ValueError("HTML block citation does not reproduce its normalized text")
        passages.append(passage)
    if not 1 <= len(passages) <= MAX_PASSAGES:
        raise ValueError("authored HTML requires 1..512 visible text blocks")
    return raw, {"namespace": "CALLER_AUTHORED", "title": payload["title"],
                 "source_url": "urn:gig:authored-html:"+payload["source_id"], "license": "GIG-AUTHORED",
                 "license_note": "Caller attestation, not independently verified third-party rights",
                 "source_acquired_at": None, "publication_time": None, "publication_state": "UNKNOWN",
                 "classification": "SYNTHETIC", "normalization": NORMALIZATION,
                 "raw_sha256": hashlib.sha256(raw).hexdigest()}, passages


def renormalize(excerpt: bytes) -> str | None:
    """Visible text of one cited byte range; None unless it is exactly one block."""
    blocks = visible_blocks(excerpt.decode("utf-8"))
    return blocks[0][0] if len(blocks) == 1 else None
