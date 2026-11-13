"""Normalize the already-permitted Treasury archive with original byte citations."""
from __future__ import annotations

import hashlib
import json
import re
from importlib.resources import files

from gig.data.ingestion import parse_treasury

METADATA_SHA256 = "61c4de1c76a561b61da0fa26a47b5fb5b205ee77e7f1819c8b77e9d7a4ed4042"
SHA256 = "99f80b80cfeff0a93e339735d72672080d9209073b5fe814c1252e88d1428227"


def treasury_document():
    package = files("gig.research_org").joinpath("fixtures")
    raw = package.joinpath("treasury_par_yield_2023.xml").read_bytes()
    metadata_raw = package.joinpath("treasury_par_yield_2023.json").read_bytes()
    if hashlib.sha256(metadata_raw).hexdigest() != METADATA_SHA256:
        raise ValueError("Treasury reference permission/provenance integrity failed")
    metadata = json.loads(metadata_raw)
    if hashlib.sha256(raw).hexdigest() != SHA256 or metadata["sha256"] != SHA256:
        raise ValueError("Treasury reference source integrity failed")
    if metadata["license"] != "http://creativecommons.org/publicdomain/zero/1.0/":
        raise ValueError("Treasury permission metadata changed")
    # Reuse the canonical admitted-source parser for numeric/date validation.
    rows = parse_treasury(raw, metadata["retrieved_at"])
    by_date = {}
    for row in rows:
        by_date.setdefault(row["event_date"], []).append(row)
    passages = []
    for match in re.finditer(rb"<entry>.*?</entry>", raw, re.S):
        entry = match.group()
        stamp = re.search(rb"<d:NEW_DATE[^>]*>([^<]+)</d:NEW_DATE>", entry)
        if stamp is None:
            raise ValueError("Treasury citation date absent")
        day = stamp[1].decode().split("T")[0]
        rates = by_date.pop(day)
        # This is explicitly a GIG normalization of financial observations, not
        # publisher-authored prose or tradeable prices. Values stay Decimal text.
        text = "U.S. Treasury daily par yield curve. Observation date "+day+". "+"; ".join(
            row["tenor"]+" yield "+row["yield_percent"]+" percent" for row in rates)+". Original publication time UNKNOWN; these are reference yields, not transaction prices."
        passages.append({"text": text, "source_start": match.start(), "source_end": match.end(),
                         "observation_date": day})
    if by_date or not 200 <= len(passages) <= 300:
        raise ValueError("Treasury normalized passage coverage failed")
    return raw, {"namespace": "OFFICIAL_TREASURY_REFERENCE", "title": "U.S. Treasury 2023 daily par yield archive",
                 "source_url": metadata["source_url"], "license": "CC0-1.0", "license_evidence_url": metadata["catalog_url"],
                 "license_note": metadata["license_note"], "source_acquired_at": metadata["retrieved_at"],
                 "publication_time": None, "publication_state": "UNKNOWN", "classification": "HISTORICAL",
                 "normalization": "GIG_XML_RECORD_RENDERING_NOT_PUBLISHER_PROSE", "raw_sha256": SHA256}, passages
