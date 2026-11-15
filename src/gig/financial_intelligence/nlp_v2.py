"""Bounded, whole-candidate qualified symbols; source mentions, never identity.

Numeric/currency/tenor/event semantics remain frozen v1 semantics. Only entity
mention extraction changes. Namespace labels confer no exchange authority.
"""
from __future__ import annotations

import re

from gig.financial_intelligence import nlp_v1

IMPLEMENTATION = "gig.financial-mention-rules.v2"
GENERAL_ENTITY = re.compile(r"\b(?:U\.S\. Treasury|Treasury|Federal Reserve)\b")
# Consume an entire non-delimited candidate before validating its syntax. Unicode,
# unsupported suffixes and oversize tokens therefore cannot yield valid prefixes.
# Quotes/brackets and sentence punctuation delimit mentions; a final full stop is
# removed separately because periods also introduce supported share-class syntax.
QUALIFIED_CANDIDATE = re.compile(
    r'''(?<![^\s()\[\]{}<>,;!?"'“”‘’«»])(?P<namespace>NYSE|NASDAQ):[ \t]*'''
    r'''(?P<symbol>[^\s()\[\]{}<>,;!?"'“”‘’«»]+)''')
VALID_SYMBOL = re.compile(r"[A-Z]{1,5}(?:\.[A-Z])?")
MAX_QUALIFIED_CHARS = 32


def extract(passages):
    result = nlp_v1.extract(passages)
    entities = []
    for passage in passages:
        text = passage["text"]
        mentions = [(m.start(), m.end(), {"mention": m.group(),
            "link_state": "UNKNOWN_NO_ISSUER_AUTHORITY",
            "provenance": nlp_v1._evidence(passage, *m.span())})
            for m in GENERAL_ENTITY.finditer(text)]
        for match in QUALIFIED_CANDIDATE.finditer(text):
            symbol = match["symbol"]
            end = match.end()
            if symbol.endswith("."):
                symbol, end = symbol[:-1], end-1
            reason = ("QUALIFIED_SYMBOL_BUDGET_EXCEEDED" if end-match.start() > MAX_QUALIFIED_CHARS
                      else "UNSUPPORTED_QUALIFIED_SYMBOL_SYNTAX" if not VALID_SYMBOL.fullmatch(symbol) else None)
            evidence = nlp_v1._evidence(passage, match.start(), end)
            if reason:
                result["unsupported_mentions"].append({"reason": reason, "provenance": evidence})
                continue
            mentions.append((match.start(), end, {"mention": text[match.start():end],
                "mention_kind": "EXCHANGE_QUALIFIED_SYMBOL_SOURCE_MENTION",
                "namespace_mention": match["namespace"], "symbol_mention": symbol,
                "share_class_mention": symbol.split(".")[1] if "." in symbol else None,
                "link_state": "UNKNOWN_NO_ISSUER_AUTHORITY",
                "instrument_identity_state": "UNKNOWN", "issuer_linking_state": "UNKNOWN",
                "exchange_membership_state": "UNKNOWN", "existence_state": "UNKNOWN",
                "provenance": evidence}))
        entities.extend(m[2] for m in sorted(mentions, key=lambda item: (item[0], item[1])))
    result.update(implementation=IMPLEMENTATION, entity_mentions=entities)
    result["limitations"] += [
        "Qualified symbols are complete source mentions only: NYSE/NASDAQ plus 1-5 ASCII uppercase letters and optional dot plus one uppercase share-class letter",
        "Namespace/share-class syntax establishes no permanent instrument identity, issuer link, exchange membership or real existence; all UNKNOWN",
        "Unsupported whole symbol candidates are rejected, never shortened into ticker facts"]
    return result
