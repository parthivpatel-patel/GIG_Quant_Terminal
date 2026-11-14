"""Original bounded BM25 implementation; corpus statistics use only visible data."""
from __future__ import annotations

import math
import re
from collections import Counter

IMPLEMENTATION = "gig.lexical-bm25.v1"
SYNONYMS = {"10-year": "bc_10year", "10year": "bc_10year", "ten-year": "bc_10year",
            "2-year": "bc_2year", "two-year": "bc_2year", "30-year": "bc_30year"}
TOKEN = re.compile(r"[a-z0-9]+(?:[_.-][a-z0-9]+)*")


def tokens(text: str):
    return TOKEN.findall(text.casefold())


def rank(passages, query: str, limit: int, *, expand: bool = False):
    return _rank(passages, query, limit, expand=expand)


def rank_candidates(passages, query: str, *, expand: bool = False):
    """All BM25 candidates for bounded hybrid fusion, with identical scoring."""
    return _rank(passages, query, 20, expand=expand, all_candidates=True)


def _rank(passages, query: str, limit: int, *, expand=False, all_candidates=False):
    if not isinstance(query, str) or not query.strip() or len(query) > 512:
        raise ValueError("bounded nonempty financial query required")
    if type(limit) is not int or not 1 <= limit <= 20 or type(expand) is not bool:
        raise ValueError("bounded exact retrieval settings required")
    if not isinstance(passages, list) or len(passages) > 4096:
        raise ValueError("financial corpus passage budget exceeded")
    terms = set(tokens(query))
    if not terms or len(terms) > 64:
        raise ValueError("query requires one to64lexical terms")
    if expand:
        terms.update(SYNONYMS[t] for t in tuple(terms) if t in SYNONYMS)
    if not passages:
        return [], sorted(terms)
    counts = [Counter(tokens(p["text"])) for p in passages]
    if any(sum(c.values()) > 1024 for c in counts):
        raise ValueError("financial passage token budget exceeded")
    average = sum(sum(c.values()) for c in counts)/len(counts)
    if average == 0:
        return [], sorted(terms)
    frequency = {term: sum(term in c for c in counts) for term in terms}
    output = []
    for passage, count in zip(passages, counts, strict=True):
        length = sum(count.values())
        score = math.fsum(math.log1p((len(counts)-frequency[term]+.5)/(frequency[term]+.5))
            * count[term]*2.2/(count[term]+1.2*(.25+.75*length/average))
            for term in sorted(terms) if count[term])
        if score > 0:
            output.append({**passage, "score": score, "score_kind": "BM25_RELEVANCE_NOT_FINANCIAL_CONFIDENCE"})
    output.sort(key=lambda p: (-p["score"], p["document_id"], p["ordinal"]))
    return (output if all_candidates else output[:limit]), sorted(terms)
