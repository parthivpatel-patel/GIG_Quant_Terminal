"""Explicit financial annotation profiles; omitted historical profiles mean v1.

The published v1 parser lives in nlp_v1. Never redirect that profile to a newer
parser: immutable snapshots must replay their original annotation semantics.
"""
from __future__ import annotations

from gig.financial_intelligence import nlp_v1, nlp_v2

# Preserve the previously public implementation identifier for legacy callers.
IMPLEMENTATION = nlp_v1.IMPLEMENTATION


def settings(profile, extract_facts=True):
    if not isinstance(profile, str) or profile not in {"v1", "v2"}:
        raise ValueError("financial NLP profile must be v1 or v2")
    if type(extract_facts) is not bool or (profile == "v2" and not extract_facts):
        raise ValueError("financial NLP v2 requires extract_facts=true")


def extract(passages, *, profile="v1"):
    settings(profile)
    # An explicit allowlist, never a dynamic module name or latest-profile alias.
    if profile == "v1":
        return nlp_v1.extract(passages)
    return nlp_v2.extract(passages)
