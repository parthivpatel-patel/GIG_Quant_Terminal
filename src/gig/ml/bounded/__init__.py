"""Bounded CPU research families registered behind one shared contract."""
from __future__ import annotations

from typing import Any

from gig.ml.bounded import (  # importing registers each family
    boosting,
    history,
    latent,
    neural,
    supervised,
    tails,
)
from gig.ml.bounded.core import (
    FAMILIES,
    IMPLEMENTATION,
    MAX_FITS,
    MAX_PREDICT,
    Family,
    admit,
    fit_family,
    predict_family,
)

__all__ = ["FAMILIES", "IMPLEMENTATION", "MAX_FITS", "MAX_PREDICT", "Family", "admit", "boosting", "capabilities",
           "fit_family", "history", "latent", "neural", "predict_family", "supervised", "tails"]


def capabilities() -> dict[str, dict[str, Any]]:
    """One capability per inventory family; fitted instances exist only in persisted research."""
    output: dict[str, dict[str, Any]] = {}
    for family in FAMILIES.values():
        entry = output.setdefault(family.capability, {
            "state": "AVAILABLE", "implemented": True, "trained": False, "trading_approved": False,
            "dependency_available": True, "family_number": family.number, "variants": [], "reason": ""})
        entry["variants"].append(family.key)
        if not family.available():
            entry.update(state="UNAVAILABLE", dependency_available=False)
        entry["reason"] = ("Bounded CPU research variants " + ", ".join(entry["variants"])
                           + " in the persisted-dataset workspace; uncalibrated; research only")
    return output
