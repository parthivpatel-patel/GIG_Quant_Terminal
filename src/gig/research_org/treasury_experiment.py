"""Lagged sign of the 10-year CMT change, on the official 2023 par-yield file.

The file is the Treasury XML retrieved on 2026-11-05. Yields are not transaction
prices, and the file has no publication clock, so the result is not a futures
backtest and cannot be a positive market claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from gig.data.ingestion import parse_treasury
from gig.research.experiment import config_hash

HYPOTHESIS_ID = "treasury-10y-lagged-sign-2023"
FIXTURE = Path(__file__).parent / "fixtures" / "treasury_par_yield_2023.xml"
PROVENANCE = Path(__file__).parent / "fixtures" / "treasury_par_yield_2023.json"
PLACEBO_DRAWS = 16
PLACEBO_SEED = 31


def _load() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = FIXTURE.read_bytes()
    meta = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    digest = hashlib.sha256(raw).hexdigest()
    if digest != meta["sha256"]:
        raise RuntimeError("treasury fixture hash mismatch")
    rows = parse_treasury(raw, meta["retrieved_at"])
    series = [row for row in rows if row["tenor"] == "BC_10YEAR"]
    if len(series) < 30:
        raise RuntimeError("treasury 10-year series is too short")
    return meta, series


def _pairs(changes: list[Decimal]) -> tuple[int, int, Decimal]:
    hits = 0
    used = 0
    signed = Decimal(0)
    for index in range(1, len(changes)):
        previous, nxt = changes[index - 1], changes[index]
        if previous == 0 or nxt == 0:
            continue
        used += 1
        if (previous > 0) == (nxt > 0):
            hits += 1
        signed += nxt if previous > 0 else -nxt
    return hits, used, signed


def run_treasury() -> tuple[str, dict[str, Any]]:
    meta, series = _load()
    levels = [Decimal(row["yield_percent"]) for row in series]
    changes = [levels[index] - levels[index - 1] for index in range(1, len(levels))]
    hits, used, signed = _pairs(changes)
    if used == 0:
        raise RuntimeError("treasury changes were all zero")
    rate = hits / used
    mean_signed = signed / Decimal(used)
    rng = np.random.default_rng(PLACEBO_SEED)
    placebo_rates: list[float] = []
    base = np.array([float(value) for value in changes])
    for _ in range(PLACEBO_DRAWS):
        values = rng.permutation(base)
        ph = pu = 0
        for index in range(1, len(values)):
            previous, nxt = float(values[index - 1]), float(values[index])
            if previous == 0.0 or nxt == 0.0:
                continue
            pu += 1
            if (previous > 0) == (nxt > 0):
                ph += 1
        if pu:
            placebo_rates.append(ph / pu)
    config = {
        "hypothesis_id": HYPOTHESIS_ID, "dataset": meta["dataset_id"], "sha256": meta["sha256"],
        "tenor": "BC_10YEAR", "signal": "sign of the previous daily yield change",
        "classification": "HISTORICAL",
    }
    report: dict[str, Any] = {
        "hypothesis_id": HYPOTHESIS_ID,
        "question": (
            "On the official 2023 daily 10-year par yield, does the sign of the previous "
            "change match the sign of the next change more often than not?"
        ),
        "strategy": "gig.research_org.treasury_experiment",
        "experiment_version": 1,
        "classification": "HISTORICAL",
        "evidence_class": "INCONCLUSIVE",
        "verdict": "INCONCLUSIVE",
        "economic_claim": "NONE",
        "publication_time": "UNKNOWN",
        "lookahead_detected": False,
        "hits": hits,
        "pairs": used,
        "hit_rate": rate,
        "mean_signed_next_change_percent": float(mean_signed),
        "placebo_draws": PLACEBO_DRAWS,
        "placebo_hit_rate_median": float(np.median(placebo_rates)) if placebo_rates else None,
        "n_obs": len(series),
        "period": {"start": series[0]["event_date"], "end": series[-1]["event_date"]},
        "dataset": {
            "id": meta["dataset_id"], "provider": "U.S. Treasury", "classification": "HISTORICAL",
            "source_url": meta["source_url"], "license": meta["license"], "sha256": meta["sha256"],
            "retrieved_at": meta["retrieved_at"], "currency": "USD yield percent",
            "instrument": "BC_10YEAR constant-maturity par yield",
            "point_in_time": "Calendar date only. The publication clock is not in the file.",
        },
        "training_boundary": "No parameter was estimated. The rule is the sign of the previous change.",
        "holdout": {"note": "The whole calendar year is the sample. It was not split to choose the rule."},
        "cost_bps_model": "None. These are yields, not executed trades.",
        "orders_submitted": 0,
        "models_promoted": 0,
        "code_rewritten": False,
        "sharpe_after_cost": None,
        "total_return": None,
        "max_drawdown": None,
        "n_trials": 1,
        "method": "Sign of the previous daily change compared with the next daily change. Zero changes are skipped.",
        "interpretation": (
            f"Inconclusive. On {used} non-zero pairs the lagged sign matched {hits} times "
            f"(hit rate {rate:.4f}). The mean yield change in the direction of that sign is "
            f"{float(mean_signed):.6f} percentage points. The file is an official historical "
            "yield curve, and it has no publication clock, so this is not evidence of a tradable edge."
        ),
        "failure_conditions": [
            "Publication time is UNKNOWN, so the calendar date is not a decision timestamp.",
            "The sample is one tenor and one calendar year.",
            "Hit rate on this file is below one half." if rate < 0.5 else "Hit rate on this file is not below one half.",
            "No transaction cost is defined because no trade is observed.",
        ],
        "reproduce": "python -m gig.research_org --compute treasury-10y-lagged-sign-2023",
        "lineage": {
            "dataset": meta["dataset_id"],
            "sha256": meta["sha256"],
            "strategy": "gig.research_org.treasury_experiment",
            "strategy_version": 1,
            "model_version": None,
            "point_in_time": "DATE_ONLY. Publication clock is not in the file.",
            "costs": "not applicable",
            "assumptions": "Par yields are not transaction prices. Zero changes are skipped.",
            "configuration": "BC_10YEAR, sign of the previous daily change, placebo seed 31, 16 draws",
            "skeptic_verdict": "INCONCLUSIVE",
            "evidence_class": "INCONCLUSIVE",
            "verdict": "INCONCLUSIVE",
            "reproduce": "python -m gig.research_org --compute treasury-10y-lagged-sign-2023",
            "artifacts": "Vendored XML and provenance JSON under research_org/fixtures, plus the ledger row.",
        },
    }
    if not math.isfinite(rate):
        raise RuntimeError("treasury hit rate was not finite")
    return config_hash(config), report
