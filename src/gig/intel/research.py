"""Global events to financial research: event studies, exposure and scenarios.

Outputs are descriptive research. A single-event study cannot separate an event
from coincident confounders; abnormal returns are not causal effects. Scenario
shocks are analyst assumptions, labeled ANALYTICAL_SCENARIO.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from gig.intel.reference import THEMES
from gig.timestamps import aware_timestamp


def _series(rows: Any, name: str) -> tuple[list[Any], np.ndarray]:
    if not isinstance(rows, list) or not 30 <= len(rows) <= 5000:
        raise ValueError(f"{name} must contain 30 to 5000 observations")
    stamps, closes = [], []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"date", "close"}:
            raise ValueError(f"{name} rows require date and close")
        value = row["close"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} close must be positive and finite")
        stamps.append(aware_timestamp(row["date"]))
        closes.append(float(value))
    if any(b <= a for a, b in zip(stamps, stamps[1:], strict=False)):
        raise ValueError(f"{name} dates must be strictly increasing")
    return stamps, np.asarray(closes)


def event_study(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {"instrument", "classification", "source", "prices", "event_at", "estimation_window", "pre", "post"}
    if not isinstance(payload, Mapping) or not required <= set(payload) or set(payload) - required - {"market_prices", "event_id"}:
        raise ValueError("event study fields do not match the contract")
    if payload["classification"] not in ("HISTORICAL", "SYNTHETIC"):
        raise ValueError("classification must be HISTORICAL or SYNTHETIC")
    for key, limit in (("instrument", 64), ("source", 200), ("event_id", 200)):
        value = payload.get(key)
        if (key != "event_id" or value is not None) and (not isinstance(value, str) or not value.strip() or len(value) > limit):
            raise ValueError(f"{key} must be a nonempty string of at most {limit} characters")
    stamps, closes = _series(payload["prices"], "prices")
    returns = np.diff(np.log(closes))
    rstamps = stamps[1:]
    market = None
    if payload.get("market_prices") is not None:
        mstamps, mcloses = _series(payload["market_prices"], "market_prices")
        if mstamps != stamps:
            raise ValueError("market_prices must share the instrument's dates")
        market = np.diff(np.log(mcloses))
    window, pre, post = payload["estimation_window"], payload["pre"], payload["post"]
    for value, low, high in ((window, 20, 1000), (pre, 0, 20), (post, 0, 60)):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError("window parameters out of range")
    event_at = aware_timestamp(payload["event_at"])
    # Price stamps may be dates without a close time. Aligning on the event's UTC
    # calendar date keeps any return that could contain the event out of the
    # estimation window, at the cost of possibly starting the window a day early.
    event_day = event_at.tz_convert("UTC").normalize()
    day0 = next((i for i, t in enumerate(rstamps) if t >= event_day), None)
    if day0 is None or day0 - pre - window < 0 or day0 + post >= len(returns):
        raise ValueError("insufficient observations around the event")
    est = slice(day0 - pre - window, day0 - pre)
    if market is None:
        mu = float(returns[est].mean())
        residual = returns[est] - mu
        expected = np.full(len(returns), mu)
        model = "MEAN_ADJUSTED"
        params: dict[str, float] = {"mean": mu}
    else:
        x, y = market[est], returns[est]
        beta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)) if np.var(x) > 0 else 0.0
        alpha = float(y.mean() - beta * x.mean())
        residual = y - (alpha + beta * x)
        expected = alpha + beta * market
        model = "MARKET_MODEL_OLS"
        params = {"alpha": alpha, "beta": beta}
    dof = window - (1 if market is None else 2)
    sigma = float(np.sqrt((residual ** 2).sum() / dof))
    rows, car = [], 0.0
    for offset in range(-pre, post + 1):
        i = day0 + offset
        ar = float(returns[i] - expected[i])
        car += ar
        rows.append({"offset": offset, "date": rstamps[i].isoformat(), "return": float(returns[i]),
                     "expected": float(expected[i]), "abnormal_return": ar, "cumulative_abnormal_return": car})
    length = pre + post + 1
    t_stat = car / (sigma * math.sqrt(length)) if sigma > 0 else None
    return {
        "instrument": payload["instrument"], "source": payload["source"],
        "data_classification": payload["classification"], "event_id": payload.get("event_id"),
        "event_at": event_at.isoformat(), "day0": rstamps[day0].isoformat(),
        "day0_rule": "FIRST_RETURN_ENDING_ON_OR_AFTER_EVENT_UTC_DATE", "model": model, "parameters": params,
        "estimation": {"start": rstamps[est.start].isoformat(), "end": rstamps[est.stop - 1].isoformat(),
                       "observations": window, "residual_sigma": sigma},
        "window": rows, "car": car, "car_t_stat": t_stat,
        "uses_post_event_data_for_estimation": False,
        "interpretation": "DESCRIPTIVE_SINGLE_EVENT_NOT_CAUSAL",
        "limitations": ["Confounding events in the window are not controlled.",
                        "Normal-residual t-statistic; no multiple-testing or clustering adjustment.",
                        "Abnormal return is relative to the stated model, not a causal effect."],
    }


def theme_exposure(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Portfolio exposure to research themes and optional analyst scenario shocks."""
    if not isinstance(payload, Mapping) or set(payload) - {"holdings", "themes", "shocks", "nav"} or "holdings" not in payload:
        raise ValueError("exposure fields do not match the contract")
    holdings = payload["holdings"]
    if not isinstance(holdings, list) or not 1 <= len(holdings) <= 500:
        raise ValueError("holdings must be a bounded nonempty list")
    nav = payload.get("nav")
    if nav is not None and (isinstance(nav, bool) or not isinstance(nav, (int, float)) or not math.isfinite(nav) or nav <= 0):
        raise ValueError("nav must be positive")
    active = payload.get("themes")
    if active is not None and (not isinstance(active, list) or set(active) - set(THEMES)):
        raise ValueError("unknown research theme")
    shocks = payload.get("shocks") or {}
    if not isinstance(shocks, Mapping) or set(shocks) - set(THEMES):
        raise ValueError("unknown scenario theme")
    for value in shocks.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not -1 <= value <= 1:
            raise ValueError("scenario shocks must be decimal returns in [-1, 1]")
    exposure: dict[str, float] = {}
    untagged = 0.0
    gross = 0.0
    for holding in holdings:
        if not isinstance(holding, Mapping) or set(holding) != {"instrument", "market_value", "themes"}:
            raise ValueError("holdings require instrument, market_value and themes")
        value = holding["market_value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("long-only funded holdings require nonnegative market_value")
        tags = holding["themes"]
        if not isinstance(tags, list) or set(tags) - set(THEMES):
            raise ValueError("unknown holding theme")
        gross += value
        if not tags:
            untagged += value
        for tag in set(tags):
            exposure[tag] = exposure.get(tag, 0.0) + float(value)
    selected = active or sorted(exposure)
    base = nav if nav is not None else gross
    rows = [{"theme": t, "label": THEMES[t]["label"], "market_value": exposure.get(t, 0.0),
             "share_of_nav": exposure.get(t, 0.0) / base if base else None} for t in selected]
    scenario = None
    if shocks:
        # A holding tagged with several shocked themes takes the most adverse
        # shock once, rather than compounding overlapping theme tags.
        pnl = 0.0
        for holding in holdings:
            applied = [shocks[t] for t in holding["themes"] if t in shocks]
            if applied:
                pnl += holding["market_value"] * min(applied)
        scenario = {"shocks": dict(shocks), "pnl": pnl, "pnl_share_of_nav": pnl / base if base else None,
                    "aggregation": "MOST_ADVERSE_SHOCK_PER_HOLDING", "classification": "ANALYTICAL_SCENARIO"}
    return {"themes": rows, "gross_market_value": gross, "untagged_market_value": untagged,
            "nav_basis": "SUPPLIED_NAV" if nav is not None else "GROSS_MARKET_VALUE",
            "scenario": scenario, "theme_tags": "CALLER_SUPPLIED",
            "interpretation": "Exposure to research themes; overlapping tags mean shares can exceed 100%."}


def risk_monitor(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count recent events by linked research theme; descriptive only."""
    from gig.intel.reference import research_links
    counts: dict[str, dict[str, Any]] = {}
    for event in events:
        links = event.get("research_links") or research_links(event.get("category", ""), event.get("latitude"), event.get("longitude"))
        for theme in links["themes"]:
            row = counts.setdefault(theme["theme"], {"theme": theme["theme"], "label": theme["label"],
                                                     "markets": theme["markets"], "events": 0, "confirmed": 0,
                                                     "unverified": 0, "other": 0, "event_ids": []})
            row["events"] += 1
            verification = event.get("verification", "")
            if verification in ("CONFIRMED_OFFICIAL", "PROVIDER_REVIEWED", "PROVIDER_CURATED", "MULTI_SOURCE_REPORTED"):
                row["confirmed"] += 1
            elif verification in ("UNVERIFIED_REPORT", "PROVIDER_AUTOMATIC_SUBJECT_TO_REVISION"):
                row["unverified"] += 1
            else:
                row["other"] += 1
            if len(row["event_ids"]) < 20:
                row["event_ids"].append(event["event_id"])
    rows = sorted(counts.values(), key=lambda r: (-r["events"], r["theme"]))
    return {"themes": rows, "link_type": "RESEARCH_HYPOTHESIS_NOT_CAUSAL",
            "note": "Counts of linked events are attention indicators, not risk forecasts or price predictions."}
