"""Observed Treasury par-yield geometry; no interpolation or spot-curve claim."""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

TENOR_YEARS = {
    "BC_1MONTH": 1 / 12,
    "BC_1_5MONTH": 1.5 / 12,
    "BC_2MONTH": 2 / 12,
    "BC_3MONTH": 3 / 12,
    "BC_4MONTH": 4 / 12,
    "BC_6MONTH": .5,
    "BC_1YEAR": 1.,
    "BC_2YEAR": 2.,
    "BC_3YEAR": 3.,
    "BC_5YEAR": 5.,
    "BC_7YEAR": 7.,
    "BC_10YEAR": 10.,
    "BC_20YEAR": 20.,
    "BC_30YEAR": 30.,
}


def quoted_yield(value: Any) -> Decimal:
    """Bound source decimals before arithmetic, formatting or persistence."""
    if not isinstance(value, str) or not 1 <= len(value) <= 64:
        raise ValueError("yield requires a bounded decimal source string")
    try:
        result = Decimal(value)
    except InvalidOperation:
        raise ValueError("invalid Treasury yield") from None
    exponent = result.as_tuple().exponent
    if (
        not result.is_finite()
        or not Decimal("-100") <= result <= Decimal("100")
        or not isinstance(exponent, int)
        or exponent < -12
    ):
        raise ValueError("yield exceeds supported finite range or precision")
    return result


def curve_analytics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize only visible quotes; spreads require the same observation date."""
    if not isinstance(rows, list) or not 1 <= len(rows) <= 1000:
        raise ValueError("curve requires one to 1000 visible observations")
    dates = []
    values = []
    identities = set()
    for row in rows:
        event = row.get("event_date")
        if not isinstance(event, str) or date.fromisoformat(event).isoformat() != event:
            raise ValueError("curve observation requires an ISO date")
        tenor = row.get("tenor")
        if not isinstance(tenor, str) or tenor not in TENOR_YEARS:
            raise ValueError("unsupported Treasury tenor")
        identity = (event, tenor)
        if identity in identities:
            raise ValueError("curve contains conflicting observation identity")
        identities.add(identity)
        dates.append(date.fromisoformat(event))
        values.append(quoted_yield(row.get("yield_percent")))
    latest = max(dates)
    curve = {row["tenor"]: value for row, day, value in zip(rows, dates, values, strict=True)
             if day == latest}

    def spread(long: str, short: str) -> dict[str, Any]:
        if long not in curve or short not in curve:
            return {"value": None, "state": "UNAVAILABLE", "unit": "basis_points",
                    "description": "Both tenors must be observed on the latest visible date"}
        with localcontext() as context:
            context.prec = 50
            amount = (curve[long] - curve[short]) * 100
        return {"value": format(amount, "f"), "state": "AVAILABLE", "unit": "basis_points",
                "description": f"Same-date par-yield difference on {latest.isoformat()}"}

    return {
        "metrics": {
            "latest_observation_date": latest.isoformat(),
            "latest_visible_tenors": len(curve),
            "spread_10y_2y": spread("BC_10YEAR", "BC_2YEAR"),
            "spread_10y_3m": spread("BC_10YEAR", "BC_3MONTH"),
            "spread_30y_5y": spread("BC_30YEAR", "BC_5YEAR"),
        },
        "chart": {
            "kind": "yield-curve-surface",
            "title": "Observed Treasury par yields",
            "axes": [
                {"key": "maturity_years", "label": "Quoted tenor", "unit": "nominal years"},
                {"key": "observation_day_offset", "label": "Observation date", "unit": "signed calendar days from latest date"},
                {"key": "yield_percent", "label": "Par yield", "unit": "percent per year"},
            ],
            "points": [
                {"maturity_years": TENOR_YEARS[row["tenor"]],
                 "observation_day_offset": (day - latest).days,
                 "yield_percent": float(value), "event_date": row["event_date"],
                 "tenor": row["tenor"], "available_at": row["available_at"],
                 "source_sha256": row["source_sha256"], "classification": row["classification"]}
                for row, day, value in sorted(zip(rows, dates, values, strict=True),
                                            key=lambda item: (item[1], TENOR_YEARS[item[0]["tenor"]]))
            ],
            "interpolated": False,
            "source_value_precision": "Exact source strings retained in table; chart coordinates are floats",
        },
        "description": "Observed par yields, not bootstrapped spot rates or tradable bond prices. Lines join observed samples without creating quotes. Tenor years are nominal display coordinates; six weeks uses 1.5/12 years.",
        "notes": ["Spreads use only the latest visible date; missing tenors remain unavailable.",
                  "Acquisition freshness does not establish current market observation freshness.",
                  "Historical publication times remain unknown; no historical trading availability is inferred."],
    }
