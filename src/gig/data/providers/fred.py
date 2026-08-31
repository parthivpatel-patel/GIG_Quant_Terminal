"""FRED (St. Louis Fed) — free macro series. Requires FRED_API_KEY."""

from __future__ import annotations

import json

import pandas as pd

from gig.data.http import GetFn, http_get

# Series a rates/macro desk actually looks at
DEFAULT_SERIES = (
    "DGS10",  # 10y Treasury
    "DGS2",  # 2y Treasury
    "T10Y2Y",  # 10y-2y curve
    "BAMLH0A0HYM2",  # HY OAS
    "VIXCLS",  # VIX
    "UNRATE",
    "CPIAUCSL",
)


def parse_observations(payload: str, series_id: str) -> pd.DataFrame:
    data = json.loads(payload)
    rows = []
    for obs in data.get("observations", []):
        val = obs.get("value", ".")
        if val in (".", "", None):
            continue
        rows.append({"dt": obs["date"], "series_id": series_id, "value": float(val)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["dt"] = pd.to_datetime(df["dt"]).dt.date
    return df


def fetch_series(
    series_id: str,
    api_key: str,
    *,
    get: GetFn = http_get,
) -> pd.DataFrame:
    if not api_key:
        raise RuntimeError("FRED_API_KEY is empty — get a free key at https://fred.stlouisfed.org/docs/api/api_key.html")
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={api_key}&file_type=json"
    )
    return parse_observations(get(url, None), series_id)


def fetch_default_macro(api_key: str, *, get: GetFn = http_get) -> pd.DataFrame:
    frames = []
    for sid in DEFAULT_SERIES:
        try:
            frames.append(fetch_series(sid, api_key, get=get))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["dt", "series_id", "value"])
    return pd.concat(frames, ignore_index=True)
