"""File-backed research database. DuckDB — SQL with no server to install."""

from __future__ import annotations

from datetime import UTC, date, datetime
from importlib import resources
from pathlib import Path

import pandas as pd

from gig.types import MarketPanel


def _schema_sql() -> str:
    return resources.files("gig.store").joinpath("schema.sql").read_text(encoding="utf-8")


class Store:
    def __init__(self, path: Path) -> None:
        try:
            import duckdb
        except ImportError as exc:
            raise ImportError("Install duckdb (pip install duckdb) — no SQL server required") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.con = duckdb.connect(str(path))

    def init(self) -> None:
        self.con.execute(_schema_sql())

    def close(self) -> None:
        self.con.close()

    def upsert_bars(self, panel: MarketPanel) -> int:
        close = panel.close.copy()
        close.index.name = "dt"
        close.columns.name = "symbol"
        volume = panel.volume.reindex_like(close).copy()
        volume.index.name = "dt"
        volume.columns.name = "symbol"
        dollar = (
            panel.dollar_volume.reindex_like(close)
            if panel.dollar_volume is not None
            else close * volume
        ).copy()
        dollar.index.name = "dt"
        dollar.columns.name = "symbol"
        df = pd.concat(
            [
                close.stack().rename("close"),
                volume.stack().rename("volume"),
                dollar.stack().rename("dollar_volume"),
            ],
            axis=1,
        ).reset_index()
        df["dt"] = pd.to_datetime(df["dt"]).dt.date
        symbols = list(panel.close.columns)
        lo, hi = df["dt"].min(), df["dt"].max()
        for i in range(0, len(symbols), 400):
            chunk = symbols[i : i + 400]
            placeholders = ",".join(["?"] * len(chunk))
            self.con.execute(
                f"DELETE FROM bars WHERE dt BETWEEN ? AND ? AND symbol IN ({placeholders})",
                [lo, hi, *chunk],
            )
        self.con.execute("INSERT INTO bars SELECT dt, symbol, close, volume, dollar_volume FROM df")
        return len(df)

    def upsert_universe(self, sectors: pd.Series, asof: date | None = None) -> None:
        asof = asof or date.today()
        df = sectors.rename("sector").rename_axis("symbol").reset_index()
        df["asof_date"] = asof
        self.con.execute("DELETE FROM universe WHERE asof_date = ?", [asof])
        self.con.execute("INSERT INTO universe SELECT symbol, sector, asof_date FROM df")

    def upsert_listings(self, listings: pd.DataFrame) -> int:
        if listings is None or listings.empty:
            return 0
        df = listings.copy()
        asof = df["asof_date"].iloc[0] if "asof_date" in df.columns else date.today()
        for col, default in (
            ("yahoo_symbol", df["symbol"] if "symbol" in df.columns else ""),
            ("exchange", ""),
            ("name", ""),
            ("is_etf", False),
            ("is_test", False),
            ("cap_bucket", "unknown"),
            ("asof_date", asof),
        ):
            if col not in df.columns:
                df[col] = default
        self.con.execute("DELETE FROM listings WHERE asof_date = ?", [asof])
        self.con.execute(
            "INSERT INTO listings SELECT symbol, yahoo_symbol, exchange, name, is_etf, is_test, cap_bucket, asof_date FROM df"
        )
        return len(df)

    def load_listings(self) -> pd.DataFrame:
        try:
            return self.con.execute(
                """
                SELECT * FROM listings
                WHERE asof_date = (SELECT MAX(asof_date) FROM listings)
                """
            ).df()
        except Exception:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "yahoo_symbol",
                    "exchange",
                    "name",
                    "is_etf",
                    "is_test",
                    "cap_bucket",
                    "asof_date",
                ]
            )

    def update_cap_buckets(self, buckets: pd.Series) -> None:
        if buckets is None or buckets.empty:
            return
        asof = self.con.execute("SELECT MAX(asof_date) FROM listings").fetchone()[0]
        if asof is None:
            return
        df = buckets.rename("cap_bucket").rename_axis("yahoo_symbol").reset_index()
        self.con.register("cap_map", df)
        self.con.execute(
            """
            UPDATE listings SET cap_bucket = cap_map.cap_bucket
            FROM cap_map
            WHERE listings.yahoo_symbol = cap_map.yahoo_symbol AND listings.asof_date = ?
            """,
            [asof],
        )

    def upsert_news(self, news: pd.DataFrame) -> int:
        if news is None or news.empty:
            return 0
        df = news.copy()
        self.con.execute("INSERT INTO news SELECT published, asof_date, symbol, title, source, sentiment, url FROM df")
        return len(df)

    def upsert_macro(self, macro: pd.DataFrame) -> int:
        if macro is None or macro.empty:
            return 0
        df = macro.copy()
        self.con.execute("DELETE FROM macro WHERE series_id IN (SELECT DISTINCT series_id FROM df)")
        self.con.execute("INSERT INTO macro SELECT dt, series_id, value FROM df")
        return len(df)

    def upsert_filings(self, filings: pd.DataFrame) -> int:
        if filings is None or filings.empty:
            return 0
        df = filings.copy()
        symbols = [str(s) for s in df["symbol"].dropna().unique().tolist()]
        if symbols:
            placeholders = ",".join(["?"] * len(symbols))
            self.con.execute(f"DELETE FROM filings WHERE symbol IN ({placeholders})", symbols)
        self.con.execute(
            "INSERT INTO filings SELECT filed_at, asof_date, symbol, form, accession, title, sentiment FROM df"
        )
        return len(df)

    def upsert_quotes(self, quotes: pd.DataFrame) -> int:
        if quotes is None or quotes.empty:
            return 0
        df = quotes.copy()
        self.con.execute("INSERT INTO quotes SELECT ts, symbol, bid, ask, last, source FROM df")
        return len(df)

    def upsert_calendar(self, cal: pd.DataFrame) -> int:
        if cal is None or cal.empty:
            return 0
        df = cal.copy()
        self.con.execute("INSERT INTO calendar SELECT event_date, symbol, event_type, extra FROM df")
        return len(df)

    def load_macro(self) -> pd.DataFrame:
        return self.con.execute("SELECT dt, series_id, value FROM macro").df()

    def load_filings(self) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM filings").df()

    def load_panel(self, start: date, end: date, symbols: list[str] | None = None) -> MarketPanel | None:
        if symbols:
            bars = self.con.execute(
                "SELECT dt, symbol, close, volume, dollar_volume FROM bars "
                "WHERE dt BETWEEN ? AND ? AND symbol IN (SELECT UNNEST(?))",
                [start, end, symbols],
            ).df()
        else:
            bars = self.con.execute(
                "SELECT dt, symbol, close, volume, dollar_volume FROM bars WHERE dt BETWEEN ? AND ?",
                [start, end],
            ).df()
        if bars.empty:
            return None
        bars["dt"] = pd.to_datetime(bars["dt"])
        close = bars.pivot(index="dt", columns="symbol", values="close").sort_index()
        volume = bars.pivot(index="dt", columns="symbol", values="volume").sort_index()
        dollar_volume = bars.pivot(index="dt", columns="symbol", values="dollar_volume").sort_index()
        uni = self.con.execute(
            """
            SELECT symbol, sector FROM (
                SELECT symbol, sector, ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY asof_date DESC) AS rn
                FROM universe
            ) t WHERE rn = 1
            """
        ).df()
        if uni.empty:
            sectors = pd.Series("unknown", index=close.columns, name="sector")
        else:
            sectors = uni.drop_duplicates("symbol").set_index("symbol")["sector"]
            sectors = sectors.reindex(close.columns).fillna("unknown")
        return MarketPanel(close=close, volume=volume, sectors=sectors, dollar_volume=dollar_volume)

    def record_experiment(self, name: str, config_hash: str, metrics: dict) -> None:
        import json

        self.con.execute(
            "INSERT INTO experiments VALUES (?, ?, ?, ?)",
            [datetime.now(UTC), name, config_hash, json.dumps(metrics, default=float)],
        )

    def stats(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for table in (
            "bars",
            "universe",
            "news",
            "factors",
            "experiments",
            "model_predictions",
            "macro",
            "filings",
            "quotes",
            "calendar",
            "listings",
        ):
            try:
                n = self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                out[table] = int(n)
            except Exception:
                out[table] = 0
        return out
