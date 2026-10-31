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
    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        try:
            import duckdb
        except ImportError as exc:
            raise ImportError("Install duckdb (pip install duckdb) — no SQL server required") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.read_only = read_only
        # Readers must not take the exclusive lock — the terminal and the trader
        # otherwise fight over the same file on Windows.
        self.con = duckdb.connect(str(path), read_only=read_only)

    def init(self) -> None:
        if self.read_only:
            return
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

    def upsert_fundamentals(self, frame: pd.DataFrame) -> int:
        """Point-in-time fundamental fields. Requires asof_date, symbol, field, value."""
        if frame is None or frame.empty:
            return 0
        df = frame.copy()
        for col in ("asof_date", "symbol", "field", "value"):
            if col not in df.columns:
                raise ValueError(f"fundamentals missing column {col}")
        if "source" not in df.columns:
            df["source"] = "manual"
        df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.date
        df["symbol"] = df["symbol"].astype(str)
        df["field"] = df["field"].astype(str)
        # SEC can emit multiple XBRL rows with the same filing date; keep last.
        df = df.drop_duplicates(subset=["asof_date", "symbol", "field"], keep="last")
        symbols = sorted(df["symbol"].unique().tolist())
        for i in range(0, len(symbols), 400):
            chunk = symbols[i : i + 400]
            placeholders = ",".join(["?"] * len(chunk))
            self.con.execute(
                f"DELETE FROM fundamentals WHERE symbol IN ({placeholders})",
                chunk,
            )
        self.con.execute(
            "INSERT INTO fundamentals SELECT asof_date, symbol, field, value, source FROM df"
        )
        return len(df)

    def load_fundamentals(self) -> pd.DataFrame:
        try:
            return self.con.execute("SELECT * FROM fundamentals").df()
        except Exception:
            return pd.DataFrame()

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

    # --- trading audit trail -------------------------------------------------

    def record_trade_run(self, row: dict) -> None:
        import json

        self.con.execute(
            "INSERT INTO trade_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                row.get("ts") or datetime.now(UTC),
                str(row["run_id"]),
                row.get("asof"),
                bool(row.get("dry_run", True)),
                bool(row.get("blocked", False)),
                float(row.get("nav") or 0.0),
                float(row.get("gross") or 0.0),
                float(row.get("net") or 0.0),
                float(row.get("turnover") or 0.0),
                int(row.get("n_orders") or 0),
                str(row.get("construction") or ""),
                json.dumps(row.get("report") or {}, default=str),
            ],
        )

    def record_target_book(
        self,
        run_id: str,
        weights: pd.Series,
        sectors: pd.Series | None = None,
        prices: pd.Series | None = None,
        asof: date | None = None,
        ts: datetime | None = None,
    ) -> int:
        if weights is None or weights.empty:
            return 0
        stamp = ts or datetime.now(UTC)
        rows = [
            (
                stamp,
                str(run_id),
                asof,
                str(symbol),
                float(weight),
                str(sectors.get(symbol, "unknown")) if sectors is not None else "unknown",
                float(prices.get(symbol)) if prices is not None and symbol in prices.index else None,
            )
            for symbol, weight in weights.items()
        ]
        self.con.executemany("INSERT INTO target_book VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        return len(rows)

    def record_orders(self, run_id: str, rows: list[dict], ts: datetime | None = None) -> int:
        if not rows:
            return 0
        stamp = ts or datetime.now(UTC)
        payload = [
            (
                stamp,
                str(run_id),
                str(r.get("symbol")),
                str(r.get("side")),
                float(r.get("shares") or 0.0),
                r.get("limit_price"),
                r.get("reference_price"),
                float(r.get("notional") or 0.0),
                r.get("current_weight"),
                r.get("target_weight"),
                str(r.get("status") or ""),
                str(r.get("broker_order_id") or ""),
                str(r.get("note") or ""),
            )
            for r in rows
        ]
        self.con.executemany(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", payload
        )
        return len(payload)

    def nav_high_water(self) -> float | None:
        """
        Peak NAV across live (non-dry-run) trading runs.

        Used by the drawdown halt. Dry runs are excluded because they record the
        account NAV without having caused any of it.
        """
        row = self.con.execute(
            "SELECT MAX(nav) FROM trade_runs WHERE dry_run = FALSE AND nav > 0"
        ).fetchone()
        if not row or row[0] is None:
            return None
        return float(row[0])

    def load_trade_runs(self, limit: int = 50) -> pd.DataFrame:
        return self.con.execute(
            f"SELECT * FROM trade_runs ORDER BY ts DESC LIMIT {int(limit)}"
        ).df()

    def load_orders(self, run_id: str | None = None, limit: int = 200) -> pd.DataFrame:
        if run_id:
            return self.con.execute(
                f"SELECT * FROM orders WHERE run_id = ? ORDER BY ts DESC LIMIT {int(limit)}",
                [run_id],
            ).df()
        return self.con.execute(f"SELECT * FROM orders ORDER BY ts DESC LIMIT {int(limit)}").df()

    def stats(self) -> dict[str, int]:
        """
        Table row counts for the status strip.

        Full ``COUNT(*)`` on ``bars`` (multi-million rows) under the process lock
        stalls every other DuckDB reader and makes the UI flash "Lake empty".
        Prefer DuckDB's estimated_size; fall back to COUNT only for small tables
        or when the estimate is missing.
        """
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
            "trade_runs",
            "target_book",
            "orders",
            "fundamentals",
        ):
            out[table] = self._row_count(table, exact=table != "bars")
        return out

    def _row_count(self, table: str, *, exact: bool = True) -> int:
        if not exact:
            try:
                row = self.con.execute(
                    "SELECT estimated_size FROM duckdb_tables() WHERE table_name = ?",
                    [table],
                ).fetchone()
                if row and row[0] is not None and int(row[0]) > 0:
                    return int(row[0])
            except Exception:
                pass
            # Never full-scan multi-million bar tables on the status path.
            try:
                hit = self.con.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()
                if not hit:
                    return 0
                # Prefer last-day row count × distinct dates when cheap enough;
                # otherwise report a sentinel >0 so the UI does not flash "empty".
                day = self.con.execute(
                    f"""
                    SELECT COUNT(*) FROM {table}
                    WHERE dt = (SELECT MAX(dt) FROM {table})
                    """
                ).fetchone()
                n_day = int(day[0]) if day and day[0] is not None else 0
                return max(n_day, 1)
            except Exception:
                return 0
        try:
            n = self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            return int(n or 0)
        except Exception:
            return 0
