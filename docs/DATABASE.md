# Database setup (no SQL server)

GIG uses **DuckDB**: one file (`data/gig.duckdb`), full SQL, no MySQL/Postgres/Docker.

## 1. Install (already in the project)

```bash
pip install -e ".[dev]"
```

`duckdb` is a regular Python package. Nothing runs as a Windows service.

## 2. Create the file and tables

```bash
python -m gig db init
```

That writes `data/gig.duckdb` and runs `src/gig/store/schema.sql`.

## 3. Pull live market data

```bash
python -m gig ingest
python -m gig db stats
```

Bars, universe, and news land in the same file. A second ingest refreshes prices; the backtest reads the cache first so you are not hitting Yahoo on every run.

## 4. Inspect it (optional GUI)

You do **not** need SQL Server Management Studio.

- **DBeaver** (free): New connection → DuckDB → path `C:\Quant Algo\data\gig.duckdb`
- **Python**

```python
import duckdb
con = duckdb.connect("data/gig.duckdb")
con.sql("SELECT symbol, COUNT(*) n FROM bars GROUP BY 1 ORDER BY 2 DESC LIMIT 10").show()
con.sql("SELECT * FROM news ORDER BY published DESC LIMIT 5").show()
```

## 5. What each table is

| Table | Contents |
|---|---|
| `bars` | Daily close, volume, dollar volume (live Yahoo) |
| `universe` | Symbol → sector as-of a date |
| `listings` | Nasdaq Trader US commons (exchange, ETF flag, cap bucket) |
| `news` | Headlines with as-of date and sentiment |
| `factors` | Optional persisted factor values |
| `experiments` | Hashed backtest configs + metrics |
| `macro` | FRED series (VIX, curve, HY OAS, …) |
| `filings` | SEC 8-K / 10-K as-of filing date |
| `quotes` | Alpaca IEX / Finnhub live snapshots |
| `calendar` | Finnhub earnings dates |

## 6. The old `gig_database.db`

That SQLite file belonged to the previous dashboard (users/sessions/signals). Do not mix it with this research lake. Leave it alone or delete it after you confirm you do not need the old UI.

## 7. If you later want Postgres

The schema is ordinary SQL. A desk would swap `Store` to a Postgres connection string. You do not need that to research or to interview.
