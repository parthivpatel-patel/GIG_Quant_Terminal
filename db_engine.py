"""
GIG — DATABASE LAYER (db_engine.py)
SQLite persistence — signals, trades, top picks, auth all survive restarts.
"""
import sqlite3, json, hashlib, secrets, time, os, logging, threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
logger = logging.getLogger("GIG_DB")
DB_PATH = os.environ.get("GIG_DB_PATH", "gig_database.db")
_local = threading.local()

def _get_conn():
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
    return _local.conn

def init_db():
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, created_at TEXT DEFAULT (datetime('now')), last_login TEXT);
        CREATE TABLE IF NOT EXISTS sessions (token TEXT PRIMARY KEY, user_id INTEGER NOT NULL, created_at TEXT DEFAULT (datetime('now')), expires_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS signals_history (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, direction TEXT, composite REAL, conviction REAL, price REAL, rsi REAL, hurst REAL, regime TEXT, mode TEXT, entry_price REAL, stop_loss REAL, target REAL, rr_ratio REAL, shares INTEGER, category TEXT DEFAULT 'signal', metadata TEXT, created_at TEXT DEFAULT (datetime('now')));
        CREATE INDEX IF NOT EXISTS idx_sig_sym ON signals_history(symbol);
        CREATE INDEX IF NOT EXISTS idx_sig_time ON signals_history(created_at);
        CREATE TABLE IF NOT EXISTS top_picks (category TEXT PRIMARY KEY, symbol TEXT NOT NULL, direction TEXT, price REAL, entry_price REAL, stop_loss REAL, target REAL, rr_ratio REAL, shares INTEGER, score REAL, mode TEXT, details TEXT, updated_at TEXT DEFAULT (datetime('now')));
        CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, direction TEXT NOT NULL, entry_price REAL NOT NULL, exit_price REAL, shares INTEGER, entry_time TEXT DEFAULT (datetime('now')), exit_time TEXT, pnl REAL, pnl_pct REAL, status TEXT DEFAULT 'open', strategy TEXT, signals_snapshot TEXT, notes TEXT);
        CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
        CREATE TABLE IF NOT EXISTS ic_weights (signal_name TEXT PRIMARY KEY, weight REAL NOT NULL, ic_mean REAL, ic_sharpe REAL, trades_count INTEGER DEFAULT 0, updated_at TEXT DEFAULT (datetime('now')));
        CREATE TABLE IF NOT EXISTS performance_daily (date TEXT PRIMARY KEY, pnl REAL DEFAULT 0, pnl_pct REAL DEFAULT 0, trades_count INTEGER DEFAULT 0, win_count INTEGER DEFAULT 0, sharpe REAL, drawdown REAL, nav REAL);
        CREATE TABLE IF NOT EXISTS kv_store (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT DEFAULT (datetime('now')));
        CREATE TABLE IF NOT EXISTS market_ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            price REAL,
            change_pct REAL,
            market_state TEXT,
            regular_price REAL,
            pre_price REAL,
            post_price REAL,
            provider TEXT,
            ts_epoch REAL NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_ticks_sym_ts ON market_ticks(symbol, ts_epoch DESC);
        CREATE TABLE IF NOT EXISTS news_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            source TEXT,
            sentiment TEXT,
            url TEXT,
            symbols TEXT,
            ts_epoch REAL NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_news_ts ON news_snapshots(ts_epoch DESC);
        CREATE TABLE IF NOT EXISTS geo_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            risk_score REAL,
            conflicts_count INTEGER,
            aircraft_count INTEGER,
            earthquakes_count INTEGER,
            fires_count INTEGER,
            payload TEXT,
            ts_epoch REAL NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_geo_ts ON geo_snapshots(ts_epoch DESC);
    """)
    conn.commit()
    logger.info(f"GIG Database initialized: {DB_PATH}")

class Auth:
    @staticmethod
    def _hash(pw):
        salt = secrets.token_hex(16)
        return f"{salt}:{hashlib.sha256(f'{salt}:{pw}'.encode()).hexdigest()}"
    @staticmethod
    def _verify(pw, stored):
        salt, h = stored.split(":", 1)
        return hashlib.sha256(f"{salt}:{pw}".encode()).hexdigest() == h
    @classmethod
    def create_user(cls, username, password):
        try:
            _get_conn().execute("INSERT INTO users (username, password_hash) VALUES (?,?)", (username, cls._hash(password)))
            _get_conn().commit(); return True
        except sqlite3.IntegrityError: return False
    @classmethod
    def login(cls, username, password):
        row = _get_conn().execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if not row or not cls._verify(password, row["password_hash"]): return None
        token = secrets.token_urlsafe(32)
        expires = (datetime.now() + timedelta(hours=24)).isoformat()
        _get_conn().execute("INSERT INTO sessions (token, user_id, expires_at) VALUES (?,?,?)", (token, row["id"], expires))
        _get_conn().execute("UPDATE users SET last_login=datetime('now') WHERE id=?", (row["id"],))
        _get_conn().commit(); return token
    @classmethod
    def verify_token(cls, token):
        row = _get_conn().execute("SELECT s.*, u.username FROM sessions s JOIN users u ON s.user_id=u.id WHERE s.token=? AND s.expires_at>datetime('now')", (token,)).fetchone()
        return {"user_id": row["user_id"], "username": row["username"]} if row else None
    @classmethod
    def logout(cls, token):
        _get_conn().execute("DELETE FROM sessions WHERE token=?", (token,)); _get_conn().commit()
    @classmethod
    def setup_default_user(cls):
        if _get_conn().execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            u = os.environ.get("GIG_USERNAME", "admin")
            p = os.environ.get("GIG_PASSWORD", "gig2026")
            cls.create_user(u, p)
            logger.info(f"Default user created: {u}")

class SignalStore:
    @classmethod
    def save_signals(cls, signals, category="signal"):
        conn = _get_conn()
        for s in signals:
            conn.execute("INSERT INTO signals_history (symbol,direction,composite,conviction,price,rsi,hurst,regime,mode,entry_price,stop_loss,target,rr_ratio,shares,category,metadata) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (s.get("symbol"),s.get("direction"),s.get("composite"),s.get("conviction"),s.get("price"),s.get("rsi"),s.get("hurst"),s.get("regime"),s.get("mode"),s.get("entry") or s.get("entry_price"),s.get("stop_loss"),s.get("target_2r") or s.get("target"),s.get("rr_ratio"),s.get("shares"),category,json.dumps({k:v for k,v in s.items() if k not in ("symbol","direction","composite","price")})))
        conn.commit()
    @classmethod
    def get_latest(cls, category=None, limit=50):
        conn = _get_conn()
        if category:
            rows = conn.execute("SELECT * FROM signals_history WHERE category=? ORDER BY created_at DESC LIMIT ?", (category, limit)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM signals_history ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
    @classmethod
    def get_signal_history(cls, symbol, limit=100):
        return [dict(r) for r in _get_conn().execute("SELECT * FROM signals_history WHERE symbol=? ORDER BY created_at DESC LIMIT ?", (symbol, limit)).fetchall()]

class TopPicks:
    CATEGORIES = ["best_stock", "best_1dte", "best_squeeze", "best_swing", "best_options"]
    @classmethod
    def update(cls, category, data):
        _get_conn().execute("INSERT OR REPLACE INTO top_picks (category,symbol,direction,price,entry_price,stop_loss,target,rr_ratio,shares,score,mode,details,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))",
            (category, data.get("symbol"), data.get("direction"), data.get("price"), data.get("entry") or data.get("entry_price"), data.get("stop_loss"), data.get("target_2r") or data.get("target"), data.get("rr_ratio"), data.get("shares"), data.get("score") or data.get("composite") or data.get("squeeze_score"), data.get("mode") or data.get("strategy"), json.dumps(data)))
        _get_conn().commit()
    @classmethod
    def get_all(cls):
        result = {}
        for row in _get_conn().execute("SELECT * FROM top_picks ORDER BY category").fetchall():
            r = dict(row)
            if r.get("details"):
                try: r["details"] = json.loads(r["details"])
                except: pass
            result[r["category"]] = r
        return result
    @classmethod
    def get(cls, category):
        row = _get_conn().execute("SELECT * FROM top_picks WHERE category=?", (category,)).fetchone()
        if row:
            r = dict(row)
            if r.get("details"):
                try: r["details"] = json.loads(r["details"])
                except: pass
            return r
        return None
    @classmethod
    def update_from_scan(cls, state, stock_signals=None, squeeze_data=None):
        top_sigs = state.get("top_signals", [])
        if stock_signals:
            tradeable = [s for s in stock_signals if s.get("tradeable")]
            if tradeable:
                best = max(tradeable, key=lambda x: abs(x.get("composite",0)) * x.get("strength",0))
                cls.update("best_stock", best)
        if top_sigs:
            with_opts = [s for s in top_sigs if s.get("options")]
            if with_opts:
                cls.update("best_options", {**max(with_opts, key=lambda x: abs(x.get("composite",0))), **(max(with_opts, key=lambda x: abs(x.get("composite",0))).get("options",{}))})
            swing_c = [s for s in top_sigs if s.get("direction") != "HOLD" and s.get("conviction",0) > 0.5]
            if swing_c:
                cls.update("best_swing", max(swing_c, key=lambda x: x.get("conviction",0)))

class TradeLog:
    @classmethod
    def open_trade(cls, symbol, direction, entry_price, shares, strategy="", signals=None):
        cur = _get_conn().execute("INSERT INTO trades (symbol,direction,entry_price,shares,strategy,signals_snapshot,status) VALUES (?,?,?,?,?,?,'open')",
            (symbol, direction, entry_price, shares, strategy, json.dumps(signals) if signals else None))
        _get_conn().commit(); return cur.lastrowid
    @classmethod
    def close_trade(cls, trade_id, exit_price, notes=""):
        t = _get_conn().execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone()
        if not t: return None
        pnl = (exit_price - t["entry_price"]) * t["shares"] if t["direction"]=="LONG" else (t["entry_price"] - exit_price) * t["shares"]
        pnl_pct = pnl / (t["entry_price"] * t["shares"]) * 100
        _get_conn().execute("UPDATE trades SET exit_price=?,exit_time=datetime('now'),pnl=?,pnl_pct=?,status='closed',notes=? WHERE id=?",
            (exit_price, round(pnl,2), round(pnl_pct,2), notes, trade_id))
        _get_conn().commit(); return {"pnl": round(pnl,2), "pnl_pct": round(pnl_pct,2)}
    @classmethod
    def get_open(cls):
        return [dict(r) for r in _get_conn().execute("SELECT * FROM trades WHERE status='open' ORDER BY entry_time DESC").fetchall()]
    @classmethod
    def get_closed(cls, limit=100):
        return [dict(r) for r in _get_conn().execute("SELECT * FROM trades WHERE status='closed' ORDER BY exit_time DESC LIMIT ?", (limit,)).fetchall()]
    @classmethod
    def get_stats(cls):
        closed = _get_conn().execute("SELECT pnl_pct FROM trades WHERE status='closed'").fetchall()
        if not closed: return {"total_trades":0,"win_rate":0,"avg_pnl":0}
        import numpy as np; arr = np.array([r["pnl_pct"] for r in closed])
        return {"total_trades":len(arr),"open_trades":_get_conn().execute("SELECT COUNT(*) FROM trades WHERE status='open'").fetchone()[0],
            "win_rate":round(float(np.mean(arr>0))*100,1),"avg_pnl_pct":round(float(np.mean(arr)),2),
            "total_pnl_pct":round(float(np.sum(arr)),2),"best_trade":round(float(np.max(arr)),2),
            "worst_trade":round(float(np.min(arr)),2),
            "profit_factor":round(float(np.sum(arr[arr>0])/max(abs(np.sum(arr[arr<0])),0.01)),2)}

class KVStore:
    @classmethod
    def set(cls, key, value):
        _get_conn().execute("INSERT OR REPLACE INTO kv_store (key,value,updated_at) VALUES (?,?,datetime('now'))", (key, json.dumps(value)))
        _get_conn().commit()
    @classmethod
    def get(cls, key, default=None):
        row = _get_conn().execute("SELECT value FROM kv_store WHERE key=?", (key,)).fetchone()
        if row:
            try: return json.loads(row["value"])
            except: return row["value"]
        return default

class MarketStore:
    @classmethod
    def save_quotes(cls, quotes: Dict[str, Dict], source: str = "scanner"):
        if not quotes:
            return
        conn = _get_conn()
        now = time.time()
        rows = []
        for sym, q in (quotes or {}).items():
            rows.append((
                sym,
                q.get("price"),
                q.get("changePct") if q.get("changePct") is not None else q.get("change_pct"),
                q.get("marketState") or q.get("market_state"),
                q.get("regularMarketPrice") or q.get("regular_price"),
                q.get("preMarketPrice") or q.get("pre_price"),
                q.get("postMarketPrice") or q.get("post_price"),
                q.get("provider") or source,
                now
            ))
        conn.executemany(
            "INSERT INTO market_ticks (symbol,price,change_pct,market_state,regular_price,pre_price,post_price,provider,ts_epoch) VALUES (?,?,?,?,?,?,?,?,?)",
            rows
        )
        conn.execute("DELETE FROM market_ticks WHERE ts_epoch < ?", (now - 14*24*3600,))
        conn.commit()

    @classmethod
    def save_news(cls, articles: List[Dict], source: str = "scanner"):
        if not articles:
            return
        conn = _get_conn()
        now = time.time()
        rows = []
        for a in articles[:200]:
            rows.append((
                a.get("title", ""),
                a.get("source") or source,
                a.get("sentiment"),
                a.get("url"),
                json.dumps(a.get("symbols") or []),
                now
            ))
        conn.executemany(
            "INSERT INTO news_snapshots (title,source,sentiment,url,symbols,ts_epoch) VALUES (?,?,?,?,?,?)",
            rows
        )
        conn.execute("DELETE FROM news_snapshots WHERE ts_epoch < ?", (now - 30*24*3600,))
        conn.commit()

    @classmethod
    def save_geo(cls, payload: Dict, source: str = "geo_live"):
        if not payload:
            return
        conn = _get_conn()
        now = time.time()
        dash = payload.get("dashboard") or {}
        risk = (dash.get("risk_score") or {}).get("composite_risk", 0)
        conn.execute(
            "INSERT INTO geo_snapshots (source,risk_score,conflicts_count,aircraft_count,earthquakes_count,fires_count,payload,ts_epoch) VALUES (?,?,?,?,?,?,?,?)",
            (
                source,
                risk,
                len((payload.get("conflicts") or {}).get("features", []) if isinstance(payload.get("conflicts"), dict) else []),
                len((payload.get("aircraft") or {}).get("states", []) if isinstance(payload.get("aircraft"), dict) else []),
                len((payload.get("earthquakes") or {}).get("features", []) if isinstance(payload.get("earthquakes"), dict) else []),
                len((payload.get("fires") or {}).get("events", []) if isinstance(payload.get("fires"), dict) else []),
                json.dumps(payload)[:500000],
                now
            )
        )
        conn.execute("DELETE FROM geo_snapshots WHERE ts_epoch < ?", (now - 30*24*3600,))
        conn.commit()

def setup():
    init_db(); Auth.setup_default_user(); return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO); setup()
    token = Auth.login("admin", "gig2026"); print(f"Login: {'OK' if token else 'FAIL'}")
    SignalStore.save_signals([{"symbol":"AAPL","direction":"LONG","composite":0.45,"price":175}])
    print(f"Signals: {len(SignalStore.get_latest(limit=5))}")
    TopPicks.update("best_stock", {"symbol":"NVDA","direction":"LONG","price":880,"composite":0.62})
    print(f"Picks: {list(TopPicks.get_all().keys())}")
    tid = TradeLog.open_trade("AAPL","LONG",175.0,50,"swing"); TradeLog.close_trade(tid,182.0)
    print(f"Stats: {TradeLog.get_stats()}")
    print(f"OK GIG Database: {DB_PATH}")
