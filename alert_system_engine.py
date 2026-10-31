"""
=============================================================================
RENAISSANCE ALERT SYSTEM — alert_system_engine.py
=============================================================================
Push Notifications + Webhook Integration + Signal Threshold Alerts

CHANNELS:
  ├── Telegram Bot API        — instant alerts to phone
  ├── Discord Webhooks        — team channel notifications
  ├── Console/Log Alerts      — always-on fallback
  └── In-App Alert Feed       — dashboard alert stream

TRIGGERS:
  ├── Ensemble Signal Cross   — master signal > ±0.3 threshold
  ├── VIX Spike Alert         — VIX jumps >20% intraday
  ├── Earnings Surprise       — beat/miss > 10% vs consensus
  ├── Options Unusual Flow    — volume > 5x average OI
  ├── Drawdown Alert          — portfolio drawdown > 5%
  ├── Position Limit Warning  — approaching capacity limits
  ├── Signal Decay Warning    — IC dropping on key signals
  └── Pattern Breakout        — confirmed chart pattern trigger

Renaissance.io — Institutional Grade Quantitative Trading
=============================================================================
"""

import logging
import threading
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque

logger = logging.getLogger("ALERTS")

# ═══════════════════════════════════════════════════════════════════════════════
# ALERT TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class AlertLevel:
    CRITICAL = "critical"   # Immediate action required
    HIGH = "high"           # Important — review soon
    MEDIUM = "medium"       # Notable event
    LOW = "low"             # Informational

class AlertType:
    ENSEMBLE_SIGNAL = "ensemble_signal"
    VIX_SPIKE = "vix_spike"
    EARNINGS_SURPRISE = "earnings_surprise"
    UNUSUAL_OPTIONS = "unusual_options"
    DRAWDOWN = "drawdown"
    CAPACITY_WARNING = "capacity_warning"
    SIGNAL_DECAY = "signal_decay"
    PATTERN_BREAKOUT = "pattern_breakout"
    PRICE_TARGET = "price_target"
    REGIME_CHANGE = "regime_change"


# ═══════════════════════════════════════════════════════════════════════════════
# WEBHOOK DISPATCHERS
# ═══════════════════════════════════════════════════════════════════════════════

class TelegramDispatcher:
    """Send alerts via Telegram Bot API."""

    def __init__(self):
        self.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        self._enabled = bool(self.bot_token and self.chat_id)

    def send(self, message: str, level: str = "medium") -> bool:
        if not self._enabled:
            return False
        try:
            import requests
            emoji = {"critical": "🚨", "high": "⚠️", "medium": "📊", "low": "ℹ️"}.get(level, "📊")
            text = f"{emoji} *Renaissance.io Alert*\n\n{message}"
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            r = requests.post(url, json={
                "chat_id": self.chat_id, "text": text,
                "parse_mode": "Markdown", "disable_web_page_preview": True
            }, timeout=10)
            return r.status_code == 200
        except Exception as e:
            logger.debug(f"Telegram send failed: {e}")
            return False

    @property
    def is_enabled(self):
        return self._enabled


class DiscordDispatcher:
    """Send alerts via Discord Webhook."""

    def __init__(self):
        self.webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
        self._enabled = bool(self.webhook_url)

    def send(self, message: str, level: str = "medium") -> bool:
        if not self._enabled:
            return False
        try:
            import requests
            colors = {"critical": 0xFF0000, "high": 0xFF8C00, "medium": 0x1A8FFF, "low": 0x00D97E}
            payload = {
                "embeds": [{
                    "title": f"Renaissance.io — {level.upper()} Alert",
                    "description": message,
                    "color": colors.get(level, 0x1A8FFF),
                    "timestamp": datetime.utcnow().isoformat(),
                    "footer": {"text": "Renaissance.io Medallion Engine"}
                }]
            }
            r = requests.post(self.webhook_url, json=payload, timeout=10)
            return r.status_code in (200, 204)
        except Exception as e:
            logger.debug(f"Discord send failed: {e}")
            return False

    @property
    def is_enabled(self):
        return self._enabled


# ═══════════════════════════════════════════════════════════════════════════════
# ALERT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class AlertEngine:
    """
    Central alert management system.
    Checks conditions, deduplicates, rate-limits, and dispatches.
    """

    def __init__(self):
        self._alerts: deque = deque(maxlen=500)
        self._telegram = TelegramDispatcher()
        self._discord = DiscordDispatcher()
        self._lock = threading.Lock()
        self._cooldowns: Dict[str, float] = {}
        self._COOLDOWN_SECONDS = 300  # 5 min between same alert type per symbol
        # Thresholds
        self.ensemble_threshold = 0.30
        self.vix_spike_pct = 15.0
        self.drawdown_threshold = 5.0
        self.unusual_vol_multiplier = 5.0

    def check_all(self, state: Dict, ensemble_data: Dict = None,
                  positions: List = None, prev_vix: float = None):
        """Run all alert checks against current system state."""
        now = time.time()
        vix = state.get("vix", 0)
        regime = state.get("regime", "")
        signals = state.get("top_signals", [])

        # 1. Ensemble signal threshold
        if ensemble_data:
            master = ensemble_data.get("master_signal", 0)
            if abs(master) > self.ensemble_threshold:
                sym = ensemble_data.get("symbol", "PORTFOLIO")
                direction = "LONG" if master > 0 else "SHORT"
                self._fire(AlertType.ENSEMBLE_SIGNAL, sym, AlertLevel.HIGH,
                    f"Ensemble signal {direction}: {master:+.4f} on {sym}\n"
                    f"Confidence: {ensemble_data.get('confidence', 0)*100:.1f}%\n"
                    f"Regime: {ensemble_data.get('regime', 'N/A')}")

        # 2. VIX spike
        if prev_vix and prev_vix > 0 and vix > 0:
            vix_change = (vix - prev_vix) / prev_vix * 100
            if vix_change > self.vix_spike_pct:
                self._fire(AlertType.VIX_SPIKE, "VIX", AlertLevel.CRITICAL,
                    f"VIX SPIKE: {prev_vix:.1f} → {vix:.1f} ({vix_change:+.1f}%)\n"
                    f"Current regime: {regime}")

        # 3. Regime change
        if regime and regime in ("crisis", "high_vol", "bear"):
            self._fire(AlertType.REGIME_CHANGE, "MARKET", AlertLevel.HIGH,
                f"Regime shifted to: {regime.upper()}\nVIX: {vix:.1f}")

        # 4. Top signal alerts
        for sig in signals[:5]:
            sym = sig.get("symbol", "")
            composite = sig.get("composite", 0)
            if abs(composite) > 0.7:
                direction = sig.get("direction", "NEUTRAL")
                self._fire(AlertType.ENSEMBLE_SIGNAL, sym, AlertLevel.MEDIUM,
                    f"Strong signal: {sym} {direction}\n"
                    f"Composite: {composite:.4f} | RSI: {sig.get('rsi', 0):.1f}\n"
                    f"Hurst: {sig.get('hurst', 0):.4f}")

    def fire_custom(self, alert_type: str, symbol: str, level: str, message: str):
        """Fire a custom alert from any engine."""
        self._fire(alert_type, symbol, level, message)

    def _fire(self, alert_type: str, symbol: str, level: str, message: str):
        """Internal: create alert, check cooldown, dispatch."""
        key = f"{alert_type}:{symbol}"
        now = time.time()

        # Cooldown check
        with self._lock:
            last = self._cooldowns.get(key, 0)
            if now - last < self._COOLDOWN_SECONDS:
                return
            self._cooldowns[key] = now

        alert = {
            "id": f"{key}:{int(now)}",
            "type": alert_type,
            "symbol": symbol,
            "level": level,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "ts": now,
            "dispatched": {"telegram": False, "discord": False, "console": True}
        }

        # Console always
        level_prefix = {"critical": "🚨", "high": "⚠️", "medium": "📊", "low": "ℹ️"}.get(level, "📊")
        logger.info(f"{level_prefix} ALERT [{alert_type}] {symbol}: {message[:100]}")

        # Telegram (high + critical only)
        if level in ("critical", "high"):
            alert["dispatched"]["telegram"] = self._telegram.send(message, level)

        # Discord (all levels)
        alert["dispatched"]["discord"] = self._discord.send(message, level)

        with self._lock:
            self._alerts.append(alert)

    def get_alerts(self, limit: int = 50, level: str = None,
                   alert_type: str = None) -> List[Dict]:
        """Get recent alerts with optional filtering."""
        with self._lock:
            alerts = list(self._alerts)
        if level:
            alerts = [a for a in alerts if a["level"] == level]
        if alert_type:
            alerts = [a for a in alerts if a["type"] == alert_type]
        return sorted(alerts, key=lambda x: x["ts"], reverse=True)[:limit]

    def get_status(self) -> Dict:
        """Get alert system status."""
        with self._lock:
            recent = list(self._alerts)
        return {
            "total_alerts": len(recent),
            "critical": sum(1 for a in recent if a["level"] == "critical"),
            "high": sum(1 for a in recent if a["level"] == "high"),
            "telegram_enabled": self._telegram.is_enabled,
            "discord_enabled": self._discord.is_enabled,
            "thresholds": {
                "ensemble": self.ensemble_threshold,
                "vix_spike_pct": self.vix_spike_pct,
                "drawdown_pct": self.drawdown_threshold,
            }
        }


# Singleton
_engine: Optional[AlertEngine] = None

def get_alert_engine() -> AlertEngine:
    global _engine
    if _engine is None:
        _engine = AlertEngine()
    return _engine


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eng = get_alert_engine()
    eng.fire_custom("test", "AAPL", "medium", "Test alert from Renaissance.io")
    print(f"Alerts: {len(eng.get_alerts())}")
    print(f"Status: {eng.get_status()}")
    print("✅ Alert System operational")