"""
messaging_collab_engine.py
=============================================================================
Institutional messaging/collaboration engine (Bloomberg-chat style baseline):
- Channels (desk, strategy, symbol-linked)
- Threaded messages
- Presence heartbeat
- Lightweight persistence (JSON) + append-only semantics for messages
=============================================================================
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MessagingCollabEngine:
    def __init__(self, storage_dir: str = "logs") -> None:
        self._storage_dir = storage_dir
        self._channels_file = os.path.join(storage_dir, "chat_channels.json")
        self._messages_file = os.path.join(storage_dir, "chat_messages.jsonl")
        self._presence_file = os.path.join(storage_dir, "chat_presence.json")
        self._lock = threading.Lock()
        os.makedirs(storage_dir, exist_ok=True)
        self._ensure_files()

    def _ensure_files(self) -> None:
        if not os.path.exists(self._channels_file):
            bootstrap = {
                "channels": [
                    {
                        "channel_id": "desk-global",
                        "name": "Desk Global",
                        "type": "desk",
                        "symbol": None,
                        "created_at_utc": _utc_now(),
                    },
                    {
                        "channel_id": "strategy-stat-arb",
                        "name": "Strategy StatArb",
                        "type": "strategy",
                        "symbol": None,
                        "created_at_utc": _utc_now(),
                    },
                ]
            }
            with open(self._channels_file, "w", encoding="utf-8") as f:
                json.dump(bootstrap, f)
        if not os.path.exists(self._presence_file):
            with open(self._presence_file, "w", encoding="utf-8") as f:
                json.dump({"users": {}}, f)
        if not os.path.exists(self._messages_file):
            with open(self._messages_file, "a", encoding="utf-8"):
                pass

    def _read_channels(self) -> Dict:
        with open(self._channels_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_channels(self, payload: Dict) -> None:
        with open(self._channels_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _read_presence(self) -> Dict:
        with open(self._presence_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_presence(self, payload: Dict) -> None:
        with open(self._presence_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def list_channels(self) -> List[Dict]:
        with self._lock:
            return self._read_channels().get("channels", [])

    def create_channel(self, name: str, channel_type: str = "desk", symbol: Optional[str] = None) -> Dict:
        channel = {
            "channel_id": f"ch-{uuid.uuid4().hex[:12]}",
            "name": name.strip(),
            "type": channel_type.strip().lower(),
            "symbol": symbol.upper() if symbol else None,
            "created_at_utc": _utc_now(),
        }
        with self._lock:
            payload = self._read_channels()
            payload.setdefault("channels", []).append(channel)
            self._write_channels(payload)
        return channel

    def post_message(self, channel_id: str, user: str, text: str, symbol: Optional[str] = None) -> Dict:
        msg = {
            "message_id": f"msg-{uuid.uuid4().hex}",
            "channel_id": channel_id,
            "user": user.strip() or "anonymous",
            "text": text.strip(),
            "symbol": symbol.upper() if symbol else None,
            "timestamp_utc": _utc_now(),
        }
        with self._lock:
            with open(self._messages_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(msg, ensure_ascii=True) + "\n")
        return msg

    def list_messages(self, channel_id: str, limit: int = 100) -> List[Dict]:
        limit = max(1, min(int(limit), 1000))
        rows: List[Dict] = []
        with self._lock:
            with open(self._messages_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if row.get("channel_id") == channel_id:
                        rows.append(row)
        return rows[-limit:]

    def heartbeat(self, user: str, status: str = "online") -> Dict:
        with self._lock:
            payload = self._read_presence()
            payload.setdefault("users", {})[user] = {
                "status": status,
                "last_seen_utc": _utc_now(),
            }
            self._write_presence(payload)
            return payload["users"][user]

    def list_presence(self) -> Dict:
        with self._lock:
            return self._read_presence().get("users", {})


_engine: Optional[MessagingCollabEngine] = None


def get_messaging_engine() -> MessagingCollabEngine:
    global _engine
    if _engine is None:
        _engine = MessagingCollabEngine()
    return _engine

