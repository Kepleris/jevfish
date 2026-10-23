"""Caché en SQLite: repetir una simulación no vuelve a llamar a la API."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


class Cache:
    def __init__(self, path: str | Path | None):
        self.conn = None
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(path)
            self.conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")

    @staticmethod
    def key(*parts) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def get(self, key: str):
        if not self.conn:
            return None
        row = self.conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value) -> None:
        if self.conn:
            self.conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(value)))
            self.conn.commit()

    def close(self) -> None:
        if self.conn:
            self.conn.close()
