"""SQLite key/value cache for stage outputs (e.g. retrieval), shared by threads of a batch run."""
import hashlib
import json
import sqlite3
import threading
from pathlib import Path


def cache_key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


class Cache:
    def __init__(self, path: str | Path):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self._db.commit()

    def get(self, key: str):
        with self._lock:
            row = self._db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def set(self, key: str, value) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, json.dumps(value)))
            self._db.commit()
