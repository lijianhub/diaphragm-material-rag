"""Agent runs in SQLite (standard library), so a run paused for approval survives a
restart and can be resumed by a different request.

``claim`` is a compare-and-set on the status: of two reviewers approving the same run at
the same moment, exactly one wins, and the action runs once.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import List, Optional, Union


class RunStore:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, status TEXT NOT NULL, created TEXT NOT NULL, updated TEXT NOT NULL, data TEXT NOT NULL)")
            db.execute("CREATE INDEX IF NOT EXISTS runs_status ON runs (status, updated)")

    def _connect(self) -> sqlite3.Connection:
        # One connection per operation: safe across the API's worker threads.
        return sqlite3.connect(self.path, timeout=10)

    def save(self, run: dict) -> None:
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO runs (id, status, created, updated, data) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET status = excluded.status, updated = excluded.updated, data = excluded.data",
                (run["id"], run["status"], run["created"], run["updated"], json.dumps(run, ensure_ascii=False)),
            )

    def get(self, run_id: str) -> Optional[dict]:
        with closing(self._connect()) as db:
            row = db.execute("SELECT data FROM runs WHERE id = ?", (run_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def claim(self, run_id: str, expected: str, new: str) -> bool:
        """Set the status to ``new`` only if it is still ``expected``; True if this call won."""
        with closing(self._connect()) as db, db:
            changed = db.execute("UPDATE runs SET status = ? WHERE id = ? AND status = ?", (new, run_id, expected)).rowcount
        return changed == 1

    def list(self, limit: int = 50, status: Optional[str] = None) -> List[dict]:
        query, args = "SELECT data FROM runs", []
        if status:
            query += " WHERE status = ?"
            args.append(status)
        query += " ORDER BY updated DESC LIMIT ?"
        args.append(limit)
        with closing(self._connect()) as db:
            return [json.loads(row[0]) for row in db.execute(query, args).fetchall()]
