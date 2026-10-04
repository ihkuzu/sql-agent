from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


class QueryError(RuntimeError):
    pass


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool


class ReadOnlyDB:
    def __init__(self, path: str | Path, timeout: float = 5.0):
        file = Path(path)
        if not file.exists():
            raise FileNotFoundError(f"database not found: {file} (run 'init-db' first)")
        self.timeout = timeout
        self._deadline = float("inf")
        self._conn = sqlite3.connect(f"{file.resolve().as_uri()}?mode=ro", uri=True)
        # the handler runs every 1000 VM steps and aborts queries past the deadline
        self._conn.set_progress_handler(self._expired, 1000)

    def _expired(self) -> int:
        return 1 if time.monotonic() > self._deadline else 0

    def close(self) -> None:
        self._conn.close()

    def tables(self) -> list[str]:
        cur = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        return [row[0] for row in cur.fetchall()]

    def columns(self, table: str) -> list[tuple[str, str]]:
        if table not in self.tables():
            raise QueryError(f"unknown table '{table}'")
        cur = self._conn.execute(f'PRAGMA table_info("{table}")')
        return [(row[1], row[2] or "") for row in cur.fetchall()]

    def query(self, sql: str, max_rows: int = 50) -> QueryResult:
        self._deadline = time.monotonic() + self.timeout
        try:
            cur = self._conn.execute(sql)
            rows = cur.fetchmany(max_rows + 1)
            columns = [d[0] for d in cur.description or []]
        except sqlite3.OperationalError as error:
            if "interrupted" in str(error):
                raise QueryError(f"query took longer than {self.timeout:g}s and was stopped") from error
            raise QueryError(str(error)) from error
        except sqlite3.Error as error:
            raise QueryError(str(error)) from error
        finally:
            self._deadline = float("inf")
        return QueryResult(columns, rows[:max_rows], len(rows) > max_rows)
