from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .db import QueryError, QueryResult, ReadOnlyDB
from .guard import UnsafeSQL, check_sql


class ToolError(RuntimeError):
    pass


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, str]
    run: Callable[..., str]


def _cell(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def format_result(result: QueryResult) -> str:
    if not result.columns:
        return "(no result)"
    lines = [" | ".join(result.columns)]
    lines += [" | ".join(_cell(v) for v in row) for row in result.rows]
    if not result.rows:
        lines.append("(0 rows)")
    if result.truncated:
        lines.append(f"(truncated to the first {len(result.rows)} rows)")
    return "\n".join(lines)


def build_tools(db: ReadOnlyDB) -> dict[str, Tool]:
    def list_tables() -> str:
        return ", ".join(db.tables()) or "(no tables)"

    def describe_table(table: str) -> str:
        try:
            cols = db.columns(table)
            sample = db.query(f'SELECT * FROM "{table}" LIMIT 3')
            count = db.query(f'SELECT COUNT(*) FROM "{table}"').rows[0][0]
        except QueryError as error:
            raise ToolError(str(error)) from error
        header = ", ".join(f"{name} {kind}".strip() for name, kind in cols)
        return f"{table} ({count} rows)\ncolumns: {header}\nsample:\n{format_result(sample)}"

    def run_sql(query: str) -> str:
        try:
            return format_result(db.query(check_sql(query)))
        except (UnsafeSQL, QueryError) as error:
            raise ToolError(str(error)) from error

    tools = [
        Tool("list_tables", "List the tables in the database.", {}, list_tables),
        Tool(
            "describe_table",
            "Show the columns, row count and three sample rows of a table.",
            {"table": "name of the table"},
            describe_table,
        ),
        Tool(
            "run_sql",
            "Run one read-only SQLite SELECT query and return the rows (max 50).",
            {"query": "the SQL query"},
            run_sql,
        ),
    ]
    return {tool.name: tool for tool in tools}
