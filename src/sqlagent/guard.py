from __future__ import annotations

import re


class UnsafeSQL(ValueError):
    pass


_START = re.compile(r"(select|with)\b", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|reindex)\b"
    r"|\breplace\s+into\b",
    re.IGNORECASE,
)


def check_sql(sql: str) -> str:
    """Return the cleaned query or raise UnsafeSQL."""
    text = sql.strip().rstrip(";").strip()
    if not text:
        raise UnsafeSQL("the query is empty")
    if ";" in text:
        raise UnsafeSQL("only one statement is allowed")
    if "--" in text or "/*" in text:
        raise UnsafeSQL("comments are not allowed in queries")
    if not _START.match(text):
        raise UnsafeSQL("only SELECT queries are allowed")
    match = _FORBIDDEN.search(text)
    if match:
        raise UnsafeSQL(f"the keyword '{match.group(0).strip()}' is not allowed")
    return text
