import sqlite3

import pytest

from sqlagent.db import QueryError, ReadOnlyDB


def test_tables_and_columns(db):
    assert db.tables() == ["customers", "order_items", "orders", "products"]
    assert ("price", "REAL") in db.columns("products")


def test_unknown_table(db):
    with pytest.raises(QueryError):
        db.columns("nope")


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ReadOnlyDB(tmp_path / "missing.db")


def test_connection_is_read_only(db_file):
    # bypass the guard on purpose, the connection itself must refuse writes
    database = ReadOnlyDB(db_file)
    with pytest.raises(QueryError):
        database.query("DELETE FROM orders")
    database.close()
    raw = sqlite3.connect(db_file)
    assert raw.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 400
    raw.close()


def test_row_cap_marks_truncation(db):
    result = db.query("SELECT * FROM orders", max_rows=10)
    assert len(result.rows) == 10 and result.truncated
    small = db.query("SELECT * FROM products", max_rows=100)
    assert len(small.rows) == 18 and not small.truncated


def test_runaway_query_is_stopped(db_file):
    database = ReadOnlyDB(db_file, timeout=0.3)
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
    with pytest.raises(QueryError, match="stopped"):
        database.query(sql)
    # the connection still works afterwards
    assert database.query("SELECT COUNT(*) FROM products").rows == [(18,)]
    database.close()


def test_bad_sql_gives_query_error(db):
    with pytest.raises(QueryError):
        db.query("SELECT nope FROM customers")
