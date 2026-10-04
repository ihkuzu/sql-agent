import pytest

from sqlagent.tools import ToolError, build_tools


def test_list_tables(db):
    assert build_tools(db)["list_tables"].run() == "customers, order_items, orders, products"


def test_describe_table_shows_count_and_sample(db):
    text = build_tools(db)["describe_table"].run(table="products")
    assert "products (18 rows)" in text
    assert "price REAL" in text
    assert text.count("\n") >= 5


def test_describe_unknown_table(db):
    with pytest.raises(ToolError):
        build_tools(db)["describe_table"].run(table="nope")


def test_run_sql_formats_rows(db):
    text = build_tools(db)["run_sql"].run(query="SELECT name, price FROM products ORDER BY price DESC LIMIT 2")
    assert text.splitlines() == ["name | price", "Laptop 16 | 1299.00", "Laptop 14 | 899.00"]


def test_run_sql_rejects_writes(db):
    with pytest.raises(ToolError, match="not allowed|only SELECT"):
        build_tools(db)["run_sql"].run(query="DROP TABLE orders")


def test_run_sql_truncation_note(db):
    text = build_tools(db)["run_sql"].run(query="SELECT * FROM orders")
    assert text.endswith("(truncated to the first 50 rows)")


def test_empty_result(db):
    text = build_tools(db)["run_sql"].run(query="SELECT * FROM orders WHERE id < 0")
    assert "(0 rows)" in text
