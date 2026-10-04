import pytest

from sqlagent.guard import UnsafeSQL, check_sql


def test_plain_select_passes():
    assert check_sql("SELECT * FROM customers;") == "SELECT * FROM customers"


def test_with_clause_passes():
    assert check_sql("with a as (select 1) select * from a").startswith("with")


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "DELETE FROM orders",
        "DROP TABLE orders",
        "INSERT INTO products VALUES (1)",
        "SELECT 1; DROP TABLE orders",
        "SELECT 1 -- hidden",
        "SELECT /* x */ 1",
        "PRAGMA table_info(orders)",
        "ATTACH DATABASE 'x.db' AS x",
        "WITH x AS (SELECT 1) DELETE FROM orders",
    ],
)
def test_unsafe_queries_are_rejected(sql):
    with pytest.raises(UnsafeSQL):
        check_sql(sql)


def test_replace_function_is_allowed():
    assert check_sql("SELECT replace(name, 'a', 'b') FROM customers")
