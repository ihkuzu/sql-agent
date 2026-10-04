import sqlite3

from sqlagent.cli import main
from sqlagent.sampledata import build_sample_db


def dump(path):
    conn = sqlite3.connect(path)
    data = [conn.execute(f"SELECT * FROM {t} ORDER BY 1, 2").fetchall()
            for t in ("customers", "products", "orders", "order_items")]
    conn.close()
    return data


def test_sample_db_is_deterministic(tmp_path):
    first = dump(build_sample_db(tmp_path / "a.db", seed=1))
    second = dump(build_sample_db(tmp_path / "b.db", seed=1))
    other = dump(build_sample_db(tmp_path / "c.db", seed=2))
    assert first == second and first != other


def test_sample_db_sizes(tmp_path):
    customers, products, orders, items = dump(build_sample_db(tmp_path / "a.db"))
    assert (len(customers), len(products), len(orders)) == (60, 18, 400)
    assert len(items) > 400


def test_cli_init_tools_and_missing_db(tmp_path, capsys):
    path = str(tmp_path / "shop.db")
    assert main(["--db", path, "tools"]) == 2
    assert "init-db" in capsys.readouterr().err
    assert main(["--db", path, "init-db"]) == 0
    assert main(["--db", path, "tools"]) == 0
    assert "run_sql" in capsys.readouterr().out
