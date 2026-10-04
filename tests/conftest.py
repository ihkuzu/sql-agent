import pytest

from sqlagent.sampledata import build_sample_db
from sqlagent.db import ReadOnlyDB


class ScriptedLLM:
    """Returns prepared replies in order and records what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, system, messages):
        self.calls.append((system, [dict(m) for m in messages]))
        return self.replies.pop(0)


@pytest.fixture(scope="session")
def db_file(tmp_path_factory):
    return build_sample_db(tmp_path_factory.mktemp("data") / "shop.db")


@pytest.fixture()
def db(db_file):
    database = ReadOnlyDB(db_file)
    yield database
    database.close()
