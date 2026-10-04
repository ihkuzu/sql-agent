import json

import pytest

from sqlagent.agent import Agent
from sqlagent.evaluation import Case, answer_matches, evaluate, format_report, load_cases


@pytest.mark.parametrize(
    "answer, rows, expected",
    [
        ("There are 60 customers.", [(60,)], True),
        ("There are 600 customers.", [(60,)], False),
        ("Revenue is 406,391.00 EUR", [(406391.0,)], True),
        ("Revenue is 406391", [(406391.0,)], True),
        ("Revenue is 406391.5", [(406391.0,)], False),
        ("The price is 1299.00", [("Laptop 16", 1299.0)], False),
        ("Laptop 16 costs 1299.00", [("Laptop 16", 1299.0)], True),
        ("It is Munich.", [("Munich",)], True),
        ("keyboard and headphones", [("Mechanical Keyboard",), ("Headphones",)], False),
        ("Mechanical Keyboard and Headphones", [("Mechanical Keyboard",), ("Headphones",)], True),
        ("avg is 12.5", [(12.5,)], True),
        ("anything", [(None,)], True),
    ],
)
def test_answer_matches(answer, rows, expected):
    assert answer_matches(answer, rows) is expected


class QuestionAwareLLM:
    # answers the count question correctly and the city question wrongly
    def chat(self, system, messages):
        question = messages[0]["content"]
        if len(messages) == 1:
            return json.dumps({"tool": "run_sql", "args": {"query": "SELECT 1"}})
        return json.dumps({"final": "60" if "How many" in question else "Atlantis"})


def test_evaluate_and_report(db):
    cases = [
        Case("a", "How many customers are there?", "SELECT COUNT(*) FROM customers"),
        Case("b", "Which city has the most customers?", "SELECT 'Munich'"),
    ]
    outcomes = evaluate(Agent(QuestionAwareLLM(), db), db, cases)
    assert [o.correct for o in outcomes] == [True, False]
    report = format_report(outcomes)
    assert "accuracy 1/2 = 0.50" in report
    assert report.splitlines()[0].startswith("ok")
    assert report.splitlines()[1].startswith("FAIL")
    assert "answer: Atlantis" in report.splitlines()[2]


def test_shipped_cases_run_against_the_sample_db(db):
    cases = load_cases("data/eval/questions.json")
    assert len(cases) == 12
    for case in cases:
        rows = db.query(case.sql, max_rows=1000).rows
        assert rows, case.id
        assert answer_matches(" ".join(str(v) for row in rows for v in row), rows)


class BrokenLLM:
    def chat(self, system, messages):
        from sqlagent.llm import LLMError

        raise LLMError("boom")


def test_model_error_fails_one_case_but_keeps_the_run_going(db):
    cases = [Case("a", "q1", "SELECT 1"), Case("b", "q2", "SELECT 1")]
    outcomes = evaluate(Agent(BrokenLLM(), db), db, cases)
    assert [o.correct for o in outcomes] == [False, False]
    assert "model error: boom" in outcomes[0].answer
