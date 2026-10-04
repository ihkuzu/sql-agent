import json

from conftest import ScriptedLLM
from sqlagent.agent import Agent, parse_action


def tool(name, **args):
    return json.dumps({"tool": name, "args": args})


def final(text):
    return json.dumps({"final": text})


def test_parse_action_variants():
    assert parse_action('{"final": "42"}') == {"final": "42"}
    assert parse_action('```json\n{"tool": "list_tables"}\n```') == {"tool": "list_tables", "args": {}}
    assert parse_action('Sure! {"tool": "run_sql", "args": {"query": "SELECT 1"}} done')["tool"] == "run_sql"


def test_parse_action_rejects_garbage():
    import pytest
    from sqlagent.agent import ActionError

    for bad in ["no json here", '{"tool": ', "[1, 2]", '{"x": 1}', '{"tool": "a", "args": 3}']:
        with pytest.raises(ActionError):
            parse_action(bad)


def test_happy_path_uses_tools_in_order(db):
    llm = ScriptedLLM([
        tool("list_tables"),
        tool("describe_table", table="customers"),
        tool("run_sql", query="SELECT COUNT(*) FROM customers"),
        final("There are 60 customers."),
    ])
    result = Agent(llm, db, require_query=False).run("How many customers are there?")
    assert result.finished and result.answer == "There are 60 customers."
    assert [s.tool for s in result.steps] == ["list_tables", "describe_table", "run_sql"]
    assert result.steps[2].observation.splitlines()[1] == "60"


def test_history_is_sent_to_the_model(db):
    llm = ScriptedLLM([tool("list_tables"), final("done")])
    Agent(llm, db, require_query=False).run("question?")
    assert [len(m) for _, m in llm.calls] == [1, 3]
    assert llm.calls[1][1][2]["content"].startswith("Observation: customers")


def test_system_prompt_lists_tools(db):
    llm = ScriptedLLM([final("x")])
    Agent(llm, db, require_query=False).run("q")
    system = llm.calls[0][0]
    for name in ("list_tables", "describe_table", "run_sql"):
        assert name in system
    assert "orders(id INTEGER" in system
    assert "orders.customer_id -> customers.id" in system


def test_invalid_reply_is_reported_and_recovered(db):
    llm = ScriptedLLM(["I think the answer is 60", final("60")])
    result = Agent(llm, db, require_query=False).run("q")
    assert result.finished
    assert result.steps[0].error and "JSON" in result.steps[0].observation


def test_unknown_tool_and_bad_arguments(db):
    llm = ScriptedLLM([tool("delete_everything"), tool("run_sql", sql="SELECT 1"), final("ok")])
    result = Agent(llm, db, require_query=False).run("q")
    assert "unknown tool" in result.steps[0].observation
    assert "Expected: query" in result.steps[1].observation


def test_blocked_sql_is_an_observation_not_a_crash(db):
    llm = ScriptedLLM([tool("run_sql", query="DROP TABLE orders"), tool("run_sql", query="SELECT COUNT(*) FROM orders"), final("400")])
    result = Agent(llm, db, require_query=False).run("q")
    assert result.steps[0].error and "only SELECT" in result.steps[0].observation
    assert not result.steps[1].error
    assert db.query("SELECT COUNT(*) FROM orders").rows == [(400,)]


def test_sql_error_goes_back_to_the_model(db):
    llm = ScriptedLLM([tool("run_sql", query="SELECT nope FROM orders"), final("could not")])
    result = Agent(llm, db, require_query=False).run("q")
    assert result.steps[0].error and "no such column" in result.steps[0].observation


def test_repeated_call_is_flagged(db):
    call = tool("list_tables")
    llm = ScriptedLLM([call, call, final("done")])
    result = Agent(llm, db, require_query=False).run("q")
    assert not result.steps[0].error
    assert result.steps[1].error and "already ran" in result.steps[1].observation


def test_step_limit(db):
    llm = ScriptedLLM([tool("run_sql", query=f"SELECT {i}") for i in range(10)])
    result = Agent(llm, db, max_steps=3, require_query=False).run("q")
    assert not result.finished and len(result.steps) == 3
    assert "3 steps" in result.answer


def test_on_step_callback(db):
    seen = []
    llm = ScriptedLLM([tool("list_tables"), final("x")])
    Agent(llm, db, require_query=False, on_step=lambda n, s: seen.append((n, s.tool))).run("q")
    assert seen == [(1, "list_tables")]


def test_final_answer_without_a_query_is_rejected(db):
    llm = ScriptedLLM([
        final("There are 5 customers."),
        tool("run_sql", query="SELECT COUNT(*) FROM customers"),
        final("There are 60 customers."),
    ])
    result = Agent(llm, db).run("How many customers are there?")
    assert result.answer == "There are 60 customers."
    assert result.steps[0].error and "not run a successful SQL query" in result.steps[0].observation
    assert result.steps[1].tool == "run_sql"


def test_failed_query_does_not_unlock_the_final_answer(db):
    llm = ScriptedLLM([tool("run_sql", query="SELECT nope"), final("x"), tool("run_sql", query="SELECT 1"), final("1")])
    result = Agent(llm, db).run("q")
    assert result.answer == "1" and len(result.steps) == 3
