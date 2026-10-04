# sql-agent

[![ci](https://github.com/ihkuzu/sql-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/ihkuzu/sql-agent/actions/workflows/ci.yml)

A small tool-using agent that answers questions about a SQLite database in plain
English. The model never sees the whole database. It decides which tool to call
(list tables, describe a table, run a query), reads the result and repeats until it
can answer.

I built it to show the parts of an agent that matter in practice: a tool loop, strict
limits on what the model may do, recovery from bad output, and a way to measure it.

```
question -> model -> {"tool": "run_sql", ...} -> guard -> read-only database
              ^                                                |
              +------------------ observation -----------------+
model -> {"final": "..."} -> answer
```

## How it works

- **Tools.** `list_tables`, `describe_table` (columns, row count, three sample rows)
  and `run_sql`. Each tool returns plain text, and every failure comes back as an
  observation so the model can correct itself instead of the program crashing.
- **Protocol.** The model replies with one JSON object per turn, either a tool call or
  `{"final": ...}`. Because it is plain JSON in plain text, the same loop works with
  any chat model. Replies wrapped in code fences or extra prose are still parsed.
- **Context.** The prompt already contains the schema, the foreign keys and the values of short text columns such as `status`, so the model does not spend its steps exploring.
- **Safety, in layers.** `run_sql` only accepts a single `SELECT` or `WITH` statement
  without comments (`guard.py`). The database is opened read-only at the SQLite level,
  so a query that slips past the guard still cannot write. Queries are stopped after a
  time limit and results are capped at 50 rows.
- **Loop control.** A step limit ends runs that go nowhere, and an identical repeated
  tool call is rejected with a hint to try something else. A final answer is only
  accepted after at least one successful query, because small models otherwise answer
  from memory.
- **Models.** Ollama (default `llama3.2:3b`, runs on your machine) or Gemini.

## Run it

```bash
pip install -e .
python -m sqlagent init-db                 # creates data/shop.db, a fictional online shop
ollama pull llama3.2:3b                    # or set SQLAGENT_LLM=gemini and GEMINI_API_KEY
python -m sqlagent ask "Which city has the most customers?" -v
```

`-v` prints every tool call. Other commands:

```bash
python -m sqlagent tools                   # what the model is allowed to do
python -m sqlagent eval                    # run the evaluation questions
python -m sqlagent --db other.db ask "..." # use your own SQLite file
```

Settings: `SQLAGENT_LLM` (`ollama` or `gemini`), `SQLAGENT_MODEL`, `SQLAGENT_DB`,
`OLLAMA_HOST`, `GEMINI_API_KEY`. Empty values count as unset.

With Docker (the container reaches Ollama on the host):

```bash
docker build -t sql-agent .
docker run --rm -e OLLAMA_HOST=host.docker.internal:11434 -v "$PWD/data:/app/data" sql-agent init-db
docker run --rm -e OLLAMA_HOST=host.docker.internal:11434 -v "$PWD/data:/app/data" sql-agent ask "How many orders were cancelled?"
```

## The sample data

`init-db` builds a fictional shop with 60 customers, 18 products, 400 orders and their
items. It is generated from a fixed seed, so everyone gets the same database and the
same correct answers.

## Evaluation

`data/eval/questions.json` holds 12 questions. Each one has a reference SQL query that
produces the expected answer. `python -m sqlagent eval` lets the agent answer every
question, then checks that all values from the reference result appear in the answer
(numbers are matched exactly, so `600` does not count for `60`). It reports accuracy,
unfinished runs and the mean number of tool calls.

The questions range from single counts to joins and aggregation, for example
"Which product category earned the most revenue from delivered orders?".

Results with `llama3.2:3b` on a laptop (RTX 2060), one run each:

| version of the agent                       | correct | accuracy |
| ------------------------------------------ | ------- | -------- |
| first version                              | 2/12    | 0.17     |
| + final answer only after a query          | 3/12    | 0.25     |
| + schema and column values in the prompt   | 8/12    | 0.67     |

The first run showed the main failure: the small model answered from memory without
touching the database (several runs with zero tool calls). Requiring a successful
query before the final answer fixed that. The remaining mistakes were wrong filters and
wasted steps on exploring the schema, so the schema now goes into the prompt up front.

What still fails: a wrong date filter (answering "no orders in 2025"), a category
question answered with a product instead of a category, and two questions that run out of steps. A 3B
model is weak at multi-step SQL.

Two caveats. The prompt was tuned while looking at these same 12 questions, so 0.67 is
optimistic for new questions. And 12 questions from one run is a small sample, so
treat differences of one or two questions as noise.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

72 tests run without a model. A scripted model replays prepared replies, which makes
the loop deterministic: the normal path, invalid JSON, unknown tools, wrong arguments,
blocked SQL, SQL errors, repeated calls and the step limit are all covered. The guard,
the read-only connection and the query timeout have their own tests, and the model
clients are tested against mocked HTTP responses.

## Limits

- It works on one SQLite file. Other databases would need their own read-only
  connection and schema lookup.
- The guard is a keyword filter on top of the read-only connection. It can reject a
  harmless query that contains a word like `update` inside a string.
- Small local models make more mistakes with joins. The step limit and the error
  feedback help, but they do not replace a stronger model.
- The Gemini client is tested against mocked responses and retries when the API is overloaded or fails to parse its own JSON.

## Roadmap

- [ ] Compare a larger model (7B) and a hosted model on the same questions
- [ ] Evaluate on questions that were not used to tune the prompt
- [ ] Let the model ask for a chart of a result
- [ ] Add a second database for a multi-database question

## License

MIT, see [LICENSE](LICENSE).
