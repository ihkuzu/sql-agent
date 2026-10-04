from __future__ import annotations

import argparse
import os
import sys

from .agent import Agent, Step
from .db import ReadOnlyDB
from .evaluation import evaluate, format_report, load_cases
from .llm import LLMError, get_llm
from .sampledata import build_sample_db

DEFAULT_DB = "data/shop.db"
DEFAULT_CASES = "data/eval/questions.json"


def _db_path(args) -> str:
    return args.db or os.getenv("SQLAGENT_DB") or DEFAULT_DB


def _print_step(number: int, step: Step) -> None:
    args = ", ".join(f"{k}={v!r}" for k, v in step.args.items())
    first = step.observation.splitlines()[0] if step.observation else ""
    flag = " [error]" if step.error else ""
    print(f"  {number}. {step.tool}({args}){flag}\n     {first[:160]}")


def cmd_init_db(args) -> int:
    path = build_sample_db(_db_path(args))
    print(f"created {path}")
    return 0


def cmd_ask(args) -> int:
    db = ReadOnlyDB(_db_path(args))
    agent = Agent(get_llm(args.llm, args.model), db, args.max_steps,
                  on_step=_print_step if args.verbose else None)
    result = agent.run(args.question)
    print(result.answer)
    return 0 if result.finished else 1


def cmd_eval(args) -> int:
    db = ReadOnlyDB(_db_path(args))
    agent = Agent(get_llm(args.llm, args.model), db, args.max_steps)
    print(format_report(evaluate(agent, db, load_cases(args.cases))))
    return 0


def cmd_tools(args) -> int:
    agent = Agent(llm=None, db=ReadOnlyDB(_db_path(args)))  # type: ignore[arg-type]
    print(agent._describe_tools())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sqlagent")
    parser.add_argument("--db", help=f"database file (default {DEFAULT_DB})")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="create the sample shop database").set_defaults(func=cmd_init_db)
    sub.add_parser("tools", help="list the tools the agent can use").set_defaults(func=cmd_tools)

    for name, func in (("ask", cmd_ask), ("eval", cmd_eval)):
        p = sub.add_parser(name, help="ask a question" if name == "ask" else "run the evaluation")
        p.add_argument("--llm", choices=["ollama", "gemini"])
        p.add_argument("--model")
        p.add_argument("--max-steps", type=int, default=8)
        if name == "ask":
            p.add_argument("question")
            p.add_argument("-v", "--verbose", action="store_true", help="show each tool call")
        else:
            p.add_argument("cases", nargs="?", default=DEFAULT_CASES)
        p.set_defaults(func=func)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (LLMError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
