from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .agent import Agent
from .db import ReadOnlyDB
from .llm import LLMError


@dataclass
class Case:
    id: str
    question: str
    sql: str


@dataclass
class Outcome:
    case: Case
    correct: bool
    finished: bool
    steps: int
    answer: str


def load_cases(path: str | Path) -> list[Case]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Case(item["id"], item["question"], item["sql"]) for item in data]


def _normalize(text: str) -> str:
    # drop thousands separators so "1,234.50" matches 1234.50
    return re.sub(r"(?<=\d),(?=\d{3})", "", text).lower()


def _present(token: str, text: str) -> bool:
    return re.search(rf"(?<![\d.]){re.escape(token)}(?!\d|\.\d)", text) is not None


def value_matches(value, text: str) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        candidates = {f"{value:.2f}"}
        if round(value, 1) == round(value, 2):
            candidates.add(f"{value:.1f}")
        if abs(value - round(value)) < 0.005:
            candidates.add(str(round(value)))
        return any(_present(c, text) for c in candidates)
    if isinstance(value, int):
        return _present(str(value), text)
    return str(value).lower() in text


def answer_matches(answer: str, rows: list[tuple]) -> bool:
    text = _normalize(answer)
    return all(value_matches(v, text) for row in rows for v in row)


def evaluate(agent: Agent, db: ReadOnlyDB, cases: list[Case]) -> list[Outcome]:
    outcomes = []
    for case in cases:
        gold = db.query(case.sql, max_rows=1000).rows
        try:
            result = agent.run(case.question)
        except LLMError as error:
            # one failing model call should not throw away the other results
            outcomes.append(Outcome(case, False, False, 0, f"model error: {error}"))
            continue
        outcomes.append(
            Outcome(case, result.finished and answer_matches(result.answer, gold),
                    result.finished, len(result.steps), result.answer)
        )
    return outcomes


def format_report(outcomes: list[Outcome]) -> str:
    lines = []
    for o in outcomes:
        mark = "ok  " if o.correct else "FAIL"
        lines.append(f"{mark} {o.case.id:<4} steps={o.steps:<2} {o.case.question}")
        if not o.correct:
            lines.append(f"       answer: {' '.join(o.answer.split())[:160]}")
    total = len(outcomes)
    correct = sum(o.correct for o in outcomes)
    unfinished = sum(not o.finished for o in outcomes)
    mean_steps = sum(o.steps for o in outcomes) / total if total else 0.0
    lines.append("")
    lines.append(f"accuracy {correct}/{total} = {correct / total:.2f}" if total else "no cases")
    lines.append(f"unfinished {unfinished}, mean tool calls {mean_steps:.1f}")
    return "\n".join(lines)
