from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from .db import ReadOnlyDB
from .llm import LLM
from .tools import Tool, ToolError, build_tools

SYSTEM_PROMPT = """You are a careful data analyst. You answer questions about a SQLite database by using tools.

Tools:
{tools}

Every reply must be exactly one JSON object and nothing else.
To use a tool: {{"tool": "<name>", "args": {{...}}}}
To give the final answer: {{"final": "<answer>"}}

Rules:
- Look at the tables and columns before you write SQL.
- Use only tables and columns you have seen.
- Base the answer only on query results. Never invent numbers.
- If an error comes back, fix the query and try again.
- If the data cannot answer the question, say so in the final answer.
- Keep the final answer short and include the exact values."""


class ActionError(ValueError):
    pass


@dataclass
class Step:
    tool: str
    args: dict
    observation: str
    error: bool = False


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)
    finished: bool = True


def parse_action(reply: str) -> dict:
    text = re.sub(r"```(?:json)?", "", reply).strip()
    start = text.find("{")
    if start < 0:
        raise ActionError("Reply with a single JSON object.")
    try:
        action, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as error:
        raise ActionError(f"Invalid JSON ({error.msg}). Reply with a single JSON object.") from error
    if not isinstance(action, dict):
        raise ActionError("Reply with a single JSON object.")
    if "final" in action:
        return {"final": str(action["final"])}
    if isinstance(action.get("tool"), str):
        args = action.get("args") or {}
        if not isinstance(args, dict):
            raise ActionError("'args' must be an object.")
        return {"tool": action["tool"], "args": args}
    raise ActionError('The object needs a "tool" or a "final" key.')


class Agent:
    def __init__(
        self,
        llm: LLM,
        db: ReadOnlyDB,
        max_steps: int = 8,
        on_step: Callable[[int, Step], None] | None = None,
    ):
        self.llm = llm
        self.max_steps = max_steps
        self.on_step = on_step
        self.tools: dict[str, Tool] = build_tools(db)
        self.system = SYSTEM_PROMPT.format(tools=self._describe_tools())

    def _describe_tools(self) -> str:
        lines = []
        for tool in self.tools.values():
            params = ", ".join(f"{k}: {v}" for k, v in tool.parameters.items()) or "no arguments"
            lines.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(lines)

    def _run_tool(self, name: str, args: dict) -> tuple[str, bool]:
        tool = self.tools.get(name)
        if tool is None:
            return f"unknown tool '{name}'. Available: {', '.join(self.tools)}", True
        try:
            return tool.run(**args), False
        except TypeError:
            expected = ", ".join(tool.parameters) or "no arguments"
            return f"wrong arguments for {name}. Expected: {expected}", True
        except ToolError as error:
            return f"error: {error}", True

    def run(self, question: str) -> AgentResult:
        messages = [{"role": "user", "content": question}]
        steps: list[Step] = []
        seen: set[str] = set()

        for number in range(1, self.max_steps + 1):
            reply = self.llm.chat(self.system, messages)
            messages.append({"role": "assistant", "content": reply})
            try:
                action = parse_action(reply)
            except ActionError as error:
                step = Step("(invalid reply)", {}, str(error), error=True)
            else:
                if "final" in action:
                    return AgentResult(action["final"], steps)
                key = json.dumps(action, sort_keys=True)
                if key in seen:
                    text = "You already ran this exact call. Try something different or give the final answer."
                    step = Step(action["tool"], action["args"], text, error=True)
                else:
                    seen.add(key)
                    text, failed = self._run_tool(action["tool"], action["args"])
                    step = Step(action["tool"], action["args"], text, failed)
            steps.append(step)
            if self.on_step:
                self.on_step(number, step)
            messages.append({"role": "user", "content": f"Observation: {step.observation}"})

        return AgentResult(
            f"I could not finish within {self.max_steps} steps.", steps, finished=False
        )
