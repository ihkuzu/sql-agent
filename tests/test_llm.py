import json

import httpx
import pytest

from sqlagent.llm import GeminiLLM, LLMError, OllamaLLM, get_llm

MESSAGES = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "{}"}]


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_ollama_request_and_reply():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"message": {"content": ' {"final": "x"} '}})

    llm = OllamaLLM("m", host="0.0.0.0:11434", client=client(handler))
    assert llm.chat("sys", MESSAGES) == '{"final": "x"}'
    assert seen["url"] == "http://localhost:11434/api/chat"
    assert seen["body"]["format"] == "json"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert len(seen["body"]["messages"]) == 3


def test_ollama_error_includes_server_message():
    body = '{"error":"model runner has unexpectedly stopped"}'
    llm = OllamaLLM(client=client(lambda r: httpx.Response(500, text=body)))
    with pytest.raises(LLMError, match="500.*unexpectedly stopped"):
        llm.chat("s", MESSAGES)


def test_gemini_request_and_reply():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["key"] = request.headers["x-goog-api-key"]
        steps = [
            {"type": "thought", "content": [{"type": "text", "text": "hidden"}]},
            {"type": "model_output", "content": [{"type": "text", "text": '{"final": "y"}'}]},
        ]
        return httpx.Response(200, json={"steps": steps})

    llm = GeminiLLM(api_key="k", client=client(handler))
    assert llm.chat("sys", MESSAGES) == '{"final": "y"}'
    assert seen["key"] == "k"
    assert seen["body"]["system_instruction"] == "sys"
    assert seen["body"]["input"].startswith("User: hi") and seen["body"]["input"].endswith("Assistant:")


def test_gemini_errors(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMError):
        GeminiLLM()
    llm = GeminiLLM(api_key="k", client=client(lambda r: httpx.Response(403, text="denied")))
    with pytest.raises(LLMError, match="403"):
        llm.chat("s", MESSAGES)
    empty = GeminiLLM(api_key="k", client=client(lambda r: httpx.Response(200, json={"steps": []})))
    with pytest.raises(LLMError, match="no text"):
        empty.chat("s", MESSAGES)


def test_get_llm_env(monkeypatch):
    monkeypatch.setenv("SQLAGENT_LLM", "")
    monkeypatch.setenv("SQLAGENT_MODEL", "")
    assert isinstance(get_llm(), OllamaLLM)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert isinstance(get_llm("gemini"), GeminiLLM)
    with pytest.raises(ValueError):
        get_llm("nope")


def test_gemini_retries_invalid_json_error():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content)["input"])
        if len(bodies) < 3:
            return httpx.Response(400, text="Model generated invalid JSON syntax")
        step = {"type": "model_output", "content": [{"type": "text", "text": "ok"}]}
        return httpx.Response(200, json={"steps": [step]})

    llm = GeminiLLM(api_key="k", pause=0, client=client(handler))
    assert llm.chat("s", MESSAGES) == "ok"
    assert len(bodies) == 3 and "Previous attempt failed" in bodies[1]


def test_gemini_gives_up_after_retries():
    llm = GeminiLLM(api_key="k", retries=1, pause=0, client=client(lambda r: httpx.Response(400, text="invalid JSON")))
    with pytest.raises(LLMError, match="400"):
        llm.chat("s", MESSAGES)


def test_gemini_retries_when_overloaded():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503, text="high demand")
        step = {"type": "model_output", "content": [{"type": "text", "text": "fine"}]}
        return httpx.Response(200, json={"steps": [step]})

    llm = GeminiLLM(api_key="k", pause=0, client=client(handler))
    assert llm.chat("s", MESSAGES) == "fine" and len(calls) == 3


def test_gemini_does_not_retry_client_errors():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(403, text="denied")

    llm = GeminiLLM(api_key="k", pause=0, client=client(handler))
    with pytest.raises(LLMError, match="403"):
        llm.chat("s", MESSAGES)
    assert len(calls) == 1


def test_gemini_does_not_retry_when_daily_quota_is_spent():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, text="limit: 20 requests per day on Free Tier")

    llm = GeminiLLM(api_key="k", pause=0, client=client(handler))
    with pytest.raises(LLMError, match="429"):
        llm.chat("s", MESSAGES)
    assert len(calls) == 1
