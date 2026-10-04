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


def test_ollama_error():
    llm = OllamaLLM(client=client(lambda r: httpx.Response(500)))
    with pytest.raises(LLMError):
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
