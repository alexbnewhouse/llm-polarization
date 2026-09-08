import json
import pytest
from harness import client as C


def test_parse_completion_stop_and_length():
    raw = {"content": "hi", "stop_type": "eos", "timings": {"prompt_n": 5, "predicted_n": 2, "predicted_per_second": 9.5}}
    c = C.parse_completion(raw)
    assert (c.text, c.finish_reason, c.prompt_n, c.predicted_n) == ("hi", "stop", 5, 2)
    assert c.timings["predicted_per_second"] == 9.5 and c.raw is raw
    assert C.parse_completion({"content": "x", "stop_type": "limit", "timings": {}}).finish_reason == "length"
    assert C.parse_completion({"content": "x", "stopped_limit": True}).finish_reason == "length"
    assert C.parse_completion({"content": "x"}).prompt_n == 0


def test_complete_builds_request_body(monkeypatch):
    cl = C.LlamaClient("http://127.0.0.1:1")
    seen = {}
    def fake_post(path, body):
        seen["path"], seen["body"] = path, body
        return {"content": "ok", "stop_type": "eos", "timings": {"prompt_n": 1, "predicted_n": 1}}
    monkeypatch.setattr(cl, "_post", fake_post)
    c = cl.complete("PROMPT", id_slot=3, seed=42, n_predict=300, temperature=0.7,
                    json_schema={"type": "object"}, stop=["<|im_end|>"])
    assert seen["path"] == "/completion" and c.text == "ok"
    b = seen["body"]
    assert b["prompt"] == "PROMPT" and b["id_slot"] == 3 and b["seed"] == 42
    assert b["n_predict"] == 300 and b["temperature"] == 0.7 and b["top_p"] == 0.95
    assert b["cache_prompt"] is True and b["json_schema"] == {"type": "object"} and b["stop"] == ["<|im_end|>"]


def test_complete_omits_json_schema_and_stop_when_absent(monkeypatch):
    cl = C.LlamaClient("http://127.0.0.1:1")
    seen = {}
    monkeypatch.setattr(cl, "_post", lambda p, b: seen.setdefault("body", b) or {"content": ""})
    cl.complete("P", id_slot=0, seed=1, n_predict=5, temperature=0.0)
    assert "json_schema" not in seen["body"] and "stop" not in seen["body"]


def test_tokenize_and_apply_template_and_health(monkeypatch):
    cl = C.LlamaClient("http://127.0.0.1:1")
    monkeypatch.setattr(cl, "_post", lambda p, b: {"tokens": [1, 2, 3]} if p == "/tokenize" else {"prompt": "RENDERED"})
    assert cl.tokenize("a b c") == 3
    assert cl.apply_template([{"role": "user", "content": "x"}]) == "RENDERED"
    monkeypatch.setattr(cl, "_get", lambda p: {"status": "ok"})
    assert cl.health() is True
    def boom(p):
        raise C.ServerError("down")
    monkeypatch.setattr(cl, "_get", boom)
    assert cl.health() is False


def test_post_wraps_network_errors_as_server_error():
    cl = C.LlamaClient("http://127.0.0.1:9", timeout=1)   # nothing listens on port 9
    with pytest.raises(C.ServerError):
        cl._post("/completion", {"prompt": "x"})
    with pytest.raises(C.ServerError):
        cl._get("/health")


def test_parse_completion_keeps_the_servers_context_accounting():
    raw = {"content": "hi", "stop_type": "limit", "truncated": True, "tokens_evaluated": 900,
           "tokens_cached": 850, "timings": {"prompt_n": 50, "predicted_n": 300}}
    c = C.parse_completion(raw)
    # finish_reason "length" alone cannot tell an n_predict cap from a slot that ran out of context.
    assert c.finish_reason == "length" and c.truncated is True
    assert c.tokens_evaluated == 900 and c.tokens_cached == 850
    old = C.parse_completion({"content": "x"})          # a build that reports none of the three
    assert (old.truncated, old.tokens_evaluated, old.tokens_cached) == (None, None, None)
