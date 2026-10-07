"""The mock llama-server is held to what the harness checks of a real one: GGUF metadata it can read back,
template parity, prompt-cache accounting, schema-shaped survey and judge replies."""
from __future__ import annotations
import json
import random
import socket
import struct
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
import pytest
from harness import log, run as R
from harness.client import LlamaClient, ServerError
from harness.dialogue import AgentHandle, DialogueRunner, DyadSpec, GenSettings
from harness.scorer import parse_score, score_schema
from harness.survey import answer_schema, parse_answer
from harness.templates import ChatTemplate, render
from harness.transcript import MENTOR, SEEKER
from sandbox.mock_server import (CHATML, ROLES, MockBackend, MockLlamaServer, config_snippet, gguf_bytes,
                                 write_gguf, write_mock_gguf)

REPO = Path(__file__).resolve().parents[2]
TPL = ChatTemplate.from_source(CHATML)


@pytest.fixture
def server(tmp_path):
    s = MockLlamaServer("seeker", write_mock_gguf(tmp_path / "seeker.gguf", "seeker"), slots=4)
    s.start()
    yield s
    s.stop()


def _post(url: str, body: bytes, path: str):
    req = urllib.request.Request(url + path, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


# ------------------------------------------------------------------------------------------------ GGUF

def test_gguf_header_is_v3_with_zero_tensors_and_aligned():
    data = gguf_bytes({"a": "x", "b": 7, "c": ["p", "q"]})
    assert data[:4] == b"GGUF"
    assert struct.unpack_from("<IQQ", data, 4) == (3, 0, 3)
    assert len(data) % 32 == 0


def test_write_gguf_refuses_unsupported_values(tmp_path):
    for bad in (1.5, True, -1, 2 ** 32, [1, 2], {"k": "v"}, None):
        with pytest.raises(ValueError):
            write_gguf(tmp_path / "x.gguf", {"key": bad})
    with pytest.raises(ValueError):
        write_gguf(tmp_path / "x.gguf", {"": "empty key"})


def test_write_gguf_leaves_an_identical_file_untouched(tmp_path):
    p = write_mock_gguf(tmp_path / "m.gguf", "mentor")
    before = p.stat().st_mtime_ns
    time.sleep(0.01)
    assert write_mock_gguf(tmp_path / "m.gguf", "mentor") == p and p.stat().st_mtime_ns == before
    write_mock_gguf(tmp_path / "m.gguf", "judge")
    assert b"sandbox-mock-judge" in p.read_bytes()


def test_gguf_round_trips_through_the_harness_reader(tmp_path):
    pytest.importorskip("gguf")
    from harness.templates import read_template_from_gguf
    import gguf
    p = write_mock_gguf(tmp_path / "s.gguf", "seeker")
    tpl = read_template_from_gguf(p)
    assert tpl.source == CHATML and tpl.bos == "<s>" and tpl.eos == "</s>"
    reader = gguf.GGUFReader(str(p))
    name = reader.fields["general.name"]
    assert bytes(name.parts[name.data[0]]).decode() == "sandbox-mock-seeker"
    assert len(reader.tensors) == 0
    custom = "{% for m in messages %}[{{ m['role'] }}] {{ m['content'] }}\n{% endfor %}"
    assert read_template_from_gguf(write_mock_gguf(tmp_path / "c.gguf", "judge", custom)).source == custom


# ------------------------------------------------------------------------------------------------ endpoints

def test_health_props_apply_template_and_tokenize(server, tmp_path):
    c = LlamaClient(server.url)
    assert c.health()
    props = c.props()
    assert props["model_path"] == str((tmp_path / "seeker.gguf").resolve())
    assert props["model_alias"] == "sandbox-mock-seeker" and props["total_slots"] == 4
    assert props["chat_template"] == CHATML and props["build_info"] == "sandbox-mock"
    assert props["default_generation_settings"]["n_ctx"] == 131072
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hello there."}]
    assert c.apply_template(msgs) == render(TPL, msgs)
    assert c.tokenize("one two  three\nfour") == 4


def test_unknown_path_is_404_and_a_bad_body_is_400(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(server.url + "/nope", timeout=5)
    assert e.value.code == 404
    assert _post(server.url, b"not json", "/completion")[0] == 400
    assert _post(server.url, b"[]", "/tokenize")[0] == 400
    status, body = _post(server.url, json.dumps({"prompt": "x", "id_slot": 9}).encode(), "/completion")
    assert status == 400 and "id_slot" in body["error"]["message"]
    assert _post(server.url, json.dumps({"messages": "x"}).encode(), "/apply-template")[0] == 400
    assert _post(server.url, b"{}", "/nope")[0] == 404


@pytest.mark.parametrize("role", [SEEKER, MENTOR])
def test_check_agent_passes_against_the_mock(tmp_path, role):
    s = MockLlamaServer(role, write_mock_gguf(tmp_path / f"{role}.gguf", role))
    s.start()
    try:
        rows = R.check_agent(AgentHandle(role, LlamaClient(s.url), TPL, "sha", 0), {"now": "2026-09-08"})
    finally:
        s.stop()
    by_name = {name: ok for name, ok, _ in rows}
    assert False not in by_name.values(), rows
    for name in ("health", "template_parity", "template_parity_user_first", "cache_reuse", "server_chat_template"):
        assert by_name[name] is True, rows
    if role == SEEKER:
        assert by_name["trailing_system"] is True


def test_prompt_cache_is_per_slot(server):
    c = LlamaClient(server.url)
    p1 = "a b c d e f g h"
    assert c.complete(p1, id_slot=0, seed=1, n_predict=5, temperature=0).prompt_n == 8
    assert c.complete(p1 + " i j", id_slot=0, seed=1, n_predict=5, temperature=0).prompt_n == 2
    assert c.complete(p1 + " i j", id_slot=1, seed=1, n_predict=5, temperature=0).prompt_n == 10
    again = c.complete(p1 + " i j", id_slot=0, seed=1, n_predict=5, temperature=0)
    assert again.prompt_n == 1 and again.tokens_evaluated == 10      # identical prompt: the last token only
    assert c.complete(p1 + " i j", id_slot=0, seed=1, n_predict=5, temperature=0, cache_prompt=False).prompt_n == 10
    assert c.complete("z " + p1, id_slot=0, seed=1, n_predict=5, temperature=0).prompt_n == 9


def test_n_predict_caps_the_reply_in_words(server):
    c = LlamaClient(server.url)
    full = c.complete("hello", id_slot=0, seed=1, n_predict=300, temperature=0.7)
    assert full.finish_reason == "stop" and full.raw["stop_type"] == "eos" and full.truncated is False
    assert full.predicted_n == len(full.text.split()) >= 20
    capped = c.complete("hello", id_slot=0, seed=1, n_predict=3, temperature=0.7)
    assert capped.text == " ".join(full.text.split()[:3]) and capped.finish_reason == "length"
    assert capped.predicted_n == 3 and capped.truncated is False


def test_context_limit_truncates_and_an_oversized_prompt_is_refused(tmp_path):
    s = MockLlamaServer("mentor", write_mock_gguf(tmp_path / "m.gguf", "mentor"), n_ctx=10)
    s.start()
    try:
        c = LlamaClient(s.url)
        comp = c.complete("one two three four five six seven", id_slot=0, seed=1, n_predict=300, temperature=0.7)
        assert comp.predicted_n == 3 and comp.truncated is True and comp.finish_reason == "length"
        with pytest.raises(ServerError, match="context"):
            c.complete(" ".join(["w"] * 10), id_slot=0, seed=1, n_predict=5, temperature=0.7)
    finally:
        s.stop()


def test_free_text_is_deterministic_in_seed_and_prompt_and_flavoured_by_role(server, tmp_path):
    c = LlamaClient(server.url)
    a = c.complete("same prompt", id_slot=0, seed=7, n_predict=300, temperature=0.7).text
    assert a == c.complete("same prompt", id_slot=1, seed=7, n_predict=300, temperature=0.7).text
    others = {c.complete("same prompt", id_slot=0, seed=s, n_predict=300, temperature=0.7).text for s in range(8, 14)}
    assert len(others - {a}) >= 3
    assert a.startswith("[mock-seeker] ") and a.rstrip().endswith("?")
    m = MockLlamaServer("mentor", write_mock_gguf(tmp_path / "m.gguf", "mentor"))
    m.start()
    try:
        assert LlamaClient(m.url).complete("same prompt", id_slot=0, seed=7, n_predict=300,
                                           temperature=0.7).text.startswith("[mock-mentor] ")
    finally:
        m.stop()


def test_stop_strings_end_the_reply(server):
    c = LlamaClient(server.url)
    full = c.complete("hello", id_slot=0, seed=1, n_predict=300, temperature=0.7).text
    word = full.split()[3]
    cut = c.complete("hello", id_slot=0, seed=1, n_predict=300, temperature=0.7, stop=[word])
    assert cut.text == full[:full.find(word)] and cut.raw["stop_type"] == "word"


def test_survey_and_judge_schema_replies_parse(server):
    c = LlamaClient(server.url)
    for lo, hi in ((1, 5), (0, 10), (1, 7), (3, 3)):
        item = {"id": "x", "battery": "b", "text": "?", "scale": {"min": lo, "max": hi}}
        seen = set()
        for seed in range(25):
            comp = c.complete(f"item {lo}-{hi}", id_slot=0, seed=seed, n_predict=32, temperature=0.0,
                              json_schema=answer_schema(item))
            parsed = parse_answer(comp.text, item)
            assert parsed.method == "json" and lo <= parsed.value <= hi
            seen.add(parsed.value)
        assert seen == {lo} if hi == lo else len(seen) > 1
    scores = set()
    for seed in range(10):
        comp = c.complete("judge this", id_slot=2, seed=seed, n_predict=160, temperature=0.0,
                          json_schema=score_schema())
        score, rationale = parse_score(comp.text)
        assert score is not None and 0 <= score <= 1 and round(score, 2) == score
        assert rationale.startswith("mock rationale")
        scores.add(score)
    assert len(scores) > 1
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"},
                                               "n": {"type": "integer", "minimum": 2, "maximum": 4}}}
    other = c.complete("x", id_slot=0, seed=1, n_predict=50, temperature=0, json_schema=schema)
    d = json.loads(other.text)
    assert d["ok"] is False and 2 <= d["n"] <= 4


def test_a_dialogue_against_two_mocks_logs_no_cache_warning(tmp_path):
    servers = [MockLlamaServer(r, write_mock_gguf(tmp_path / f"{r}.gguf", r)) for r in (SEEKER, MENTOR)]
    for s in servers:
        s.start()
    try:
        seeker = AgentHandle(SEEKER, LlamaClient(servers[0].url), TPL, "s" * 64, slot=3)
        mentor = AgentHandle(MENTOR, LlamaClient(servers[1].url), TPL, "m" * 64, slot=3)
        w = log.JsonlWriter(tmp_path / "turns.jsonl")
        runner = DialogueRunner("r1", 5, seeker, mentor, GenSettings(), w)
        spec = DyadSpec("d1", {"topic": "t"}, "You are Dana, a nurse.", "Note to self: I am Dana.", "reinforced", 11, 3)
        transcript = runner.run(spec, attempt=1)
    finally:
        for s in servers:
            s.stop()
    rows = log.read_jsonl(tmp_path / "turns.jsonl")
    assert len(rows) == 6 and transcript.n_messages == 6
    assert all(r["cache_warning"] is False and r["finish_reason"] == "stop" and r["prompt_n"] > 0 for r in rows), rows
    later = [r for r in rows if r["turn"] > 1]
    assert all(r["prompt_n"] <= r["expected_new"] + 1 for r in later)
    assert [r["text"].split()[0] for r in rows] == ["[mock-seeker]", "[mock-mentor]"] * 3


def test_concurrent_completions_on_different_slots(tmp_path):
    s = MockLlamaServer("mentor", write_mock_gguf(tmp_path / "m.gguf", "mentor"), slots=4, delay=0.01)
    s.start()
    results: dict[int, list] = {}
    try:
        c = LlamaClient(s.url)

        def work(slot):
            base = f"slot {slot} " + " ".join(f"w{slot}x{i}" for i in range(50))
            first = c.complete(base, id_slot=slot, seed=slot, n_predict=30, temperature=0.7)
            second = c.complete(base + " more words", id_slot=slot, seed=slot, n_predict=30, temperature=0.7)
            results[slot] = [first, second]

        t0 = time.monotonic()
        threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        elapsed = time.monotonic() - t0
    finally:
        s.stop()
    assert sorted(results) == [0, 1, 2, 3]
    for slot, (first, second) in results.items():
        assert first.prompt_n == 52 and second.prompt_n == 2, slot       # no slot saw another slot's prompt
    serial = sum(c.predicted_n for pair in results.values() for c in pair) * 0.01
    assert elapsed < serial * 0.6, (elapsed, serial)


# ------------------------------------------------------------------------------------------------ backend

def _port_free(port: int) -> bool:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def test_backend_three_roles_three_urls_distinct_models_and_stop_frees_ports(tmp_path):
    b = MockBackend(tmp_path / "mock", slots=2)
    assert b.info() == {"running": False, "slots": 2, "delay": 0.0, "seeker": None, "mentor": None, "judge": None}
    info = b.start()
    try:
        assert info["running"] and b.running and b.start() == info
        urls = [info[r]["url"] for r in ROLES]
        assert len(set(urls)) == 3
        shas = {log.sha256_file(Path(info[r]["gguf_path"])) for r in ROLES}
        assert len(shas) == 3
        for r in ROLES:
            props = LlamaClient(info[r]["url"]).props()
            assert props["model_alias"] == f"sandbox-mock-{r}" and props["model_path"] == info[r]["gguf_path"]
            assert props["total_slots"] == 2
        assert config_snippet(info)["judge"] == info["judge"]
        ports = [int(u.rsplit(":", 1)[1]) for u in urls]
    finally:
        b.stop()
    assert not b.running and b.info()["seeker"] is None
    assert all(_port_free(p) for p in ports)
    assert not LlamaClient(urls[0]).health()


def test_backend_base_port_gives_consecutive_ports(tmp_path):
    rng = random.Random()
    for _ in range(20):
        base = rng.randrange(20000, 60000)
        if all(_port_free(base + i) for i in range(3)):
            break
    else:
        pytest.skip("no three consecutive free ports found")
    b = MockBackend(tmp_path, base_port=base)
    info = b.start()
    try:
        assert [info[r]["url"] for r in ROLES] == [f"http://127.0.0.1:{base + i}" for i in range(3)]
        with pytest.raises(OSError):
            MockBackend(tmp_path / "other", base_port=base).start()
        assert b.running
    finally:
        b.stop()


def test_harness_check_run_and_score_against_the_mock_backend(tmp_path, monkeypatch, capsys):
    """The real CLI code paths, in process: check passes, a two-dyad run completes with no cache warning,
    every survey answer parses on the schema path, and the judge's scores parse."""
    pytest.importorskip("gguf")
    monkeypatch.setattr(R, "HASH_CACHE", tmp_path / "hashes.json")
    b = MockBackend(tmp_path / "mock", slots=2)
    info = b.start()
    try:
        cfg = {"data_dir": str(tmp_path / "data"), "batteries": str(REPO / "instruments" / "batteries.json"),
               "grid": None, "run_seed": 3, "now": "2026-09-08", **config_snippet(info)}
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(cfg))
        manifest = tmp_path / "dyads.jsonl"
        manifest.write_text("".join(json.dumps(
            {"dyad_id": f"d{i}", "condition": {"topic": "immigration_enforcement"}, "persona_text": "You are Dana.",
             "persona_reminder": "Note to self: Dana.", "persona_mode": mode, "seed": 100 + i, "n_turns": 2}) + "\n"
            for i, mode in enumerate(("reinforced", "once"))))
        assert R.main(["check", "--config", str(cfg_path), "--manifest", str(manifest)]) == 0
        assert R.main(["run", "--config", str(cfg_path), "--manifest", str(manifest), "--run-id", "mock1"]) == 0
        assert R.main(["score", "--config", str(cfg_path), "--run-id", "mock1", "--scope", "pilot"]) == 0
    finally:
        b.stop()
    out = capsys.readouterr().out
    assert "FAIL" not in out
    run = tmp_path / "data" / "mock1"
    status = log.read_jsonl(run / "status.jsonl")
    assert sorted(r["dyad_id"] for r in status if r["status"] == "complete") == ["d0", "d1"]
    turns = log.read_jsonl(run / "turns.jsonl")
    assert len(turns) == 8 and not any(r["cache_warning"] for r in turns)
    surveys = log.read_jsonl(run / "surveys.jsonl")
    assert surveys and all(r["answer_method"] == "json" and r["answer"] is not None for r in surveys)
    scores = log.read_jsonl(run / "scores.jsonl")
    assert scores and all(r["score"] is not None and not r.get("error") for r in scores)
