import pytest
from harness import log
from harness.dialogue import AgentHandle, GenSettings, DyadSpec, DialogueRunner, DialogueError, expected_new_tokens
from harness.templates import ChatTemplate
from harness.transcript import SEEKER, MENTOR
from harness.tests.fakes import FakeClient
from harness.tests.conftest import CHATML


def make_runner(tmp_path, seeker_replies=None, mentor_replies=None, fail_on=None):
    tpl = ChatTemplate.from_source(CHATML)
    sc = FakeClient(seeker_replies or ["I need advice about the border."], fail_on=fail_on)
    mc = FakeClient(mentor_replies or ["Tell me more."])
    seeker = AgentHandle(SEEKER, sc, tpl, "seekerhash", slot=2, alias="s")
    mentor = AgentHandle(MENTOR, mc, tpl, "mentorhash", slot=2, alias="m")
    w = log.JsonlWriter(tmp_path / "turns.jsonl")
    return DialogueRunner("run1", 99, seeker, mentor, GenSettings(), w, clock=lambda: "T"), sc, mc, w


def spec(mode="reinforced", n_turns=3):
    return DyadSpec("d1", {"topic": "immigration_enforcement"}, "You are Dana.", "Note to self: Dana.", mode, 5, n_turns)


def test_dyad_spec_from_row():
    s = DyadSpec.from_row({"dyad_id": "x", "condition": {"topic": "t"}, "persona_text": "p",
                           "persona_reminder": "r", "persona_mode": "once", "seed": 1, "n_turns": 2})
    assert s.dyad_id == "x" and s.n_turns == 2 and s.persona_mode == "once"


def test_turn_order_and_row_count(tmp_path):
    runner, sc, mc, w = make_runner(tmp_path)
    t = runner.run(spec(n_turns=3), attempt=1)
    rows = log.read_jsonl(tmp_path / "turns.jsonl")
    assert [(r["turn"], r["agent"]) for r in rows] == [(1, SEEKER), (1, MENTOR), (2, SEEKER), (2, MENTOR), (3, SEEKER), (3, MENTOR)]
    assert t.n_messages == 6 and len(sc.calls) == 3 and len(mc.calls) == 3
    assert all(r["run_id"] == "run1" and r["attempt"] == 1 and r["adherence"] is None and r["ts"] == "T" for r in rows)


def test_every_request_is_pinned_and_cached(tmp_path):
    runner, sc, mc, _ = make_runner(tmp_path)
    runner.run(spec(), attempt=1)
    for c in sc.calls + mc.calls:
        assert c["id_slot"] == 2 and c["cache_prompt"] is True
        assert c["n_predict"] == 300 and c["temperature"] == 0.7 and c["top_p"] == 0.95


def test_seeker_opens_from_system_only_and_reminder_placement(tmp_path):
    runner, sc, mc, _ = make_runner(tmp_path)
    runner.run(spec("reinforced", n_turns=2), attempt=1)
    first = sc.calls[0]["prompt"]
    assert first.startswith("<|im_start|>system\nYou are Dana.<|im_end|>\n")
    assert "<|im_start|>user" not in first.split("<|im_start|>system\nNote to self")[0]
    assert first.endswith("<|im_start|>system\nNote to self: Dana.<|im_end|>\n<|im_start|>assistant\n")
    second = sc.calls[1]["prompt"]
    assert second.endswith("<|im_start|>system\nNote to self: Dana.<|im_end|>\n<|im_start|>assistant\n")
    assert second.count("Note to self") == 1
    for c in mc.calls:
        assert "<|im_start|>system" not in c["prompt"] and "Note to self" not in c["prompt"]


def test_once_mode_has_no_reminder(tmp_path):
    runner, sc, _, _ = make_runner(tmp_path)
    runner.run(spec("once", n_turns=2), attempt=1)
    assert all("Note to self" not in c["prompt"] for c in sc.calls)


def test_mentor_sees_seeker_lines_as_user(tmp_path):
    runner, sc, mc, _ = make_runner(tmp_path, seeker_replies=["SEEKER-LINE"], mentor_replies=["MENTOR-LINE"])
    runner.run(spec(n_turns=2), attempt=1)
    p = mc.calls[1]["prompt"]
    assert "<|im_start|>user\nSEEKER-LINE<|im_end|>" in p and "<|im_start|>assistant\nMENTOR-LINE<|im_end|>" in p


def test_seeds_distinct_and_logged(tmp_path):
    runner, sc, mc, _ = make_runner(tmp_path)
    runner.run(spec(n_turns=2), attempt=1)
    rows = log.read_jsonl(tmp_path / "turns.jsonl")
    seeds = [r["seed"] for r in rows]
    assert len(set(seeds)) == 4
    assert rows[0]["seed"] == log.derive_seed(99, "d1", 1, 1, SEEKER)
    assert [c["seed"] for c in sc.calls] == [rows[0]["seed"], rows[2]["seed"]]


def test_provenance_fields(tmp_path):
    runner, sc, _, _ = make_runner(tmp_path)
    runner.run(spec(n_turns=1), attempt=2)
    r = log.read_jsonl(tmp_path / "turns.jsonl")[0]
    assert r["model_sha256"] == "seekerhash" and r["persona_mode"] == "reinforced"
    assert r["prompt_sha256"] == log.sha256_text(sc.calls[0]["prompt"]) and r["prompt_chars"] == len(sc.calls[0]["prompt"])
    assert r["finish_reason"] == "stop" and r["attempt"] == 2 and r["predicted_n"] > 0


def test_cache_warning_is_false_when_cache_holds(tmp_path):
    runner, _, _, _ = make_runner(tmp_path)
    runner.run(spec("reinforced", n_turns=3), attempt=1)
    rows = log.read_jsonl(tmp_path / "turns.jsonl")
    assert all(r["cache_warning"] is False for r in rows)
    assert all(r["prompt_n"] <= r["expected_new"] + 64 for r in rows)


def test_cache_warning_true_when_server_reprefills_everything(tmp_path):
    runner, sc, mc, _ = make_runner(tmp_path)
    def cold_complete(prompt, **kw):
        c = FakeClient.complete(sc, prompt, **kw)
        c.prompt_n = len(prompt.split()) + 500   # pretend the cache was lost
        return c
    sc.complete = cold_complete
    runner.run(spec(n_turns=2), attempt=1)
    rows = [r for r in log.read_jsonl(tmp_path / "turns.jsonl") if r["agent"] == SEEKER]
    assert rows[1]["cache_warning"] is True


def test_expected_new_tokens():
    fc = FakeClient()
    assert expected_new_tokens(fc, "a b c d", None) == 4
    assert expected_new_tokens(fc, "a b c d e f", "a b c d") == 2
    assert expected_new_tokens(fc, "x y", "a b") == 2


def test_server_error_writes_error_row_and_raises(tmp_path):
    runner, sc, mc, _ = make_runner(tmp_path, fail_on=2)   # second seeker call fails
    with pytest.raises(DialogueError) as ei:
        runner.run(spec(n_turns=3), attempt=1)
    assert ei.value.turn == 2 and ei.value.agent == SEEKER and ei.value.dyad_id == "d1"
    rows = log.read_jsonl(tmp_path / "turns.jsonl")
    assert rows[-1]["finish_reason"] == "error" and rows[-1]["text"] == "" and "fake failure" in rows[-1]["error"]
    assert len(rows) == 3
