"""The turn loop: order, log rows, seeds, slot pinning, persona modes and the KV-cache reuse audit."""
import pytest
from harness import log
from harness.dialogue import AgentHandle, GenSettings, DyadSpec, DialogueRunner, DialogueError, CacheReuseLost, expected_new_tokens
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
    assert rows[0]["seed"] == log.derive_seed(99, 5, "d1", 1, 1, SEEKER)   # 5 = the spec's per-dyad seed
    assert [c["seed"] for c in sc.calls] == [rows[0]["seed"], rows[2]["seed"]]


def test_provenance_fields(tmp_path):
    runner, sc, _, _ = make_runner(tmp_path)
    runner.run(spec(n_turns=1), attempt=2)
    r = log.read_jsonl(tmp_path / "turns.jsonl")[0]
    assert r["model_sha256"] == "seekerhash" and r["persona_mode"] == "reinforced"
    assert r["prompt_sha256"] == log.sha256_text(sc.calls[0]["prompt"]) and r["prompt_chars"] == len(sc.calls[0]["prompt"])
    assert r["finish_reason"] == "stop" and r["attempt"] == 2 and r["predicted_n"] > 0
    assert r["id_slot"] == 2                                    # the slot whose KV cache this turn reused
    assert (r["temperature"], r["top_p"], r["n_predict"]) == (0.7, 0.95, 300)


def test_turn_row_carries_context_accounting_fields(tmp_path):
    runner, sc, mc, _ = make_runner(tmp_path)
    runner.run(spec(n_turns=2), attempt=1)
    rows = log.read_jsonl(tmp_path / "turns.jsonl")
    assert all(r["truncated"] is False for r in rows)
    assert all(r["tokens_evaluated"] == r["prompt_n"] for r in rows)
    seeker_rows = [r for r in rows if r["agent"] == SEEKER]
    assert seeker_rows[0]["tokens_cached"] == 0                 # no previous prompt on the slot yet
    assert seeker_rows[1]["tokens_cached"] > 0                  # the second turn reuses the cached prefix


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


def test_lost_cache_on_a_mid_dialogue_turn_fails_the_dyad_after_logging_the_row(tmp_path):
    # The single largest lever in the pipeline fails silently: a turn that re-prefills the whole transcript
    # costs ~4,000x the budgeted prefill and nothing errors. So a mid-dialogue turn whose prefill is both
    # unexpected (cache_warning) and large (over cache_reuse_limit) is a hard failure of the dyad, not a
    # flag in a column nobody reads until the wave is a week late.
    runner, sc, mc, _ = make_runner(tmp_path)
    runner.settings.cache_reuse_limit = 1000
    def cold_complete(prompt, **kw):
        c = FakeClient.complete(sc, prompt, **kw)
        c.prompt_n = 5000                        # the server prefilled the whole transcript again
        return c
    sc.complete = cold_complete
    with pytest.raises(DialogueError) as ei:
        runner.run(spec(n_turns=3), attempt=1)
    assert ei.value.turn == 2 and ei.value.agent == SEEKER
    assert isinstance(ei.value.cause, CacheReuseLost) and "5000" in str(ei.value.cause)
    rows = log.read_jsonl(tmp_path / "turns.jsonl")
    assert rows[-1]["agent"] == SEEKER and rows[-1]["turn"] == 2 and rows[-1]["cache_warning"] is True
    assert rows[-1]["prompt_n"] == 5000 and rows[-1]["finish_reason"] == "stop"   # the row is kept as evidence


def test_first_turn_may_prefill_the_whole_persona_without_failing(tmp_path):
    # Turn 1 has nothing cached: a 5,000-token persona prefill there is normal, not a lost cache.
    runner, sc, mc, _ = make_runner(tmp_path)
    runner.settings.cache_reuse_limit = 1000
    calls = {"n": 0}
    def first_cold(prompt, **kw):
        c = FakeClient.complete(sc, prompt, **kw)
        calls["n"] += 1
        if calls["n"] == 1:
            c.prompt_n = 5000
        return c
    sc.complete = first_cold
    runner.run(spec(n_turns=2), attempt=1)      # does not raise


def test_a_large_but_expected_prefill_is_not_a_lost_cache(tmp_path):
    # A long partner line makes expected_new large too; that is not a cache loss and must not fail the dyad.
    runner, sc, mc, _ = make_runner(tmp_path, mentor_replies=[" ".join(["word"] * 1500)])
    runner.settings.cache_reuse_limit = 1000
    runner.run(spec(n_turns=2), attempt=1)      # seeker turn 2 prefills ~1,500 new tokens, all of them expected


def test_cache_reuse_limit_none_disables_the_hard_failure(tmp_path):
    runner, sc, mc, _ = make_runner(tmp_path)
    runner.settings.cache_reuse_limit = None
    def cold_complete(prompt, **kw):
        c = FakeClient.complete(sc, prompt, **kw)
        c.prompt_n = 5000
        return c
    sc.complete = cold_complete
    runner.run(spec(n_turns=2), attempt=1)
    assert all(r["cache_warning"] for r in log.read_jsonl(tmp_path / "turns.jsonl") if r["agent"] == SEEKER)


# --- red-team H1: reasoning output is kept out of the partner's view --------------------------------------

class ReasoningClient(FakeClient):
    """Returns `reasoning_content` beside the reply, as a server that parses reasoning does."""
    def complete(self, prompt, **kw):
        c = FakeClient.complete(self, prompt, **kw)
        c.raw = {"content": c.text, "reasoning_content": "private plan"}
        return c


def test_reasoning_returned_separately_is_logged_and_never_passed_on(tmp_path):
    runner, sc, mc, _ = make_runner(tmp_path, mentor_replies=["Here is my advice."])
    runner.agents[MENTOR].client = ReasoningClient(["Here is my advice."])
    runner.run(spec(n_turns=2), attempt=1)
    rows = log.read_jsonl(tmp_path / "turns.jsonl")
    mentor = [r for r in rows if r["agent"] == MENTOR]
    assert all(r["reasoning"] == "private plan" and r["text"] == "Here is my advice." for r in mentor)
    assert all(r["reasoning"] is None for r in rows if r["agent"] == SEEKER)
    assert "private plan" not in sc.calls[1]["prompt"] and "Here is my advice." in sc.calls[1]["prompt"]


def test_a_think_block_is_stripped_into_the_reasoning_field(tmp_path):
    reply = "<think>I am playing Dana, stay in role</think>\n\nQuestion?"
    runner, sc, mc, _ = make_runner(tmp_path, seeker_replies=[reply])
    runner.run(spec(n_turns=2), attempt=1)
    seeker = [r for r in log.read_jsonl(tmp_path / "turns.jsonl") if r["agent"] == SEEKER]
    assert seeker[0]["text"] == "Question?" and seeker[0]["reasoning"] == "I am playing Dana, stay in role"
    assert all("<think>" not in c["prompt"] and "stay in role" not in c["prompt"] for c in mc.calls)
    assert "Question?" in mc.calls[0]["prompt"]
    from harness.dialogue import split_reasoning, UnterminatedThink
    assert split_reasoning("plan</think>Answer") == ("Answer", "plan")          # the prompt opened <think>
    assert split_reasoning("Plain reply.\n") == ("Plain reply.\n", None)         # untouched
    with pytest.raises(UnterminatedThink):
        split_reasoning("<think>still thinking when n_predict ran out")


def test_harmony_channel_markup_fails_the_dyad_instead_of_reaching_the_partner(tmp_path):
    harmony = ("<|channel|>analysis<|message|>The user leans right; I should push back on them.<|end|>"
               "<|start|>assistant<|channel|>final<|message|>Here is my advice.")
    runner, sc, mc, _ = make_runner(tmp_path, mentor_replies=[harmony])
    with pytest.raises(DialogueError) as e:
        runner.run(spec(n_turns=2), attempt=1)
    from harness.dialogue import HarmonyMarkup
    assert isinstance(e.value.cause, HarmonyMarkup) and "harmony channel markup" in str(e.value)
    rows = log.read_jsonl(tmp_path / "turns.jsonl")
    assert rows[-1]["agent"] == MENTOR and rows[-1]["error"].startswith("HarmonyMarkup")   # logged first
    assert rows[-1]["text"] == harmony
    assert len(sc.calls) == 1                                       # the seeker never saw it


def test_missing_timings_are_marked_not_read_as_zero(tmp_path, capsys):
    # Red-team L8: prompt_n defaulted to 0, which passed every cache check in silence.
    class NoTimings(FakeClient):
        def complete(self, prompt, **kw):
            c = FakeClient.complete(self, prompt, **kw)
            c.prompt_n = c.predicted_n = None
            c.timings = {}
            return c
    runner, sc, mc, _ = make_runner(tmp_path)
    runner.agents[MENTOR].client = NoTimings(["Tell me more."])
    runner.run(spec(n_turns=3), attempt=1)
    rows = [r for r in log.read_jsonl(tmp_path / "turns.jsonl") if r["agent"] == MENTOR]
    assert all(r["timings_missing"] is True and r["prompt_n"] is None and r["cache_warning"] is None
               for r in rows)
    seeker = [r for r in log.read_jsonl(tmp_path / "turns.jsonl") if r["agent"] == SEEKER]
    assert all("timings_missing" not in r for r in seeker)
    assert capsys.readouterr().err.count("reported no timings") == 1          # warned once, not every turn


def test_an_empty_reply_is_flagged_and_passed_on_not_failed(tmp_path):
    # Red-team L8: an empty or whitespace-only reply (or one that was only reasoning) is logged with
    # empty_reply and the dialogue goes on; refusal-like silence is a finding, not a failure.
    runner, sc, mc, _ = make_runner(tmp_path, mentor_replies=["", "  \n", "<think>hm</think>", "Fine."])
    t = runner.run(spec(n_turns=4), attempt=1)
    assert t.n_messages == 8
    rows = [r for r in log.read_jsonl(tmp_path / "turns.jsonl") if r["agent"] == MENTOR]
    assert [r.get("empty_reply", False) for r in rows] == [True, True, True, False]
    assert all("error" not in r for r in rows)
    seeker = [r for r in log.read_jsonl(tmp_path / "turns.jsonl") if r["agent"] == SEEKER]
    assert all("empty_reply" not in r for r in seeker)
