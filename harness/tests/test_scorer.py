import json
import pytest
from harness import log, scorer
from harness.dialogue import AgentHandle, GenSettings
from harness.scorer import Scorer
from harness.templates import ChatTemplate
from harness.transcript import SEEKER, MENTOR
from harness.tests.fakes import FakeClient
from harness.tests.conftest import CHATML


def test_score_schema():
    s = scorer.score_schema()
    assert s["properties"]["score"] == {"type": "number", "minimum": 0, "maximum": 1}
    assert set(s["required"]) == {"score", "rationale"}


def test_build_judge_messages_contents():
    m = scorer.build_judge_messages("line_to_line", "PERSONA", "immigration", "LINE", ["P1", "P2"], "PARTNER")
    assert m[0]["role"] == "system" and m[1]["role"] == "user"
    u = m[1]["content"]
    assert "PERSONA" in u and "LINE" in u and "P1" in u and "P2" in u and "PARTNER" in u and "immigration" in u
    a = scorer.build_judge_messages("alignment", "PERSONA", "t", "L", [], None)[1]["content"]
    assert "agree" in a.lower()
    with pytest.raises(ValueError):
        scorer.build_judge_messages("vibes", "p", "t", "l", [], None)


def rows():
    out = []
    for turn in range(1, 6):
        out.append({"dyad_id": "d", "attempt": 1, "turn": turn, "agent": SEEKER, "text": f"s{turn}", "finish_reason": "stop"})
        out.append({"dyad_id": "d", "attempt": 1, "turn": turn, "agent": MENTOR, "text": f"m{turn}", "finish_reason": "stop"})
    out.append({"dyad_id": "d", "attempt": 1, "turn": 6, "agent": SEEKER, "text": "", "finish_reason": "error"})
    return out


def test_select_targets_pilot_and_main():
    pilot = scorer.select_targets(rows(), "pilot")
    assert len(pilot) == 5 * 2 + 5 * 1          # seeker: 2 metrics, mentor: 1
    assert all(r["finish_reason"] != "error" for r, _ in pilot)
    main = scorer.select_targets(rows(), "main")
    turns = sorted({r["turn"] for r, _ in main})
    assert turns == [4, 5] and all(r["agent"] == SEEKER for r, _ in main)
    assert {m for _, m in main} == {"prompt_to_line", "line_to_line"}
    with pytest.raises(ValueError):
        scorer.select_targets(rows(), "all")


def test_latest_complete_attempts():
    st = [{"dyad_id": "a", "attempt": 1, "status": "failed"}, {"dyad_id": "a", "attempt": 2, "status": "complete"},
          {"dyad_id": "b", "attempt": 1, "status": "started"}]
    assert scorer.latest_complete_attempts(st) == {"a": 2}


def test_parse_score():
    assert scorer.parse_score('{"score": 0.8, "rationale": "fits"}') == (0.8, "fits")
    assert scorer.parse_score('{"score": 2, "rationale": "x"}') == (None, "x")
    assert scorer.parse_score('garbage') == (None, "")


def make_run(tmp_path):
    p = log.run_paths(tmp_path, "r1")
    log.JsonlWriter(p.dyads).write({"dyad_id": "d", "attempt": 1, "condition": {"topic": "immigration"},
                                    "persona_text": "PERSONA", "persona_reminder": "", "persona_mode": "once",
                                    "seed": 1, "n_turns": 2})
    st = log.JsonlWriter(p.status)
    st.write({"dyad_id": "d", "attempt": 1, "status": "started"}); st.write({"dyad_id": "d", "attempt": 1, "status": "complete"})
    tw = log.JsonlWriter(p.turns)
    for turn in (1, 2):
        tw.write({"run_id": "r1", "dyad_id": "d", "attempt": 1, "turn": turn, "agent": SEEKER, "text": f"s{turn}", "finish_reason": "stop"})
        tw.write({"run_id": "r1", "dyad_id": "d", "attempt": 1, "turn": turn, "agent": MENTOR, "text": f"m{turn}", "finish_reason": "stop"})
    manifest = {"seeker": {"model_sha256": "S"}, "mentor": {"model_sha256": "M"}}
    return p, manifest


def make_scorer(tmp_path, judge_hash="J", replies=None):
    jc = FakeClient(replies or ['{"score": 0.75, "rationale": "ok"}'])
    judge = AgentHandle("judge", jc, ChatTemplate.from_source(CHATML), judge_hash, slot=0)
    p, manifest = make_run(tmp_path)
    sc = Scorer("r1", 99, judge, log.JsonlWriter(p.scores), GenSettings(), clock=lambda: "T")
    return sc, jc, p, manifest


def test_score_run_pilot_writes_rows_and_skips_done(tmp_path):
    sc, jc, p, manifest = make_scorer(tmp_path)
    n = sc.score_run(p, "pilot", manifest)
    assert n == 2 * 2 + 2 * 1 and len(jc.calls) == n
    rows_ = log.read_jsonl(p.scores)
    assert all(r["score"] == 0.75 and r["judge_sha256"] == "J" and r["ts"] == "T" for r in rows_)
    r0 = rows_[0]
    assert set(r0) >= {"run_id", "dyad_id", "attempt", "turn", "agent", "metric", "score", "rationale", "raw_text",
                       "judge_sha256", "judge_prompt_sha256", "seed", "ts"}
    assert r0["seed"] == log.derive_seed(99, "d", 1, r0["turn"], f"judge:{r0['agent']}:{r0['metric']}")
    assert sc.score_run(p, "pilot", manifest) == 0            # idempotent
    for c in jc.calls:
        assert c["json_schema"] == scorer.score_schema() and c["temperature"] == 0.0 and c["n_predict"] == 160
        assert "PERSONA" in c["prompt"] and c["id_slot"] == 0


def test_score_run_refuses_judge_equal_to_dialogue_model(tmp_path):
    sc, _, p, manifest = make_scorer(tmp_path, judge_hash="M")
    with pytest.raises(ValueError):
        sc.score_run(p, "pilot", manifest)


def test_score_run_logs_errors_and_continues(tmp_path):
    sc, jc, p, manifest = make_scorer(tmp_path)
    jc.fail_on = 2
    n = sc.score_run(p, "pilot", manifest)
    rows_ = log.read_jsonl(p.scores)
    assert n == 5 and len(rows_) == 6 and sum(1 for r in rows_ if r.get("error")) == 1


def test_score_run_duplicate_row_uses_position_not_value_equality(tmp_path):
    # Two seeker rows at turns 2 and 3 share identical text ("DUP"): any position lookup keyed by
    # a *value* derived from the row's contents (its (turn, agent) pair included, since the two
    # rows still collide once collapsed into one key when a duplicate turn/agent combination is
    # possible) risks feeding a row its own line back as an "earlier" one, or mixing up which
    # duplicate is "prior" to which. Resolving position by object identity (id(row)) has neither
    # failure mode: the FIRST DUP row must not see its own text as an earlier line, and the SECOND
    # must see the first DUP row's text exactly once.
    p = log.run_paths(tmp_path, "r1")
    log.JsonlWriter(p.dyads).write({"dyad_id": "d", "attempt": 1, "condition": {"topic": "immigration"},
                                    "persona_text": "PERSONA", "persona_reminder": "", "persona_mode": "once",
                                    "seed": 1, "n_turns": 3})
    st = log.JsonlWriter(p.status)
    st.write({"dyad_id": "d", "attempt": 1, "status": "started"}); st.write({"dyad_id": "d", "attempt": 1, "status": "complete"})
    tw = log.JsonlWriter(p.turns)
    tw.write({"run_id": "r1", "dyad_id": "d", "attempt": 1, "turn": 1, "agent": SEEKER, "text": "S1", "finish_reason": "stop"})
    tw.write({"run_id": "r1", "dyad_id": "d", "attempt": 1, "turn": 2, "agent": SEEKER, "text": "DUP", "finish_reason": "stop"})
    tw.write({"run_id": "r1", "dyad_id": "d", "attempt": 1, "turn": 3, "agent": SEEKER, "text": "DUP", "finish_reason": "stop"})
    manifest = {"seeker": {"model_sha256": "S"}, "mentor": {"model_sha256": "M"}}
    jc = FakeClient(['{"score": 0.5, "rationale": "x"}'])
    judge = AgentHandle("judge", jc, ChatTemplate.from_source(CHATML), "J", slot=0)
    sc = Scorer("r1", 99, judge, log.JsonlWriter(p.scores), GenSettings(), clock=lambda: "T")
    sc.score_run(p, "pilot", manifest)
    # pilot order (seeker rows only here): S1(prompt_to_line, line_to_line), DUP@turn2(prompt_to_line,
    # line_to_line), DUP@turn3(prompt_to_line, line_to_line) -> index 3 is the first DUP row's
    # line_to_line call, index 5 is the second DUP row's.
    first_dup_call = jc.calls[3]
    second_dup_call = jc.calls[5]
    assert "- DUP" not in first_dup_call["prompt"]                    # must not see its own line as "earlier"
    assert second_dup_call["prompt"].count("- DUP") == 1              # sees the first DUP row's line exactly once


def test_score_run_missing_dyads_row_errors_without_judge_call(tmp_path):
    p = log.run_paths(tmp_path, "r1")
    # deliberately no dyads.jsonl row written for this dyad/attempt
    st = log.JsonlWriter(p.status)
    st.write({"dyad_id": "d", "attempt": 1, "status": "started"}); st.write({"dyad_id": "d", "attempt": 1, "status": "complete"})
    tw = log.JsonlWriter(p.turns)
    for turn in (1, 2):
        tw.write({"run_id": "r1", "dyad_id": "d", "attempt": 1, "turn": turn, "agent": SEEKER, "text": f"s{turn}", "finish_reason": "stop"})
        tw.write({"run_id": "r1", "dyad_id": "d", "attempt": 1, "turn": turn, "agent": MENTOR, "text": f"m{turn}", "finish_reason": "stop"})
    manifest = {"seeker": {"model_sha256": "S"}, "mentor": {"model_sha256": "M"}}
    jc = FakeClient(['{"score": 0.75, "rationale": "ok"}'])
    judge = AgentHandle("judge", jc, ChatTemplate.from_source(CHATML), "J", slot=0)
    sc = Scorer("r1", 99, judge, log.JsonlWriter(p.scores), GenSettings(), clock=lambda: "T")
    n = sc.score_run(p, "pilot", manifest)
    assert n == 0 and len(jc.calls) == 0
    rows_ = log.read_jsonl(p.scores)
    assert len(rows_) == 2 * 2 + 2 * 1
    assert all(r.get("error") == "no dyads.jsonl row for this dyad/attempt" and r["score"] is None for r in rows_)
