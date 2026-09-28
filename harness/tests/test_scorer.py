"""The judge: target selection per scope, idempotent scoring, judge refusals, the flag rule, agreement."""
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
    assert scorer.parse_score('{"score": 0.5, "rationale": null}') == (0.5, "")   # not the string "None"


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
    manifest = {"seeker": {"model_sha256": "S"}, "mentor": {"model_sha256": "M", "family": "qwen"}}
    return p, manifest


def make_scorer(tmp_path, judge_hash="J", replies=None):
    jc = FakeClient(replies or ['{"score": 0.75, "rationale": "ok"}'])
    judge = AgentHandle("judge", jc, ChatTemplate.from_source(CHATML), judge_hash, slot=0, family="gemma")
    p, manifest = make_run(tmp_path)
    sc = Scorer("r1", 99, judge, log.JsonlWriter(p.scores), GenSettings(), clock=lambda: "T",
                harness_commit="COMMIT")
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
    assert r0["seed"] == log.derive_seed(99, 1, "d", 1, r0["turn"], f"judge:{r0['agent']}:{r0['metric']}")
    assert all(r["id_slot"] == 0 and r["harness_commit"] == "COMMIT" for r in rows_)
    assert r0["prompt_chars"] == len(jc.calls[0]["prompt"])
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
    # Two seeker rows are fully value-identical: same turn (2), same agent (SEEKER), same text
    # ("DUP") -- a duplicate append after a crash-retry, or a repeated turn number. Any position
    # lookup keyed by a value derived from the row's contents (list.index(row)'s dict equality,
    # or a (turn, agent)-derived key) cannot tell the two rows apart and collapses them to one
    # index, which can feed the FIRST row its own line back as an "earlier" one. Resolving
    # position by object identity (id(row)) has neither failure mode.
    p = log.run_paths(tmp_path, "r1")
    log.JsonlWriter(p.dyads).write({"dyad_id": "d", "attempt": 1, "condition": {"topic": "immigration"},
                                    "persona_text": "PERSONA", "persona_reminder": "", "persona_mode": "once",
                                    "seed": 1, "n_turns": 3})
    st = log.JsonlWriter(p.status)
    st.write({"dyad_id": "d", "attempt": 1, "status": "started"}); st.write({"dyad_id": "d", "attempt": 1, "status": "complete"})
    tw = log.JsonlWriter(p.turns)
    tw.write({"run_id": "r1", "dyad_id": "d", "attempt": 1, "turn": 1, "agent": SEEKER, "text": "S1", "finish_reason": "stop"})
    tw.write({"run_id": "r1", "dyad_id": "d", "attempt": 1, "turn": 2, "agent": SEEKER, "text": "DUP", "finish_reason": "stop"})
    tw.write({"run_id": "r1", "dyad_id": "d", "attempt": 1, "turn": 2, "agent": SEEKER, "text": "DUP", "finish_reason": "stop"})
    tw.write({"run_id": "r1", "dyad_id": "d", "attempt": 1, "turn": 3, "agent": SEEKER, "text": "S3", "finish_reason": "stop"})
    manifest = {"seeker": {"model_sha256": "S"}, "mentor": {"model_sha256": "M", "family": "qwen"}}
    jc = FakeClient(['{"score": 0.5, "rationale": "x"}'])
    judge = AgentHandle("judge", jc, ChatTemplate.from_source(CHATML), "J", slot=0, family="gemma")
    sc = Scorer("r1", 99, judge, log.JsonlWriter(p.scores), GenSettings(), clock=lambda: "T")
    sc.score_run(p, "pilot", manifest)
    # pilot order (seeker rows only here): S1(prompt_to_line, line_to_line), DUP#1@turn2(prompt_to_line,
    # line_to_line), DUP#2@turn2(prompt_to_line, line_to_line), S3(prompt_to_line, line_to_line) ->
    # index 3 is the first DUP row's line_to_line call, index 5 is the second DUP row's.
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
    manifest = {"seeker": {"model_sha256": "S"}, "mentor": {"model_sha256": "M", "family": "qwen"}}
    jc = FakeClient(['{"score": 0.75, "rationale": "ok"}'])
    judge = AgentHandle("judge", jc, ChatTemplate.from_source(CHATML), "J", slot=0, family="gemma")
    sc = Scorer("r1", 99, judge, log.JsonlWriter(p.scores), GenSettings(), clock=lambda: "T")
    n = sc.score_run(p, "pilot", manifest)
    assert n == 0 and len(jc.calls) == 0
    rows_ = log.read_jsonl(p.scores)
    assert len(rows_) == 2 * 2 + 2 * 1
    assert all(r.get("error") == "no dyads.jsonl row for this dyad/attempt" and r["score"] is None for r in rows_)


# --- persona-stability decision, amendment 3 (docs/decisions/persona-stability.md §3) ---------------------

def test_model_family_from_path_or_alias():
    f = scorer.model_family
    assert f("/models/Qwen3.6-35B-A3B-Q4_K_M.gguf") == "qwen"
    assert f("gpt-oss-20b-mxfp4.gguf") == "gpt-oss"
    assert f("Olmo-3-7B-Instruct-Q8_0.gguf") == "olmo"
    assert f("glm-4.7-flash") == "glm" and f("gemma-3-27b-it") == "gemma"
    assert f("Meta-Llama-3.1-8B") == "llama" and f("Mistral-7B") == "mistral" and f("Phi-4") == "phi"
    assert f("DeepSeek-R1-Distill") == "deepseek"
    assert f("/fake/model.gguf") is None and f("") is None and f(None) is None


def test_score_run_refuses_a_judge_from_the_mentors_family(tmp_path):
    # The mentor's stance score is the turn-level DV; a same-family judge is self-favouring and shares
    # its political priors. Not the seeker's family: that rule is "not the seeker of this dialogue".
    sc, _, p, manifest = make_scorer(tmp_path)
    del manifest["mentor"]["family"]
    manifest["mentor"]["model_path"] = "/m/Olmo-3-7B-Instruct.gguf"
    sc.judge.family = "olmo"
    with pytest.raises(ValueError, match="family"):
        sc.score_run(p, "pilot", manifest)
    sc.judge.family = "qwen"
    assert sc.score_run(p, "pilot", manifest) > 0


def test_score_run_judge_of_the_seekers_family_is_allowed_but_an_unknown_family_refuses(tmp_path):
    # Red-team H3: a mentor served from an ollama blob (sha256-...) with no --alias has no detectable family,
    # and a same-family judge used to be accepted in silence. Unknown now refuses unless the config says.
    sc, _, p, manifest = make_scorer(tmp_path)
    manifest["seeker"]["model_path"] = "/s/Qwen3-4B.gguf"
    del manifest["mentor"]["family"]
    manifest["mentor"]["model_path"] = "/home/alex/.ollama/models/blobs/sha256-d372de8e"
    sc.judge.family = "qwen"
    with pytest.raises(ValueError, match="mentor.family"):
        sc.score_run(p, "pilot", manifest)
    with pytest.raises(ValueError, match="same-family|mentor's model family"):
        sc.score_run(p, "pilot", manifest, mentor_family="Qwen3.6")        # the config states it
    assert sc.score_run(p, "pilot", manifest, mentor_family="gpt-oss") > 0
    sc.judge.family = None
    with pytest.raises(ValueError, match="judge.family"):
        sc.score_run(p, "pilot", manifest, mentor_family="gpt-oss")


def test_model_family_covers_the_study_models_and_their_directories():
    f = scorer.model_family
    for name, family in (("QwQ-32B-Q4_K_M.gguf", "qwen"),
                         ("/models/Qwen3.6-35B-A3B/model-Q4_K_M.gguf", "qwen"),
                         ("gpt-oss-20b-mxfp4.gguf", "gpt-oss"), ("OLMo-2-1124-13B.gguf", "olmo"),
                         ("GLM-4.7-Flash-Q4.gguf", "glm"), ("Llama-3.1-8B.gguf", "llama"),
                         ("gemma-3-27b.gguf", "gemma"), ("Ministral-8B.gguf", "mistral"),
                         ("DeepSeek-V3.gguf", "deepseek"), ("phi4-mini.gguf", "phi")):
        assert f(name) == family, name
    assert f("/home/alex/.ollama/models/blobs/sha256-d372de8e") is None
    assert f("/home/alex/llama.cpp/models/blobs/sha256-d372de8e") is None     # only the nearest directory
    assert scorer.declared_family("Qwen3") == "qwen" and scorer.declared_family(" Granite ") == "granite"
    assert scorer.declared_family(None) is None and scorer.declared_family("") is None


def test_a_second_judge_scores_the_same_targets_again(tmp_path):
    # Two judges on the stance metric is the cross-judge agreement design: rows are done per judge, so a
    # second judge does not find the first one's rows and skip everything.
    sc, jc, p, manifest = make_scorer(tmp_path, judge_hash="J1")
    n1 = sc.score_run(p, "pilot", manifest)
    judge2 = AgentHandle("judge", FakeClient(['{"score": 0.5, "rationale": "x"}']),
                         ChatTemplate.from_source(CHATML), "J2", slot=0, family="gemma")
    sc2 = Scorer("r1", 99, judge2, log.JsonlWriter(p.scores), GenSettings(), clock=lambda: "T")
    n2 = sc2.score_run(p, "pilot", manifest)
    assert n1 == n2 == 6
    assert sc.score_run(p, "pilot", manifest) == 0 and sc2.score_run(p, "pilot", manifest) == 0
    hashes = {r["judge_sha256"] for r in log.read_jsonl(p.scores)}
    assert hashes == {"J1", "J2"}


def test_select_targets_stance_scope_is_mentor_alignment_on_the_main_cadence():
    out = []
    for turn in range(1, 10):
        out.append({"dyad_id": "d", "attempt": 1, "turn": turn, "agent": SEEKER, "text": "s", "finish_reason": "stop"})
        out.append({"dyad_id": "d", "attempt": 1, "turn": turn, "agent": MENTOR, "text": "m", "finish_reason": "stop"})
    stance = scorer.select_targets(out, "stance")
    assert all(r["agent"] == MENTOR and m == "alignment" for r, m in stance)
    assert sorted(r["turn"] for r, _ in stance) == [4, 8, 9]


def test_subsample_dyads_is_deterministic_and_keeps_whole_dyads():
    ids = [f"d{i:03d}" for i in range(400)]
    a = scorer.subsample_dyads(ids, 0.25, run_seed=7)
    b = scorer.subsample_dyads(ids, 0.25, run_seed=7)
    assert a == b and 60 <= len(a) <= 140 and set(a) <= set(ids)
    assert scorer.subsample_dyads(ids, 0.25, run_seed=8) != a
    assert scorer.subsample_dyads(ids, 1.0, run_seed=7) == set(ids)
    with pytest.raises(ValueError):
        scorer.subsample_dyads(ids, 0.0, run_seed=7)
    rows = [{"dyad_id": d, "attempt": 1, "turn": 4, "agent": MENTOR, "text": "m", "finish_reason": "stop"} for d in ids]
    picked = scorer.select_targets(rows, "stance", subsample=0.25, run_seed=7)
    assert {r["dyad_id"] for r, _ in picked} == a


def _score_row(dyad, turn, score, metric="prompt_to_line", judge="J", attempt=1, agent=SEEKER):
    return {"dyad_id": dyad, "attempt": attempt, "turn": turn, "agent": agent, "metric": metric,
            "judge_sha256": judge, "score": score}


def test_flag_rule_three_consecutive_scored_turns_under_threshold():
    # "Three consecutive" means three consecutive SCORED seeker turns, not three dialogue turns: in main
    # scope the seeker is scored on a turn-4 cadence, so turns 4, 8, 12 under threshold is a flag.
    rows = [_score_row("a", t, s) for t, s in ((4, 0.9), (8, 0.5), (12, 0.4), (16, 0.3), (20, 0.9))]
    rows += [_score_row("b", t, s) for t, s in ((4, 0.5), (8, 0.9), (12, 0.5), (16, 0.9), (20, 0.5))]
    rows += [_score_row("c", t, s) for t, s in ((4, 0.5), (8, 0.5))]            # only two scored turns
    flags = scorer.flag_dialogues(rows, threshold=0.6)
    assert flags[("a", 1)]["flagged"] is True and flags[("a", 1)]["first_flag_turn"] == 16
    assert flags[("b", 1)]["flagged"] is False and flags[("b", 1)]["first_flag_turn"] is None
    assert flags[("c", 1)]["flagged"] is False
    assert flags[("a", 1)]["scored_turns"] == 5 and flags[("a", 1)]["turns_under"] == 3
    assert flags[("a", 1)]["min_score"] == 0.3


def test_flag_rule_a_score_at_threshold_is_not_under_it_and_null_scores_are_skipped():
    rows = [_score_row("a", t, s) for t, s in ((1, 0.6), (2, 0.6), (3, 0.6))]
    assert scorer.flag_dialogues(rows, threshold=0.6)[("a", 1)]["flagged"] is False
    # a judge failure (null score) is an unscored turn: it neither counts toward the run nor breaks it
    rows = [_score_row("a", t, s) for t, s in ((1, 0.1), (2, None), (3, 0.1), (4, 0.1))]
    f = scorer.flag_dialogues(rows, threshold=0.6)[("a", 1)]
    assert f["flagged"] is True and f["first_flag_turn"] == 4 and f["unscored_turns"] == 1


def test_flag_rule_uses_one_metric_one_judge_and_seeker_rows_only():
    rows = [_score_row("a", t, 0.1) for t in (1, 2, 3)]
    rows += [_score_row("a", t, 0.1, metric="alignment", agent=MENTOR) for t in (1, 2, 3)]
    rows += [_score_row("a", t, 0.9, metric="line_to_line") for t in (1, 2, 3)]
    assert scorer.flag_dialogues(rows, threshold=0.6)[("a", 1)]["flagged"] is True
    assert scorer.flag_dialogues(rows, threshold=0.6, metric="line_to_line")[("a", 1)]["flagged"] is False
    with pytest.raises(ValueError):
        scorer.flag_dialogues(rows, threshold=0.6, metric="alignment")       # a mentor metric is not adherence
    two_judges = rows + [_score_row("a", t, 0.9, judge="J2") for t in (1, 2, 3)]
    with pytest.raises(ValueError, match="judge"):
        scorer.flag_dialogues(two_judges, threshold=0.6)
    assert scorer.flag_dialogues(two_judges, threshold=0.6, judge_sha256="J2")[("a", 1)]["flagged"] is False
    with pytest.raises(ValueError):
        scorer.flag_dialogues(rows, threshold=0.6, run_length=0)


def test_cross_judge_agreement_on_shared_targets():
    j1 = [_score_row("a", t, s, metric="alignment", agent=MENTOR, judge="J1") for t, s in ((4, 0.2), (8, 0.5), (12, 0.9), (16, 0.7))]
    j2 = [_score_row("a", t, s, metric="alignment", agent=MENTOR, judge="J2") for t, s in ((4, 0.3), (8, 0.4), (12, 1.0), (20, 0.1))]
    j2_null = [_score_row("a", 8, None, metric="alignment", agent=MENTOR, judge="J2")]      # ignored
    a = scorer.cross_judge_agreement(j1 + j2 + j2_null, metric="alignment")
    assert a["metric"] == "alignment" and a["judges"] == ["J1", "J2"]
    pair = a["pairs"][0]
    assert pair["judges"] == ["J1", "J2"] and pair["n"] == 3        # turns 4, 8, 12 are shared; 16 and 20 are not
    assert abs(pair["mean_abs_diff"] - 0.1) < 1e-9 and pair["pearson_r"] > 0.95
    assert pair["within_0.1"] == 1.0
    assert a["per_judge"]["J1"]["n"] == 4 and a["per_judge"]["J2"]["n"] == 4      # the null row is not a score
    assert scorer.cross_judge_agreement(j1, metric="alignment")["pairs"] == []
