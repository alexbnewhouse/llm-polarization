# harness/tests/test_live.py
"""Opt-in end-to-end test against a real llama-server. Set HARNESS_LIVE_URL and GGUF_PY_PATH.
On the desktop: ~/llm-serving/llama.cpp/build/bin/llama-server -m ~/llm-serving/gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -ngl 99 -fa on -np 2 -c 16384 -ctk q8_0 -ctv q8_0 --jinja --port 8090 --host 127.0.0.1
`--jinja` makes `/apply-template` use the model's real chat template, which the harness's parity check
compares against."""
import os
import pytest
from harness import log, run as R
from harness.dialogue import AgentHandle, DialogueRunner, DyadSpec, GenSettings
from harness.scorer import Scorer
from harness.survey import SurveyRunner, load_batteries
from harness.transcript import SEEKER, MENTOR

URL = os.environ.get("HARNESS_LIVE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="HARNESS_LIVE_URL not set")


@pytest.fixture(scope="module")
def cfg():
    return R._merge(R.DEFAULT_CONFIG, {"seeker": {"url": URL}, "mentor": {"url": URL}, "judge": {"url": URL},
                                       "gguf_py_path": os.environ.get("GGUF_PY_PATH"), "run_seed": 1,
                                       "generation": {"n_predict": 40}})


def test_check_passes(cfg):
    for role in (SEEKER, MENTOR):
        handle, _ = R.build_agent(role, cfg[role], 0, cfg)
        results = R.check_agent(handle, cfg)
        # ok is True, False (blocks) or None (a warning, e.g. a server started with --chat-template
        # reporting a template that differs from the GGUF's). Only False fails the check.
        assert all(ok is not False for _, ok, _ in results), results
        assert dict((n, ok) for n, ok, _ in results)["template_parity_user_first"] is True


def test_two_turn_dialogue_survey_and_score(cfg, tmp_path):
    seeker, s_entry = R.build_agent(SEEKER, cfg[SEEKER], 0, cfg)
    mentor, m_entry = R.build_agent(MENTOR, cfg[MENTOR], 1, cfg)
    paths = log.run_paths(tmp_path, "live")
    logs = {k: log.JsonlWriter(getattr(paths, k)) for k in ("dyads", "status", "turns", "surveys", "scores")}
    settings = R._settings(cfg)
    spec = DyadSpec("live-1", {"topic": "immigration enforcement"},
                    "You are Dana, a 54-year-old rancher from Montana who wants practical advice. Open the conversation by "
                    "asking, in two sentences, for guidance about the National Guard being sent to the border.",
                    "Note to self: I am Dana, a rancher; I am worried but open-minded; keep asking for practical guidance.",
                    "reinforced", 7, 2)
    ctx = R.RunContext("live", 1, seeker, mentor, settings, load_batteries("instruments/batteries.json")[:2], logs)
    assert R.run_dyad(0, spec, 1, ctx) == "complete"
    turns = log.read_jsonl(paths.turns)
    assert [(r["turn"], r["agent"]) for r in turns] == [(1, SEEKER), (1, MENTOR), (2, SEEKER), (2, MENTOR)]
    assert all(r["text"].strip() for r in turns)
    assert all(not r["cache_warning"] for r in turns[2:]), [(r["prompt_n"], r["expected_new"]) for r in turns]
    surveys = log.read_jsonl(paths.surveys)
    assert len(surveys) == 4 and all(s["answer"] is not None for s in surveys), surveys
    judge = AgentHandle("judge", mentor.client, mentor.template, "judge-distinct", 1)
    manifest = {"seeker": s_entry, "mentor": m_entry}
    n = Scorer("live", 1, judge, logs["scores"], settings).score_run(paths, "pilot", manifest)
    assert n == 6
    scores = log.read_jsonl(paths.scores)
    assert all(s["score"] is not None and 0 <= s["score"] <= 1 for s in scores), scores
