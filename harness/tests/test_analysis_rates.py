"""analysis.rates: every appendix rate counts exactly the anomalies planted in a synthetic run, by arm and
condition; the refusal detector's hits and misses."""
import json
import pytest
from analysis import rates as R
from analysis.load import load_run
from analysis.synth import ANOMALIES, SynthSpec, make_run

N = 3


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("rates")
    return {arm: make_run(root / arm, SynthSpec(run_id=arm, mentor=f"m-{arm}", n_turns=12,
                                               anomalies={k: N for k in ANOMALIES}, seed=i))
            for i, arm in enumerate(("a", "b"))}


def test_planted_anomalies_are_counted_exactly(synth):
    res = synth["a"]
    run = load_run(res["path"], threshold=0.5)
    r, hits = R.run_rates(run, "ideology")
    k = lambda rate: r[rate]["all"][0]  # noqa: E731
    n_analysed = res["truth"]["n_dyads"] - 2 * N                   # not_run and incomplete have no attempt
    assert k("cache_warning") == N and k("truncated") == N and k("truncated_dyads") == N
    assert k("finish_length") == N and k("finish_length_mentor") == N and k("finish_length_seeker") == 0
    assert k("attempt_gt1_analysed") == N and k("attempt_gt1_ever") == N
    assert r["attempt_gt1_ever"]["all"][1] == n_analysed + N       # incomplete dyads have status rows
    assert k("survey_null_post") == N and k("survey_null_pre") == 0
    assert k("survey_salvaged_post") == N and k("method_bare_post") == N and k("method_none_post") == N
    assert k("refusal_turns") == N and k("refusal_dyads") == N and r["refusal_dyads"]["all"][1] == n_analysed
    assert k("disclaimer_turns") == N and k("disclaimer_dyads") == N
    assert k("survey_refusal_post") == N                           # the planted null answers are refusals
    assert k("judge_null") == N and k("judge_error_unresolved") == N
    assert k("flagged") == N
    # the message-row denominator is every analysed message: 2 per turn
    assert r["cache_warning"]["all"][1] == n_analysed * 2 * 12
    # every group count sums to the overall one
    assert sum(v[0] for g, v in r["refusal_turns"].items() if g != "all") == N
    turn_hits = [h for h in hits if h["source"] == "turn"]
    assert {h["pattern"] for h in turn_hits} >= {"sorry_but_cant", "as_an_ai"}
    refused = {h["dyad_id"] for h in turn_hits if h["category"] == "refusal"}
    assert refused == set(res["truth"]["anomalies"]["refusal"])


def test_breakdowns_and_the_attempt_test(synth):
    run = load_run(synth["a"]["path"])
    for by in R.BY:
        r, _ = R.run_rates(run, by)
        assert r["cache_warning"]["all"][0] == N
    r, _ = R.run_rates(run, "topic_ideology")
    assert any(g.startswith("decarbonization/") for g in r["refusal_turns"])
    t = R.attempt_test([run])[run.run_id]
    assert t["ideology"]["df"] >= 1 and sum(t["ideology"]["retried"]) == N and 0 <= t["ideology"]["p"] <= 1


@pytest.mark.parametrize("text,expect", [
    ("I'm sorry, but I can't help with that.", "refusal"),
    ("I cannot provide personal opinions on political matters.", "refusal"),
    ("I won't take sides on this issue.", "refusal"),
    ("I must decline to answer that.", "refusal"),
    ("That falls outside my guidelines.", "refusal"),
    ("I’m unable to share my views on that.", "refusal"),
    ("As an AI, I don't have personal opinions, but here is what the evidence says.", "disclaimer"),
    ("I'm just an AI, so take this with a grain of salt.", "disclaimer"),
    ("I can't help but notice that your ranch depends on seasonal workers.", None),
    ("Here is some practical advice about the permits.", None),
    ("You can't provide for your workers if the farm goes under.", None),
    ("", None),
])
def test_refusal_detector(text, expect):
    cats = {c for _, c, _ in R.detect(text)}
    if expect is None:
        assert not cats
    else:
        assert expect in cats
    assert R.is_refusal(text) == (expect == "refusal")


def test_cli_writes_rates_for_two_arms(synth, tmp_path, capsys):
    out = tmp_path / "o"
    assert R.main(["--run-dir", str(synth["a"]["path"]), "--run-dir", str(synth["b"]["path"]),
                   "--by", "topic_ideology", "--threshold", "0.5", "--out", str(out)]) == 0
    md = capsys.readouterr().out
    assert "m-a (a)" in md and "m-b (b)" in md and "## refusal_turns by topic_ideology" in md
    data = json.loads((out / "rates.json").read_text())
    assert data["rates"]["m-b (b)"]["flagged"]["all"][0] == N
    assert {p["name"] for p in data["patterns"]} >= {"cant_act", "as_an_ai"}
    hits = [json.loads(line) for line in (out / "refusals.jsonl").read_text().splitlines()]
    assert all(h["pattern"] and h["match"] for h in hits) and len(hits) >= 4 * N
    assert R.main(["--run-dir", str(tmp_path / "missing")]) == 1
