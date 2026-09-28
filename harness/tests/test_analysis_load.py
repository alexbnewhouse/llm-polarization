"""analysis.load: the latest-complete-attempt rule, the technical exclusions with their scopes, adherence
and flags, on a synthetic run with planted anomalies."""
import json
import pytest
from analysis import load as L
from analysis.synth import ANOMALIES, SynthSpec, make_run

PLANTED = {k: 2 for k in ANOMALIES}


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("load") / "r1"
    return make_run(root, SynthSpec(run_id="r1", n_turns=12, anomalies=PLANTED))


def test_exclusions_by_reason_match_the_planted_anomalies(synth):
    run = L.load_run(synth["path"])
    t = synth["truth"]["anomalies"]
    by = {}
    for e in run.exclusions:
        by.setdefault(e["reason"], set()).add(e["dyad_id"])
    assert by["not_run"] == set(t["not_run"])
    assert by["incomplete"] == set(t["incomplete"])
    assert by["truncated"] == set(t["truncated"])
    assert by["judge_failure"] == set(t["judge_error"])
    # the retries' failed first attempts are dropped, so they leave no error rows behind
    assert "error_rows" not in by and "short_dialogue" not in by
    s = L.exclusion_summary(run)
    assert s["planned"] == synth["truth"]["n_dyads"]
    assert s["itt"] == s["planned"] - 6
    assert s["attempt_gt_1"] == 2
    assert sum(s["by_ideology"]["truncated"].values()) == 2
    md = L.exclusions_markdown(run)
    assert "ITT sample" in md and "truncated (dyad)" in md and "judge_failure (turn)" in md


def test_latest_complete_attempt_is_the_one_analysed(synth):
    run = L.load_run(synth["path"])
    for dyad_id in synth["truth"]["anomalies"]["retry"]:
        d = run.dyad(dyad_id)
        assert d["attempt"] == 2 and d["max_attempt"] == 2 and d["in_itt"]
        assert {r["attempt"] for r in run.turns if r["dyad_id"] == dyad_id} == {2}
        assert any(r["attempt"] == 1 for r in run.raw["turns"] if r["dyad_id"] == dyad_id)
    for dyad_id in synth["truth"]["anomalies"]["incomplete"]:
        d = run.dyad(dyad_id)
        assert d["attempt"] is None and not d["in_itt"]
        assert not [r for r in run.turns if r["dyad_id"] == dyad_id]


def test_controls_are_outside_the_adherence_sample_and_flags_come_from_a_threshold(synth):
    run = L.load_run(synth["path"], threshold=0.5)
    controls = [d for d in run.dyads if d["control"]]
    assert controls and all(d["in_adherence"] is None and d["flagged"] is None for d in controls)
    assert all(d["dose"] == 0 for d in controls)
    flagged = {d["dyad_id"] for d in run.dyads if d["flagged"]}
    assert flagged == set(synth["truth"]["anomalies"]["low_adherence"])
    low = run.dyad(synth["truth"]["anomalies"]["low_adherence"][0])
    assert low["adherence_mean"] < 0.4 and low["adherence_n"] == 3        # turns 4, 8, 12
    # without a threshold and without flags.jsonl nothing is flagged, and nothing pretends to be
    assert all(d["flagged"] is None for d in L.load_run(synth["path"]).dyads)


def test_flags_use_the_main_cadence_for_rows_without_a_scope(tmp_path):
    """Rows written before scope was recorded: those off the main cadence (a pilot pass scoring every turn)
    must not feed the registered flag rule."""
    res = make_run(tmp_path / "r5", SynthSpec(run_id="r5", n_per_role=1, n_turns=12))
    dyad = next(d for d in res["dyads"] if d["ideology"] == "strong_left")["dyad_id"]
    p = tmp_path / "r5" / "scores.jsonl"
    extra = [{"run_id": "r5", "dyad_id": dyad, "attempt": 1, "turn": t, "agent": "seeker",
              "metric": "prompt_to_line", "judge_sha256": SynthSpec().judge, "score": 0.1} for t in (1, 2, 3)]
    p.write_text(p.read_text() + "".join(json.dumps(r) + "\n" for r in extra))
    assert L.load_run(res["path"], threshold=0.5).dyad(dyad)["flagged"] is False


def test_flags_and_adherence_use_main_scope_rows_only(tmp_path):
    """A pilot pass on the same judge scores the main cadence's turns too: its rows say so, and neither the
    flag rule nor mean adherence reads them, although their turns are on the cadence."""
    res = make_run(tmp_path / "r6", SynthSpec(run_id="r6", n_per_role=1, n_turns=12))
    dyad = next(d for d in res["dyads"] if d["ideology"] == "strong_left")["dyad_id"]
    p = tmp_path / "r6" / "scores.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    assert {(r["scope"], r["subsample"]) for r in rows} == {("main", None)}
    before = L.load_run(res["path"], threshold=0.5).dyad(dyad)
    extra = [{"run_id": "r6", "dyad_id": dyad, "attempt": 1, "turn": t, "agent": "seeker", "scope": "pilot",
              "subsample": None, "metric": "prompt_to_line", "judge_sha256": SynthSpec().judge, "score": 0.1}
             for t in range(1, 13)]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows + extra))
    run = L.load_run(res["path"], threshold=0.5)
    d = run.dyad(dyad)
    assert d["flagged"] is False and d["adherence_mean"] == before["adherence_mean"] and d["adherence_n"] == 3
    assert len(run.scores_in("pilot")) == 12 and not any("no scope" in n for n in run.notes)
    # a flags.jsonl written by `flags --scope pilot` is not the registered rule
    (tmp_path / "r6" / "flags.jsonl").write_text(json.dumps(
        {"dyad_id": dyad, "attempt": 1, "metric": "prompt_to_line", "scope": "pilot",
         "judge_sha256": SynthSpec().judge, "flagged": True, "threshold": 0.5}) + "\n")
    assert L.load_run(res["path"]).dyad(dyad)["flagged"] is None


def test_score_scope_falls_back_to_the_turn_cadence():
    row = {"dyad_id": "d", "turn": 8, "agent": "seeker"}
    assert L.score_scope(row, 12) == "main" and L.score_scope({**row, "agent": "mentor"}, 12) == "stance"
    assert L.score_scope({**row, "turn": 7}, 12) == "pilot" and L.score_scope({**row, "turn": 7}, 7) == "main"
    assert L.score_scope({**row, "scope": "pilot"}, 12) == "pilot"


def test_survey_rows_of_other_settings_are_kept_apart(tmp_path):
    res = make_run(tmp_path / "s", SynthSpec(run_id="s", n_per_role=1, n_control=2, unconstrained=3))
    p = tmp_path / "s" / "surveys.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    free = res["truth"]["unconstrained"]["dyads"]
    # the unconstrained check never enters the analysed rows, whichever origin is analysed
    for origin in ("run", "readministered"):
        run = L.load_run(res["path"], origin=origin)
        assert all(r.get("schema", True) for r in run.surveys)
        assert len(L.unconstrained_surveys(run)) == 3 * 15
    assert not L.load_run(res["path"], origin="readministered").surveys
    assert any("unconstrained" in n for n in L.load_run(res["path"], origin="readministered").notes)
    run = L.load_run(res["path"], origin="readministered", include_unconstrained=True)
    assert {r["dyad_id"] for r in run.surveys} == set(free) and not any(r["schema"] for r in run.surveys)
    # a re-administration at temperature 0.7 is flagged in the exclusion log, not mixed into the greedy pass
    warm = [{**r, "origin": "readministered", "temperature": 0.7, "answer": 1} for r in rows
            if r["origin"] == "run" and r["dyad_id"] == free[0] and r["phase"] == "post"]
    greedy = [{**r, "origin": "readministered"} for r in rows
              if r["origin"] == "run" and r["phase"] == "post"]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows + greedy + warm))
    run = L.load_run(res["path"], origin="readministered")
    assert all(r["temperature"] == 0.0 for r in run.surveys) and len(run.surveys) == len(greedy)
    ex = [e for e in run.exclusions if e["reason"] == "survey_settings"]
    assert [(e["dyad_id"], e["scope"]) for e in ex] == [(free[0], "row")]
    assert "temperature 0.7" in ex[0]["detail"] and run.dyad(free[0])["in_itt"]
    assert "survey_settings (row)" in L.exclusions_markdown(run)
    run = L.load_run(res["path"], origin="readministered", survey_temperature=0.7)
    assert {r["temperature"] for r in run.surveys} == {0.7} and len(run.surveys) == len(warm)


def test_baseline_distribution_per_arm_and_item(tmp_path):
    from analysis.synth import make_baseline
    a = make_baseline(tmp_path / "ba", run_id="ba", k=10, mentor="arm-a")
    make_baseline(tmp_path / "bb", run_id="bb", k=5, mentor="arm-b", shift=1.0, seed=1)
    b = L.load_baseline(a["path"])
    assert b.arm == "arm-a" and b.settings["temperature"] == 0.7 and len(b.answers) == 10
    assert b.counts["error"] == 1 and b.counts["truncated"] == 1
    dist = b.distribution()
    assert dist["ideo_immigration"]["n"] == 10 and sum(dist["ideo_immigration"]["counts"].values()) == 10
    assert dist["agree_cross_partisan"]["n"] == 10        # its error row was filled in by a re-run
    assert dist["ideo_gender_racial_equality"]["missing"] == 1        # administration 2 was cut off
    c = dist["therm_independents"]["counts"]
    assert dist["therm_independents"]["mode"] == max(c, key=c.get)
    both = L.baseline_distributions([a["path"], tmp_path / "bb"])
    assert set(both) == {"arm-a", "arm-b"} and both["arm-b"]["ideo_immigration"]["n"] == 5
    with pytest.raises(ValueError, match="two baseline runs"):
        L.baseline_distributions([a["path"], a["path"]])
    with pytest.raises(ValueError, match="not a baseline run"):
        L.load_baseline(make_run(tmp_path / "r", SynthSpec(run_id="r", n_per_role=1, n_control=2))["path"])


def test_flags_jsonl_is_read_when_present(tmp_path):
    res = make_run(tmp_path / "r2", SynthSpec(run_id="r2", n_per_role=1, n_turns=12,
                                              anomalies={"low_adherence": 1}))
    dyad_id = res["truth"]["anomalies"]["low_adherence"][0]
    control = next(d["dyad_id"] for d in res["dyads"] if d["ideology"] == "none")
    rows = [{"dyad_id": dyad_id, "attempt": 1, "metric": "prompt_to_line", "judge_sha256": SynthSpec().judge,
             "flagged": True, "threshold": 0.5},
            {"dyad_id": control, "attempt": 1, "metric": "prompt_to_line", "judge_sha256": SynthSpec().judge,
             "flagged": True, "threshold": 0.5}]
    (tmp_path / "r2" / "flags.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    run = L.load_run(tmp_path / "r2")
    assert run.dyad(dyad_id)["flagged"] is True
    assert run.dyad(control)["flagged"] is None                        # control flags are ignored (F10)
    assert any("flags.jsonl" in n for n in run.notes)


def test_two_judges_need_a_prefix(tmp_path):
    res = make_run(tmp_path / "r3", SynthSpec(run_id="r3", n_per_role=1, n_control=2))
    p = tmp_path / "r3" / "scores.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    extra = [{**r, "judge_sha256": "OTHER" + r["judge_sha256"][5:]} for r in rows]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows + extra))
    with pytest.raises(ValueError, match="pass --judge"):
        L.load_run(res["path"])
    assert L.load_run(res["path"], judge="OTHER").judge.startswith("OTHER")
    with pytest.raises(ValueError, match="matches 0"):
        L.load_run(res["path"], judge="NOPE")


def test_error_rows_and_short_dialogues_in_a_complete_attempt_are_excluded(tmp_path):
    res = make_run(tmp_path / "r4", SynthSpec(run_id="r4", n_per_role=1, n_control=2))
    p = tmp_path / "r4" / "turns.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    victim, short = rows[0]["dyad_id"], rows[-1]["dyad_id"]
    rows[1] = {**rows[1], "finish_reason": "error", "error": "boom"}
    rows = [r for r in rows if not (r["dyad_id"] == short and r["turn"] == 4)]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    run = L.load_run(res["path"])
    assert "error_rows" in run.dyad(victim)["excluded"] and not run.dyad(victim)["in_itt"]
    assert "short_dialogue" in run.dyad(short)["excluded"]


def test_cli_writes_the_exclusion_log(synth, tmp_path, capsys):
    assert L.main(["--run-dir", str(synth["path"]), "--out", str(tmp_path / "o")]) == 0
    assert "# Exclusions" in capsys.readouterr().out
    for f in ("exclusions.md", "exclusions.json", "exclusions.jsonl", "dyads.csv"):
        assert (tmp_path / "o" / f).exists()
    assert json.loads((tmp_path / "o" / "exclusions.json").read_text())["by_reason"]["not_run"] == 2
    assert L.main(["--run-dir", str(tmp_path / "nope")]) == 1
