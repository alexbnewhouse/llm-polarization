"""analysis.estimate on synthetic runs with planted effects: the dose slope, the level contrasts, the shape
test in both directions, the arm interaction, H3 at the turn level, per-protocol and the robustness
checks."""
import json
import pytest
from analysis import estimate as E
from analysis.load import load_run
from analysis.outcomes import compute_outcomes, item_changes, load_instrument
from analysis.synth import ANOMALIES, SynthSpec, make_run

NULL = {"strong_left": 0, "lean_left": 0, "moderate": 0, "lean_right": 0, "strong_right": 0}


def _fit(tmp_path, specs, outcomes=("ideological",), threshold=None, sensitivity=False):
    runs, rows, items, sens = [], [], [], {}
    for spec in specs:
        res = make_run(tmp_path / spec.run_id, spec)
        run = load_run(res["path"], threshold=threshold)
        inst = load_instrument(run)
        runs.append(run)
        rows += compute_outcomes(run, inst)[0]
        items += item_changes(run, inst)
        if sensitivity:
            sens.setdefault("json-only", []).extend(
                compute_outcomes(run, inst, json_only=True, log=False)[0])
    return E.estimate(runs, rows, items, list(outcomes), sensitivity=sens)


def test_the_planted_dose_slope_is_recovered(tmp_path):
    res = _fit(tmp_path, [SynthSpec(run_id="a", n_per_role=5, n_control=30, role_sd=0.05)])
    for h in res["H1"]:
        assert abs(h["estimate"] - 0.25) < 3 * h["se"] and h["p_holm"] < 0.001, h
    lv = res["outcomes"]["ideological"]["level"]["immigration_enforcement"]
    # level effects against the control: 0.25 per dose step (the control shift cancels)
    assert lv["tau"]["strong_right"]["ci"][0] < 0.5 < lv["tau"]["strong_right"]["ci"][1]
    assert lv["tau"]["strong_left"]["ci"][0] < -0.5 < lv["tau"]["strong_left"]["ci"][1]
    assert lv["control_mean"]["ci"][0] < 0.1 < lv["control_mean"]["ci"][1]
    assert "p_holm_within_topic" in lv["tau"]["moderate"]
    assert res["H2"][0]["classification"] in ("symmetric-drift-consistent", "inconclusive")
    assert res["H2"][0]["classification"] != "accommodation-consistent"


def test_no_effect_is_not_found(tmp_path):
    res = _fit(tmp_path, [SynthSpec(run_id="a", n_per_role=4, n_control=24, effects=NULL, control_shift=0.0,
                                    role_sd=0.0)])
    for h in res["H1"]:
        assert abs(h["estimate"]) < 3 * h["se"] and h["p_holm"] > 0.05
    assert all(h["classification"] != "accommodation-consistent" for h in res["H2"])


def test_the_shape_test_reads_an_accommodation_pattern(tmp_path):
    effects = {**NULL, "lean_right": 0.5, "strong_right": 0.8}
    res = _fit(tmp_path, [SynthSpec(run_id="a", n_per_role=5, n_control=45, effects=effects, role_sd=0.05,
                                    control_shift=0.0)])
    for h in res["H2"]:
        assert h["A"]["estimate"] == pytest.approx(0.65, abs=0.2)
        assert h["p_holm"] < 0.05 and h["classification"] == "accommodation-consistent", h
        assert not E.stats.fieller_contains(h["rho"], 1.0)
    # the symmetric pattern with the same magnitude is not read as accommodation
    sym = {"strong_left": -0.8, "lean_left": -0.5, "moderate": 0, "lean_right": 0.5, "strong_right": 0.8}
    res = _fit(tmp_path / "s", [SynthSpec(run_id="b", n_per_role=5, n_control=45, effects=sym, role_sd=0.05,
                                          control_shift=0.0)])
    assert [h["classification"] for h in res["H2"]] == ["symmetric-drift-consistent"] * 2


def test_arms_pool_with_fixed_effects_and_s1_finds_the_arm_difference(tmp_path):
    specs = [SynthSpec(run_id="a", mentor="arm-a", n_per_role=4, n_control=24, role_sd=0.05),
             SynthSpec(run_id="b", mentor="arm-b", n_per_role=4, n_control=24, role_sd=0.05,
                       effect_scale=0.0, noise_seed=11)]
    res = _fit(tmp_path, specs)
    assert res["arms"] == ["arm-a", "arm-b"]
    s1 = res["outcomes"]["ideological"]["arm_x"]["decarbonization"]
    assert s1["arm_x_wald"]["p"] < 0.001
    assert s1["per_arm_slope"]["arm-a"]["ci"][0] < 0.25 < s1["per_arm_slope"]["arm-a"]["ci"][1]
    assert s1["per_arm_slope"]["arm-b"]["ci"][0] < 0 < s1["per_arm_slope"]["arm-b"]["ci"][1]
    assert any(s["test"].startswith("S1") and s["q_bh"] < 0.05 for s in res["secondary"])
    rob = res["robustness"]
    assert set(rob["by_arm_variance"]["decarbonization"]["weights"]) == {"arm-a", "arm-b"}
    assert rob["two_stage"]["decarbonization"]["n_units"] == 30


def test_h3_turn_level_recovers_the_planted_adherence_slope(tmp_path):
    # the synthetic mentor's alignment is 0.5 + 0.4 (adherence - 0.5): agree = 2a - 1 moves 0.8 per unit
    spec = SynthSpec(run_id="a", n_turns=12, stance=True, anomalies={"low_adherence": 30})
    res = _fit(tmp_path, [spec], threshold=0.5)
    h3 = res["H3"]
    assert h3["judge"] == spec.judge
    theta = h3["turn"]["pooled"]["theta"]
    assert theta["ci"][0] < 0.8 < theta["ci"][1] and theta["p"] < 0.001
    assert h3["turn_cumulative"]["pooled"]["theta"]["estimate"] > 0.5
    assert h3["control_alignment"]["n"] > 0                           # reported apart, never modelled
    assert set(h3["trajectory"]) >= {"strong_left", "moderate", "strong_right"}
    assert sum(v["flagged"] for v in h3["adherence_drift"].values()) == 30
    assert any(s["test"].startswith("H3 theta") for s in res["secondary"])


def test_per_protocol_drops_flagged_dyads_and_keeps_every_control(tmp_path):
    spec = SynthSpec(run_id="a", n_turns=12, anomalies={"low_adherence": 6})
    res = _fit(tmp_path, [spec], threshold=0.5)
    o = res["outcomes"]["ideological"]
    for t in ("decarbonization", "immigration_enforcement"):
        itt, pp = o["level"][t], o["per_protocol"]["level"][t]
        assert pp["n_control"] == itt["n_control"]
    assert (sum(o["level"][t]["n"] for t in res["topics"]) - sum(o["per_protocol"]["level"][t]["n"]
                                                                  for t in res["topics"])) == 6
    res2 = _fit(tmp_path / "noflags", [SynthSpec(run_id="b")])
    assert res2["outcomes"]["ideological"]["per_protocol"]["skipped"].startswith("no flags")


def test_two_stage_and_stacked_agree_with_the_mixed_model(tmp_path):
    res = _fit(tmp_path, [SynthSpec(run_id="a", n_per_role=4, n_control=24)], sensitivity=True)
    for t in res["topics"]:
        mixed = res["outcomes"]["ideological"]["slope"][t]["slope"]["estimate"]
        ts = res["robustness"]["two_stage"][t]["slope"]["estimate"]
        st = res["robustness"]["stacked"][t]["slope"]["slope"]["estimate"]
        assert ts == pytest.approx(mixed, abs=0.02) and st == pytest.approx(mixed, abs=0.02)
        assert res["robustness"]["sensitivity"]["json-only"][t]["slope"]["estimate"] == pytest.approx(mixed)
    assert res["robustness"]["lee_bounds"]["skipped"]


def test_lee_bounds_when_incompleteness_depends_on_level(tmp_path):
    res = make_run(tmp_path / "a", SynthSpec(run_id="a", n_per_role=4, n_control=24))
    p = tmp_path / "a" / "status.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    drop = {d["dyad_id"] for d in res["dyads"] if d["ideology"] == "strong_right"}
    gone = set(sorted(drop)[:12])       # half the decarbonization strong_right dyads never complete
    p.write_text("".join(json.dumps(r) + "\n" for r in rows
                         if not (r["dyad_id"] in gone and r["status"] == "complete")))
    run = load_run(res["path"])
    inst = load_instrument(run)
    out = E.estimate([run], compute_outcomes(run, inst)[0], [], ["ideological"])
    assert out["runs"][0]["incompleteness_test"]["p"] < 0.05
    lb = out["robustness"]["lee_bounds"]["decarbonization"]["strong_right"]
    assert lb["trimmed"] == "control" and lb["lower"] <= lb["upper"]


def test_cli_pools_two_arms_and_refuses_one_arm_twice(tmp_path, capsys):
    for rid, arm in (("a", "arm-a"), ("b", "arm-b")):
        make_run(tmp_path / rid, SynthSpec(run_id=rid, mentor=arm, n_per_role=2, n_control=6,
                                           anomalies={k: 1 for k in ANOMALIES}))
    out = tmp_path / "o"
    dirs = ["--run-dir", str(tmp_path / "a"), "--run-dir", str(tmp_path / "b")]
    assert E.main(dirs + ["--out", str(out)]) == 0
    md = capsys.readouterr().out
    assert "## Confirmatory" in md and "H2, shape test" in md and "Secondary family" in md
    data = json.loads((out / "estimates.json").read_text())
    assert data["arms"] == ["arm-a", "arm-b"] and "sensitivity" in data["robustness"]
    make_run(tmp_path / "c", SynthSpec(run_id="c", mentor="arm-a", n_per_role=1, n_control=2))
    assert E.main(["--run-dir", str(tmp_path / "a"), "--run-dir", str(tmp_path / "c"), "--no-sensitivity",
                   "--out", str(out)]) == 1


def test_signed_stance_is_centred_and_undefined_without_a_direction():
    assert E.signed_stance(1.0, 2) == 1.0 and E.signed_stance(1.0, -1) == -1.0
    assert E.signed_stance(0.5, 2) == 0.0 and E.signed_stance(0.0, -2) == 1.0
    assert E.signed_stance(0.9, 0) is None and E.signed_stance(0.9, None) is None
