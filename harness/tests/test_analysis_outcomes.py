"""analysis.outcomes: the instrument lookup by sha256, index definitions from the `indices` block and from
the Shared-definitions fallback, recoding by direction, change scores and per-index missingness."""
import json
import pytest
from analysis import outcomes as O
from analysis.load import load_run
from analysis.synth import SynthSpec, instrument, make_run


def _run(tmp_path, **kw):
    spec = SynthSpec(run_id="r", n_per_role=1, n_control=2, **kw)
    res = make_run(tmp_path / "r", spec)
    return res, load_run(res["path"])


def test_indices_from_the_instrument_block_and_recoding(tmp_path):
    res, run = _run(tmp_path)
    inst = O.load_instrument(run)
    assert set(inst.indices) >= {"ideological", "therm_gap", "affective_abs", "norms", "topic_item"}
    assert inst.indices["ideological"]["reverse"] == ["ideo_gender_racial_equality", "ideo_redistribution",
                                                      "ideo_multiculturalism", "ideo_gun_control",
                                                      "ideo_decarbonization"]
    by = inst.by_id
    ans = {i: 5 for i in by}                  # "strongly agree" with everything
    # 2 right items at 5, 5 left items reversed to 1: (2*5 + 5*1) / 7
    assert O.index_value(inst, "ideological", ans, None, {}) == pytest.approx(15 / 7)
    ans.update({"therm_rep_voters": 8, "therm_rep_politicians": 6, "therm_dem_voters": 2,
                "therm_dem_politicians": 4})
    cache = {}
    cache["therm_gap"] = O.index_value(inst, "therm_gap", ans, None, cache)
    assert cache["therm_gap"] == pytest.approx(4.0)
    assert O.index_value(inst, "affective_abs", {**ans}, None, {"therm_gap": -4.0}) == pytest.approx(4.0)
    # the topic item is the dyad's own, recoded high = right: decarbonization is a left item
    assert O.index_value(inst, "topic_item", {"ideo_decarbonization": 5}, "decarbonization", {}) == 1
    assert O.index_value(inst, "topic_item", {"ideo_enforcement_militarization": 5},
                         "immigration_enforcement", {}) == 5
    assert O.index_value(inst, "ideological", {**ans, "ideo_gun_control": None}, None, {}) is None


def test_change_scores_follow_the_planted_shift(tmp_path):
    res, run = _run(tmp_path, noise_sd=0.0, role_sd=0.0, control_shift=0.0, therm_per_dose=1.0,
                    effects={"strong_left": -1, "lean_left": 0, "moderate": 0, "lean_right": 0,
                             "strong_right": 1})
    inst = O.load_instrument(run)
    rows, counts = O.compute_outcomes(run, inst)
    by = {r["dyad_id"]: r for r in rows}
    sr = next(r for r in rows if r["ideology"] == "strong_right")
    sl = next(r for r in rows if r["ideology"] == "strong_left")
    ct = next(r for r in rows if r["control"])
    assert sr["ideological_pre"] == pytest.approx(3.0) and sr["ideological_change"] == pytest.approx(1.0)
    assert sl["ideological_change"] == pytest.approx(-1.0) and ct["ideological_change"] == pytest.approx(0.0)
    # therm: rep 4 -> 5, dem 6 -> 5 for strong_right, so the gap goes from -2 to 0
    assert sr["therm_gap_change"] == pytest.approx(2.0) and sl["therm_gap_change"] == pytest.approx(-2.0)
    assert sr["affective_abs_change"] == pytest.approx(-2.0)
    assert sl["affective_abs_change"] == pytest.approx(2.0)
    assert sr["topic_item_change"] == pytest.approx(1.0)
    assert counts["indices"][:4] == ["ideological", "therm_gap", "affective_abs", "norms"]
    assert len(by) == res["truth"]["n_dyads"]


def test_a_missing_answer_drops_one_index_not_the_dyad(tmp_path):
    res, run = _run(tmp_path, anomalies={"null_answer": 1, "salvaged": 1})
    inst = O.load_instrument(run)
    rows, counts = O.compute_outcomes(run, inst)
    null_id = res["truth"]["anomalies"]["null_answer"][0]
    r = next(r for r in rows if r["dyad_id"] == null_id)
    assert r["ideological_change"] is None and r["norms_change"] is not None and r["in_itt"]
    ex = [e for e in run.exclusions if e["dyad_id"] == null_id]
    got = [(e["reason"], e["index"], e["scope"]) for e in ex]
    assert got == [("missing_survey", "ideological", "index")]
    assert counts["null"] == 1 and counts["salvaged"] == 1
    # --json-only: the salvaged answer becomes missing too
    run2 = load_run(res["path"])
    rows2, counts2 = O.compute_outcomes(run2, inst, json_only=True)
    sal = res["truth"]["anomalies"]["salvaged"][0]
    assert next(r for r in rows2 if r["dyad_id"] == sal)["norms_change"] is None
    assert counts2["salvaged_dropped"] == 1


def test_fallback_to_shared_definitions_for_a_legacy_instrument(tmp_path):
    res, run = _run(tmp_path, instrument="legacy")
    inst = O.load_instrument(run)
    assert any("Shared definitions" in w for w in inst.warnings)
    assert any("topic items not in the instrument" in w for w in inst.warnings)
    assert "topic_item" not in inst.indices
    assert inst.indices["ideological"]["reverse"] == ["ideo_gender_racial_equality", "ideo_redistribution",
                                                      "ideo_multiculturalism", "ideo_gun_control"]
    assert inst.indices["therm_gap"]["plus"] == ["therm_rep_voters", "therm_rep_politicians"]
    rows, _ = O.compute_outcomes(run, inst)
    assert all(r["ideological_change"] is not None for r in rows if r["in_itt"])


def test_an_ideological_item_with_no_direction_is_refused():
    items = instrument("legacy")["items"] + [{"id": "ideo_new", "battery": "ideological",
                                              "scale": {"min": 1, "max": 5}, "text": "x"}]
    with pytest.raises(ValueError, match="no direction"):
        O.parse_indices({}, items, [])


def test_the_instrument_must_hash_to_what_the_run_recorded(tmp_path):
    res, run = _run(tmp_path)
    path = res["path"] / "instrument.json"
    good = path.read_text()
    path.write_text(good.replace("Rate", "Score"))
    with pytest.raises(ValueError, match="no instrument file hashing to"):
        O.load_instrument(run)
    inst = O.load_instrument(run, allow_mismatch=True)
    assert any("used anyway" in w for w in inst.warnings)
    copy = tmp_path / "archived.json"
    copy.write_text(good)
    assert O.load_instrument(run, override=str(copy)).sha256 == run.manifest["batteries"]["sha256"]


def test_the_repository_instrument_parses(tmp_path):
    """Whatever version instruments/batteries.json is at, the analysis can build every index from it."""
    data = json.loads((O.REPO / "instruments" / "batteries.json").read_text())
    idx = O.parse_indices(data, data["items"], [])
    assert {"ideological", "therm_gap", "affective_abs", "norms"} <= set(idx)


def test_cli_writes_outcomes(tmp_path, capsys):
    res, _ = _run(tmp_path)
    assert O.main(["--run-dir", str(res["path"]), "--out", str(tmp_path / "o")]) == 0
    assert "## ideological: mean post - pre" in capsys.readouterr().out
    assert (tmp_path / "o" / "outcomes.csv").read_text().startswith("run_id,")
