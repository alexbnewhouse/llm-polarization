"""sandbox.analysis: survey shifts and judge-score curves, on hand-built rows whose numbers are worked out
by hand below, and on a run written by the harness's own code (tests/fakerun.py)."""
from __future__ import annotations
import json
import math
import statistics
import pytest
from sandbox import analysis, runs
from sandbox.tests.fakerun import make_fake_run

L5 = {"min": 1, "max": 5}
T10 = {"min": 0, "max": 10}
FLAT = {"min": 1, "max": 1}


def _dyad(d, ideology, attempt=1):
    return {"dyad_id": d, "attempt": attempt, "condition": {"topic": "t", "ideology": ideology}}


def _status(d, attempt, status):
    return {"dyad_id": d, "attempt": attempt, "status": status}


def _survey(d, attempt, phase, item, answer, scale=L5, battery="b5", **extra):
    return {"dyad_id": d, "attempt": attempt, "phase": phase, "origin": "run", "item_id": item, "battery": battery,
            "scale": scale, "answer": answer, **extra}


def _pairs(d, attempt, values):
    """values: {item: (pre, post)} in item order."""
    out = []
    for item, (pre, post) in values.items():
        scale, battery = {"t1": (T10, "therm"), "z1": (FLAT, "b5")}.get(item, (L5, "b5"))
        out.append(_survey(d, attempt, "pre", item, pre, scale, battery))
        out.append(_survey(d, attempt, "post", item, post, scale, battery))
    return out


@pytest.fixture
def hand():
    """A (left) retried: attempt 1 failed with junk answers, attempt 2 complete. B (left), C (right), E (control,
    ideology null). D (left) never completed and is excluded."""
    dyads = [_dyad("A", "left", 1), _dyad("A", "left", 2), _dyad("B", "left"), _dyad("C", "right"),
             _dyad("D", "left"), _dyad("E", None)]
    status = [_status("A", 1, "started"), _status("A", 1, "failed"), _status("B", 1, "started"),
              _status("C", 1, "started"), _status("D", 1, "started"), _status("E", 1, "started"),
              _status("A", 2, "started"), _status("B", 1, "complete"), _status("C", 1, "complete"),
              _status("D", 1, "failed"), _status("E", 1, "complete"), _status("A", 2, "complete")]
    surveys = (_pairs("A", 1, {"i1": (1, 5), "i2": (1, 5), "t1": (0, 10)})                    # ignored: attempt 1
               + _pairs("A", 2, {"i1": (2, 4), "i2": (3, 3), "t1": (5, 8), "z1": (1, 1)})
               + _pairs("B", 1, {"i1": (4, 5), "i2": (None, 5), "t1": (6, 4)})
               + [_survey("B", 1, "post", "i1", 3)]                                              # duplicate: last wins
               + _pairs("C", 1, {"i1": (1, 1), "i2": (2, 5), "t1": (10, 0)})
               + _pairs("D", 1, {"i1": (1, 5)})                                                  # ignored: failed
               + _pairs("E", 1, {"i1": (3, 4), "i2": (3, 3), "t1": (5, 5)})
               + [_survey("A", 2, "post", "i1", 1, origin="readministered")])                    # ignored: origin
    del surveys[surveys.index(_survey("C", 1, "pre", "i1", 1))]["origin"]   # a row without origin counts as "run"
    return dyads, status, surveys


def test_survey_shifts_by_hand(hand):
    dyads, status, surveys = hand
    s = analysis.survey_shifts(dyads, status, surveys, factor="ideology", level_order={"ideology": ["right", "left"]})
    assert set(s) == {"factor", "levels", "n_dyads", "items", "batteries", "by_item", "by_battery"}
    assert s["factor"] == "ideology" and s["levels"] == ["right", "left", "(none)"]
    assert s["n_dyads"] == {"right": 1, "left": 2, "(none)": 1}
    assert s["items"] == [{"item_id": "i1", "battery": "b5", "scale": L5},
                          {"item_id": "i2", "battery": "b5", "scale": L5},
                          {"item_id": "t1", "battery": "therm", "scale": T10},
                          {"item_id": "z1", "battery": "b5", "scale": FLAT}]
    assert s["batteries"] == ["b5", "therm"]
    left = s["by_item"]["left"]
    # i1: A 2->4 (+2), B 4->3 (+-1, the duplicate post row's last value): mean_delta 0.5, sd sqrt(4.5), se 1.5
    assert left["i1"] == pytest.approx({"n": 2, "mean_pre": 3.0, "mean_post": 3.5, "mean_delta": 0.5, "se_delta": 1.5})
    # i2: B's pre is null, so only A's pair (3->3) counts
    assert left["i2"] == {"n": 1, "mean_pre": 3.0, "mean_post": 3.0, "mean_delta": 0.0, "se_delta": None}
    # t1: A 5->8 (+3), B 6->4 (-2): mean 0.5, sd sqrt(12.5), se 2.5
    assert left["t1"] == pytest.approx({"n": 2, "mean_pre": 5.5, "mean_post": 6.0, "mean_delta": 0.5, "se_delta": 2.5})
    assert left["z1"] == {"n": 1, "mean_pre": 1.0, "mean_post": 1.0, "mean_delta": 0.0, "se_delta": None}
    assert s["by_item"]["right"]["i2"] == {"n": 1, "mean_pre": 2.0, "mean_post": 5.0, "mean_delta": 3.0,
                                           "se_delta": None}
    assert s["by_item"]["right"]["z1"] == {"n": 0, "mean_pre": None, "mean_post": None, "mean_delta": None,
                                           "se_delta": None}
    # by battery, each dyad's mean of delta / (max - min), z1 (zero range) skipped:
    #   b5:    A mean(2/4, 0/4) = 0.25, B mean(-1/4) = -0.25 -> mean 0, sd sqrt(0.125), se 0.25
    #   therm: A 3/10 = 0.3, B -2/10 = -0.2 -> mean 0.05, se 0.25
    assert s["by_battery"]["left"]["b5"] == pytest.approx({"n": 2, "mean_delta_norm": 0.0, "se_delta_norm": 0.25})
    assert s["by_battery"]["left"]["therm"] == pytest.approx({"n": 2, "mean_delta_norm": 0.05, "se_delta_norm": 0.25})
    #   C: b5 mean(0/4, 3/4) = 0.375, therm -10/10 = -1
    assert s["by_battery"]["right"] == {"b5": {"n": 1, "mean_delta_norm": 0.375, "se_delta_norm": None},
                                        "therm": {"n": 1, "mean_delta_norm": -1.0, "se_delta_norm": None}}
    #   E (ideology null): b5 mean(1/4, 0) = 0.125, therm 0
    assert s["by_battery"]["(none)"] == {"b5": {"n": 1, "mean_delta_norm": 0.125, "se_delta_norm": None},
                                         "therm": {"n": 1, "mean_delta_norm": 0.0, "se_delta_norm": None}}
    json.dumps(s)


def test_survey_shifts_without_a_factor_and_level_orders(hand):
    dyads, status, surveys = hand
    s = analysis.survey_shifts(dyads, status, surveys)
    assert s["factor"] is None and s["levels"] == ["all"] and s["n_dyads"] == {"all": 4}
    deltas = [2, -1, 0, 1]                                     # i1: A, B, C, E
    assert s["by_item"]["all"]["i1"]["n"] == 4
    assert s["by_item"]["all"]["i1"]["mean_delta"] == pytest.approx(0.5)
    assert s["by_item"]["all"]["i1"]["se_delta"] == pytest.approx(statistics.stdev(deltas) / 2)
    # First appearance without an order; with one, unknown levels follow it and "(none)" goes last.
    assert analysis.survey_shifts(dyads, status, surveys, factor="ideology")["levels"] == ["left", "right", "(none)"]
    order = {"ideology": [None, "left"]}
    assert analysis.survey_shifts(dyads, status, surveys, factor="ideology", level_order=order)["levels"] == \
        ["left", "right", "(none)"]
    readmin = analysis.survey_shifts(dyads, status, surveys, origin="readministered")
    assert readmin["by_item"]["all"]["i1"]["n"] == 0 and [i["item_id"] for i in readmin["items"]] == ["i1"]


def _score(d, attempt, turn, judge, value, metric="alignment", agent="mentor", **extra):
    return {"dyad_id": d, "attempt": attempt, "turn": turn, "agent": agent, "metric": metric, "judge_sha256": judge,
            "score": value, **extra}


J1, J2 = "aaaa" + "1" * 60, "bbbb" + "2" * 60


@pytest.fixture
def scored(hand):
    dyads, status, _ = hand
    scores = [_score("A", 1, 1, J1, 0.0),                                    # ignored: failed attempt
              _score("A", 2, 1, J1, 0.2), _score("A", 2, 2, J1, 1.0), _score("B", 1, 1, J1, 0.6),
              _score("B", 1, 2, J1, None),                                   # unparseable: unscored
              _score("B", 1, 2, J1, None, error="judge down"),
              _score("C", 1, 1, J1, 0.9), _score("E", 1, 1, J1, 0.5), _score("D", 1, 1, J1, 0.1),
              _score("A", 2, 1, J2, 0.3), _score("B", 1, 1, J2, 0.4),
              _score("A", 2, 1, J1, 0.7, metric="prompt_to_line", agent="seeker")]
    return dyads, status, scores


def test_score_curves_refuse_two_judges_without_a_choice(scored):
    dyads, status, scores = scored
    c = analysis.score_curves(dyads, status, scores, factor="ideology")
    assert set(c) == {"metric", "judge", "judges", "factor", "levels", "error", "series"}
    assert c["error"] and "judge" in c["error"] and c["series"] == {} and c["levels"] == [] and c["judge"] is None
    assert c["judges"] == [J1, J2] and c["metric"] == "alignment"
    for prefix in ("zz", "", None):
        bad = analysis.score_curves(dyads, status, scores, factor="ideology", judge=prefix)
        assert bad["error"] and bad["series"] == {}


def test_score_curves_with_a_judge_prefix_by_hand(scored):
    dyads, status, scores = scored
    c = analysis.score_curves(dyads, status, scores, factor="ideology", judge="aaaa",
                              level_order={"ideology": ["right", "left"]})
    assert c["error"] is None and c["judge"] == J1 and c["levels"] == ["right", "left", "(none)"]
    # left turn 1: A 0.2, B 0.6 -> mean 0.4, sd sqrt(0.08), se 0.2; turn 2: A 1.0 (B's null and error rows drop)
    assert c["series"]["left"] == [pytest.approx({"turn": 1, "n": 2, "mean": 0.4, "se": 0.2}),
                                   {"turn": 2, "n": 1, "mean": 1.0, "se": None}]
    assert c["series"]["right"] == [{"turn": 1, "n": 1, "mean": 0.9, "se": None}]
    assert c["series"]["(none)"] == [{"turn": 1, "n": 1, "mean": 0.5, "se": None}]
    second = analysis.score_curves(dyads, status, scores, judge="bbbb")
    assert second["levels"] == ["all"] and second["series"]["all"] == [
        pytest.approx({"turn": 1, "n": 2, "mean": 0.35, "se": statistics.stdev([0.3, 0.4]) / math.sqrt(2)})]
    one = analysis.score_curves(dyads, status, scores, metric="prompt_to_line")   # one judge: no choice needed
    assert one["error"] is None and one["judge"] == J1 and one["series"]["all"] == [
        {"turn": 1, "n": 1, "mean": 0.7, "se": None}]
    empty = analysis.score_curves(dyads, status, [], factor="ideology")
    assert empty["error"] is None and empty["judges"] == [] and empty["series"] == {} and empty["levels"] == []
    with pytest.raises(ValueError):
        analysis.score_curves(dyads, status, scores, metric="nonsense")


# --- a run written by the harness ------------------------------------------------------------------------------

def test_analyze_run_on_a_harness_written_run(tmp_path):
    data = tmp_path / "data"
    make_fake_run(data, "r1", fail={"d02": 1, "d05": 2})
    a = analysis.analyze_run(data, "r1", factor="ideology")
    assert set(a) == {"factor", "metric", "judge", "factors", "status_counts", "metrics_available", "judges",
                      "survey", "scores"}
    assert a["factor"] == "ideology" and a["metric"] == "alignment" and a["judge"] is None
    assert a["status_counts"] == {"complete": 5, "failed": 1, "running": 0}
    assert a["factors"]["ideology"] == ["strong_left", "moderate", "strong_right", "none"]
    assert a["metrics_available"] == ["prompt_to_line", "line_to_line", "alignment"] and len(a["judges"]) == 1
    survey = a["survey"]
    assert survey["levels"] == ["strong_left", "moderate", "strong_right", "none"]       # the grid's order
    assert survey["n_dyads"] == {"strong_left": 2, "moderate": 1, "strong_right": 1, "none": 1}  # d05 never completed
    assert survey["batteries"] == ["ideological", "thermometer", "agreement"] and len(survey["items"]) == 13
    assert survey["by_battery"]["strong_left"]["thermometer"]["n"] == 2
    scores = a["scores"]
    assert scores["error"] is None and scores["judge"] == a["judges"][0] and scores["levels"] == survey["levels"]
    assert [p["turn"] for p in scores["series"]["strong_left"]] == [1, 2, 3]
    assert all(p["n"] == 2 for p in scores["series"]["strong_left"])
    json.dumps(a)

    # The retried dyad counts with its complete attempt: the pooled shift equals the per-dyad pairs that
    # dyad_detail reports for each dyad's latest complete attempt.
    pooled = analysis.analyze_run(data, "r1")
    assert pooled["survey"]["levels"] == ["all"] and pooled["survey"]["n_dyads"] == {"all": 5}
    details = [runs.dyad_detail(data, "r1", d) for d in ("d01", "d02", "d03", "d04", "d06")]
    assert details[1]["attempt"] == 2
    for item in pooled["survey"]["items"]:
        deltas = [p["delta"] for det in details for p in det["survey_pairs"] if p["item_id"] == item["item_id"]]
        got = pooled["survey"]["by_item"]["all"][item["item_id"]]
        assert got["n"] == 5 and got["mean_delta"] == pytest.approx(statistics.fmean(deltas))
        assert got["se_delta"] == pytest.approx(statistics.stdev(deltas) / math.sqrt(5))

    assert analysis.analyze_run(data, "r1", factor="role", metric="prompt_to_line")["scores"]["error"] is None
    with pytest.raises(ValueError):
        analysis.analyze_run(data, "r1", factor="no_such_key")
    with pytest.raises(runs.RunNotFound):
        analysis.analyze_run(data, "nope")


def test_analyze_run_follows_a_given_level_order(tmp_path):
    data = tmp_path / "data"
    make_fake_run(data, "r1", scores=False)
    order = {"ideology": ["none", "strong_right", "moderate", "strong_left"]}
    a = analysis.analyze_run(data, "r1", factor="ideology", level_order=order)
    assert a["survey"]["levels"] == ["none", "strong_right", "moderate", "strong_left"]
    assert a["factors"]["ideology"] == ["none", "strong_right", "moderate", "strong_left"]
    assert a["scores"]["series"] == {} and a["scores"]["error"] is None and a["judges"] == []
