"""analysis.calibrate: the stratified label sheet, and judge agreement and the flag threshold read back
from a labelled sheet."""
import csv
import json
import pytest
from analysis import calibrate as C
from analysis.load import load_run
from analysis.synth import SynthSpec, make_run


@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    root = tmp_path_factory.mktemp("cal") / "pilot"
    return make_run(root, SynthSpec(run_id="pilot", n_turns=12, anomalies={"low_adherence": 20}))


def test_sheet_is_stratified_blind_and_never_overwritten(pilot, tmp_path, capsys):
    rows = C.sample_turns(pilot["path"], n=120, seed=1)
    assert len(rows) == 120 and len({(r["dyad_id"], r["turn"]) for r in rows}) == 120
    strata = {}
    for r in rows:
        strata[(r["ideology"], r["turn_bin"])] = strata.get((r["ideology"], r["turn_bin"]), 0) + 1
    assert len(strata) == 20 and set(strata.values()) == {6}          # 5 levels x 4 turn bins, even
    assert all(r["agent"] == "seeker" and r["ideology"] != "none" and r["label"] == "" for r in rows)
    assert all("score" not in k for k in rows[0])                     # the labeller never sees the judge
    assert C.sample_turns(pilot["path"], n=120, seed=1) == rows       # deterministic
    out = tmp_path / "labels.csv"
    assert C.main(["sample", "--run-dir", str(pilot["path"]), "--n", "40", "--out", str(out)]) == 0
    header = out.read_text().splitlines()[0].split(",")
    assert header[:4] == ["sample_id", "dyad_id", "attempt", "turn"]
    assert "text" in header and "label" in header
    # refuses to overwrite a sheet
    assert C.main(["sample", "--run-dir", str(pilot["path"]), "--out", str(out)]) == 1


def _label(pilot, tmp_path, noise_flip=0):
    """Fill a sheet with labels on 1-5 derived from the judge's own score (4-5 adherent, 1-2 not)."""
    run = load_run(pilot["path"])
    score = {(s["dyad_id"], s["turn"]): s["score"] for s in run.scores
             if s["metric"] == "prompt_to_line" and s.get("score") is not None}
    rows = [r for r in C.sample_turns(pilot["path"], n=400, seed=2) if (r["dyad_id"], r["turn"]) in score]
    for i, r in enumerate(rows):
        s = score[(r["dyad_id"], r["turn"])]
        r["label"] = 5 if s > 0.8 else 4 if s > 0.6 else 2 if s > 0.35 else 1
        if i < noise_flip:
            r["label"] = 1 if r["label"] >= 4 else 5
        r["label2"] = r["label"] if i % 3 else ""
    path = tmp_path / "labels.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=C.SHEET_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path, rows


def test_agreement_and_threshold_from_labels(pilot, tmp_path):
    path, rows = _label(pilot, tmp_path)
    res = C.score_labels(pilot["path"], path, adherent_at=4)
    assert res["n_pairs"] == len(rows) and res["unmatched"] == 0
    # labels are coarse (a 0.85 score is labelled 5), so agreement is high but not perfect
    assert res["krippendorff_alpha_interval"] > 0.7 and res["weighted_kappa_quadratic"] > 0.7
    assert res["alpha_ci95"][0] <= res["krippendorff_alpha_interval"] <= res["alpha_ci95"][1]
    t = res["threshold"]
    # planted: adherent turns score about 0.85, low-adherence ones about 0.3
    assert 0.3 < t["threshold"] < 0.8 and t["balanced_accuracy"] == pytest.approx(1.0)
    assert res["inter_labeller_alpha"] == pytest.approx(1.0) and res["n_double_labelled"] > 0
    (tmp_path / "noisy").mkdir()
    flipped, _ = _label(pilot, tmp_path / "noisy", noise_flip=15)
    worse = C.score_labels(pilot["path"], flipped, adherent_at=4)
    assert worse["krippendorff_alpha_interval"] < res["krippendorff_alpha_interval"]
    assert worse["threshold"]["balanced_accuracy"] < 1.0


def test_score_cli_and_bad_labels(pilot, tmp_path, capsys):
    path, _ = _label(pilot, tmp_path)
    out = tmp_path / "cal.json"
    assert C.main(["score", "--run-dir", str(pilot["path"]), "--labels", str(path), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "Krippendorff alpha" in printed and "flags --threshold" in printed
    assert json.loads(out.read_text())["threshold"]["threshold"] is not None
    text = path.read_text().splitlines()
    bad = tmp_path / "bad.csv"
    first = text[1].split(",")
    first[C.SHEET_COLUMNS.index("label")] = "9"
    bad.write_text("\n".join([text[0], ",".join(first)] + text[2:]) + "\n")
    assert C.main(["score", "--run-dir", str(pilot["path"]), "--labels", str(bad), "--out", str(out)]) == 1
