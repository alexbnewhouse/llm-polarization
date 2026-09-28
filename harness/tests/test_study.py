"""The study lock (study.json, checked by check, run and score) and the descope rule (randomize --subset-of):
gap audit F9, docs/pap/pre-analysis-plan.md sections 11 and 12."""
import json
from pathlib import Path
import pytest
from harness import grid as G, log, randomize as RZ, run as R, study as S
from harness.tests.test_run import _fake_servers, manifest_rows, write_cfg

ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "prompts" / "grid.json"
CATALOGUE = ROOT / "prompts" / "personas" / "catalogue.json"


@pytest.fixture(scope="module")
def wave(tmp_path_factory):
    """The full wave manifest at 135 per cell from the real catalogue (3 variants per level), as written."""
    d = tmp_path_factory.mktemp("wave")
    rows, assignment = RZ.build_manifest(G.load_grid(GRID), RZ.load_catalogue(CATALOGUE), seed=20261005,
                                         prefix="w1")
    out = d / "w1-dyads.jsonl"
    RZ.write_manifest(out, rows, assignment, grid_path=GRID, catalogue_path=CATALOGUE)
    return out


def test_subset_keeps_rows_by_dyad_id_and_never_re_randomizes(wave, tmp_path, capsys):
    out = tmp_path / "w1-n90-dyads.jsonl"
    assert RZ.main(["--subset-of", str(wave), "--per-variant", "30", "--control", "90", "--out", str(out)]) == 0
    assert "kept 1980 of 2970 rows" in capsys.readouterr().out
    parent = wave.read_bytes().splitlines(keepends=True)
    kept = out.read_bytes().splitlines(keepends=True)
    assert len(kept) == 1980
    # byte for byte, in the parent's order: the same dyads with the same seeds, never a new draw
    it = iter(parent)
    assert all(any(line == p for p in it) for line in kept)
    rows = [json.loads(line) for line in kept]
    per_variant, control = {}, {}
    for r in rows:
        c = r["condition"]
        if c["ideology"] == "none":
            control[c["topic"]] = control.get(c["topic"], 0) + 1
        else:
            key = (c["topic"], c["ideology"], c["openness"], c["role"])
            per_variant[key] = per_variant.get(key, 0) + 1
        assert int(r["dyad_id"][-3:]) <= (90 if c["ideology"] == "none" else 30)
    assert set(per_variant.values()) == {30} and len(per_variant) == 20 * 3
    assert control == {"immigration_enforcement": 90, "decarbonization": 90}
    a = json.loads((tmp_path / "w1-n90-assignment.json").read_text())
    assert a["kind"] == "subset" and a["rows"] == 1980 and a["rows_dropped"] == 990
    assert a["parent"]["sha256"] == log.sha256_file(wave)
    assert a["parent"]["assignment"]["sha256"] == log.sha256_file(RZ.assignment_log_path(wave))
    assert a["output"]["sha256"] == log.sha256_file(out)
    assert a["filter"] == {"per_variant": 30, "control": 90, "ideology": None, "topic": None}
    assert S.subset_parent(out) == log.sha256_file(wave) and S.subset_parent(wave) is None
    # what the audit found: re-randomizing at 90 reuses dyad_ids with other seeds
    again, _ = RZ.build_manifest(G.load_grid(GRID), RZ.load_catalogue(CATALOGUE), seed=20261005, n_per_cell=90,
                                 prefix="w1")
    by_id = {r["dyad_id"]: r["seed"] for r in rows}
    assert {r["dyad_id"] for r in again} == set(by_id)
    assert sum(by_id[r["dyad_id"]] == r["seed"] for r in again) < len(again) // 10
    R.validate_manifest_rows(rows)
    G.check_conditions(rows, G.load_grid(GRID))


def test_subset_cuts_levels_and_topics_and_refuses_what_it_cannot_subset(wave, tmp_path, capsys):
    out = tmp_path / "cut-dyads.jsonl"
    assert RZ.main(["--subset-of", str(wave), "--ideology", "strong_left,moderate,strong_right",
                    "--topic", "decarbonization", "--out", str(out)]) == 0
    rows = log.read_jsonl(out)
    assert {r["condition"]["ideology"] for r in rows} == {"strong_left", "moderate", "strong_right", "none"}
    assert {r["condition"]["topic"] for r in rows} == {"decarbonization"}
    assert len(rows) == 3 * 2 * 135 + 135                     # the control is never cut by level
    capsys.readouterr()
    for bad in (["--per-variant", "1", "--out", str(wave)],                 # the parent itself
                ["--per-variant", "1", "--out", str(wave.with_name("w1.jsonl"))],   # its assignment log
                ["--out", str(tmp_path / "all.jsonl")],                          # no filter at all
                ["--per-variant", "0", "--control", "0", "--out", str(tmp_path / "none.jsonl")]):
        assert RZ.main(["--subset-of", str(wave), *bad]) == 1
        assert capsys.readouterr().err.startswith("error: ")
    hand = tmp_path / "hand.jsonl"
    hand.write_text(json.dumps(manifest_rows(1)[0]) + "\n")
    assert RZ.main(["--subset-of", str(hand), "--per-variant", "1", "--out", str(tmp_path / "o.jsonl")]) == 1
    assert "three-digit index" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        RZ.main(["--subset-of", str(wave), "--seed", "1", "--out", str(tmp_path / "o.jsonl")])
    assert e.value.code == 2 and "nothing that would randomize" in capsys.readouterr().err


def _indexed_rows(n=3):
    rows = manifest_rows(n)
    for i, r in enumerate(rows, 1):
        r["dyad_id"] = f"w1-x-{i:03d}"
    return rows


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_the_study_lock_is_written_once_and_checked_by_check_run_and_score(tmp_path, monkeypatch, capsys):
    _fake_servers(tmp_path, monkeypatch)
    lock = tmp_path / "study.json"
    man = _write(tmp_path / "w1-dyads.jsonl", _indexed_rows(3))
    # write_cfg always writes the same file, so every call gets the config it names afresh
    cfg = lambda **over: write_cfg(tmp_path, study=str(lock), **over)
    cli = lambda cmd, c, *a: R.main([cmd, "--config", str(c() if callable(c) else c), *a])
    assert cli("study", cfg, "--manifest", str(man)) == 0
    rec = json.loads(lock.read_text())
    assert rec["run_seed"] == 5 and rec["now"] == "2026-09-08" and rec["grid"] is None
    assert rec["input_manifest"]["sha256"] == log.sha256_file(man)
    assert rec["batteries"]["sha256"] == log.sha256_file(ROOT / "instruments" / "batteries.json")
    assert rec["judge"] == {"model_sha256": "HASH-j.gguf", "family": "gemma", "model_path": "/j.gguf"}
    assert rec["harness_commit"] and rec["study_version"] == 1
    assert cli("study", cfg, "--manifest", str(man)) == 0            # written once; now only compared
    assert "matches this config" in capsys.readouterr().out
    assert cli("check", cfg, "--manifest", str(man)) == 0
    assert "ok   study" in capsys.readouterr().out
    # arm 1 runs the study's manifest; its manifest.json records the lock
    assert cli("run", cfg, "--manifest", str(man), "--run-id", "arm1") == 0
    mf = json.loads(log.run_paths(tmp_path / "data", "arm1").manifest.read_text())
    assert mf["study"] == {"path": str(lock), "sha256": log.sha256_file(lock)}
    # another run_seed, another now, or a manifest that is not the study's: refused before anything runs
    for over, expect in (({"run_seed": 6}, "run_seed is 6, study.json has 5"),
                         ({"now": "2026-10-05"}, "now is '2026-10-05'")):
        assert cli("run", cfg(**over), "--manifest", str(man), "--run-id", "arm2") == 1
        out = capsys.readouterr().out
        assert "FAIL study" in out and expect in out
    other = _write(tmp_path / "other.jsonl", manifest_rows(2))
    assert cli("run", cfg, "--manifest", str(other), "--run-id", "arm2") == 1
    assert "is not a subset of the study's" in capsys.readouterr().out
    assert not (tmp_path / "data" / "arm2" / "manifest.json").exists()
    # a descoped subset of the study's manifest is the study's manifest
    sub = tmp_path / "w1-n2-dyads.jsonl"
    RZ.write_subset(man, sub, per_variant=2)
    assert cli("run", cfg, "--manifest", str(sub), "--run-id", "arm2") == 0
    mf = json.loads(log.run_paths(tmp_path / "data", "arm2").manifest.read_text())
    assert mf["input_manifest"]["parent_sha256"] == rec["input_manifest"]["sha256"]
    # score: the study's judge scores; another judge does not, and writes no record
    capsys.readouterr()
    judge2 = lambda: cfg(judge={"url": "http://j2", "family": "gemma"})
    assert cli("score", judge2, "--run-id", "arm1", "--scope", "main") == 1
    assert "judge.model_sha256 is HASH-j2.gguf, study.json has HASH-j.gguf" in capsys.readouterr().err
    assert not list((tmp_path / "data" / "arm1").glob("judge-*.json"))
    assert cli("score", cfg, "--run-id", "arm1", "--scope", "main") == 0
    # a run started under another run_seed, without the lock, cannot be scored into the study
    assert cli("run", lambda: write_cfg(tmp_path, run_seed=9), "--manifest", str(man), "--run-id", "loose") == 0
    assert cli("score", cfg, "--run-id", "loose", "--scope", "main") == 1
    assert "run_seed is 9" in capsys.readouterr().err


def test_the_study_lock_needs_a_known_judge_and_reports_every_difference(tmp_path, monkeypatch, capsys):
    _fake_servers(tmp_path, monkeypatch)
    lock = tmp_path / "study.json"
    man = _write(tmp_path / "w1-dyads.jsonl", _indexed_rows(2))
    unknown = write_cfg(tmp_path, study=str(lock), judge={"url": "http://j"})
    assert R.main(["study", "--config", str(unknown), "--manifest", str(man)]) == 1
    assert "judge's model family is unknown" in capsys.readouterr().err and not lock.exists()
    assert R.main(["study", "--config", str(write_cfg(tmp_path)), "--manifest", str(man)]) == 1
    assert "config.study is null" in capsys.readouterr().err
    assert R.main(["check", "--config", str(write_cfg(tmp_path, study=str(lock)))]) == 1
    out = capsys.readouterr().out
    assert "FAIL study" in out and "does not exist" in out
    assert R.main(["study", "--config", str(write_cfg(tmp_path, study=str(lock))), "--manifest", str(man)]) == 0
    capsys.readouterr()
    moved = write_cfg(tmp_path, study=str(lock), run_seed=7, judge={"url": "http://j", "family": "phi"})
    assert R.main(["study", "--config", str(moved), "--manifest", str(man)]) == 1
    out = capsys.readouterr().out
    assert "FAIL study run_seed is 7" in out and "FAIL study judge.family is 'phi'" in out
    assert R.main(["check", "--config", str(write_cfg(tmp_path))]) == 0
    assert "warn study config.study is null" in capsys.readouterr().out
