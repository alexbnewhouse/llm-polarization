"""The export directory (sandbox/export.py): the repo study goes through harness.randomize and writes what the
printed `python -m harness.randomize` command writes; changed files are written beside the manifest and the
config points at them; a generic study goes through sandbox.study; a run's manifest becomes a config."""
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import pytest
from harness.log import sha256_file
from harness.run import load_config
from sandbox import export as E, repo_study as R, study as S
from sandbox.runs import RunNotFound
from sandbox.tests.fakerun import make_fake_run

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "studies" / "example-institutional-trust.study.json"


def pilot_spec() -> dict:
    """The repo study at the pilot's size: 20 treated cells x 5 x 2 modes + 2 controls x 5 = 210 rows."""
    spec = R.load_repo_study(ROOT)
    spec["randomization"] = copy.deepcopy(R.RANDOMIZATION_PRESETS["pilot"])
    return spec


def at(path: str) -> Path:
    """An export path string (root-relative inside the repo, else absolute) as a Path."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def lines(path) -> list[dict]:
    return [json.loads(line) for line in at(path).read_text(encoding="utf-8").splitlines()]


def sidecar(result) -> dict:
    return json.loads(at(result["assignment"]).read_text(encoding="utf-8"))


# --- the repo study ---------------------------------------------------------------------------------------

def test_the_repo_study_exports_what_the_printed_randomize_command_writes(tmp_path):
    spec = pilot_spec()
    result = E.export_study(spec, root=ROOT, workspace=tmp_path / "ws")
    src = {k: E.display_path(p, ROOT) for k, p in R.repo_sources(ROOT).items()}

    assert result["engine"] == "harness.randomize"
    assert result["unchanged_from_repo"] == {"grid": True, "catalogue": True, "batteries": True}
    assert (result["grid"], result["catalogue"], result["batteries"]) == (src["grid"], src["catalogue"],
                                                                          src["batteries"])
    assert result["rows"] == 210 and result["cells_filter"] is None and result["notes"] == []
    out_dir = Path(result["dir"])
    assert out_dir.parent == tmp_path / "ws" / "exports"
    assert Path(result["manifest"]).name == "p-dyads.jsonl" and Path(result["assignment"]).name == "p-assignment.json"
    assert sorted(p.name for p in out_dir.iterdir()) == ["config.json", "p-assignment.json", "p-dyads.jsonl",
                                                          "study.json"]
    assert json.loads((out_dir / "study.json").read_text(encoding="utf-8")) == spec

    config = json.loads(at(result["config"]).read_text(encoding="utf-8"))
    assert config["grid"] == "prompts/grid.json" and config["batteries"] == "instruments/batteries.json"
    assert {k: v for k, v in config.items() if k not in ("grid", "batteries")} == spec["run"]
    assert load_config(at(result["config"]))["seeker"]["url"]

    argv = result["commands"]["randomize"]
    assert argv[:3] == ["python", "-m", "harness.randomize"]
    assert argv[argv.index("--out") + 1] == result["manifest"]
    cli_out = tmp_path / "cli" / "p-dyads.jsonl"
    cli_out.parent.mkdir()
    argv = [sys.executable, *argv[1:]]
    argv[argv.index("--out") + 1] = str(cli_out)
    done = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr

    assert cli_out.read_bytes() == at(result["manifest"]).read_bytes()
    ours, theirs = sidecar(result), json.loads((cli_out.parent / "p-assignment.json").read_text(encoding="utf-8"))
    assert ours["output"]["path"] == result["manifest"]
    for a in (ours, theirs):
        del a["generated"]
        del a["output"]["path"]
    assert json.dumps(ours) == json.dumps(theirs)          # same keys, same order, same values
    assert ours["grid"]["path"] == "prompts/grid.json" and ours["catalogue"]["path"] == src["catalogue"]

    for kind in ("check", "run", "score"):
        cmd = result["commands"][kind]
        assert cmd[:4] == ["python", "-m", "harness.run", kind]
        assert cmd[cmd.index("--config") + 1] == result["config"]
    run_cmd, score_cmd = result["commands"]["run"], result["commands"]["score"]
    assert run_cmd[run_cmd.index("--run-id") + 1] == "RUN_ID" == score_cmd[score_cmd.index("--run-id") + 1]
    assert run_cmd[run_cmd.index("--manifest") + 1] == result["manifest"]
    assert score_cmd[score_cmd.index("--scope") + 1] == "pilot"


def test_a_changed_instrument_is_written_into_the_export_and_the_config_points_at_it(tmp_path):
    spec = pilot_spec()
    spec["instrument"]["items"][0]["text"] += " (edited)"
    result = E.export_study(spec, root=ROOT, workspace=tmp_path / "ws")
    assert result["engine"] == "harness.randomize"
    assert result["unchanged_from_repo"] == {"grid": True, "catalogue": True, "batteries": False}
    batteries = Path(result["dir"]) / "batteries.json"
    assert result["batteries"] == str(batteries)
    assert json.loads(batteries.read_text(encoding="utf-8")) == spec["instrument"]
    assert batteries.read_text(encoding="utf-8").endswith("}\n")
    assert json.loads(at(result["config"]).read_text(encoding="utf-8"))["batteries"] == str(batteries)


def test_a_changed_persona_template_writes_the_catalogue_and_still_uses_harness_randomize(tmp_path):
    spec = pilot_spec()
    spec["templates"]["persona"] = "Edited. " + spec["templates"]["persona"]
    result = E.export_study(spec, root=ROOT, workspace=tmp_path / "ws")
    assert result["engine"] == "harness.randomize"
    assert result["unchanged_from_repo"] == {"grid": True, "catalogue": False, "batteries": True}
    catalogue = Path(result["dir"]) / "catalogue.json"
    assert result["catalogue"] == str(catalogue) and result["grid"] == "prompts/grid.json"
    assert json.loads(catalogue.read_text(encoding="utf-8")) == R.to_repo_files(spec)[1]
    a = sidecar(result)
    assert a["catalogue"]["path"] == str(catalogue) and a["catalogue"]["sha256"] == sha256_file(catalogue)
    rows = lines(result["manifest"])
    assert all(r["persona_text"].startswith("Edited. ") for r in rows if r["condition"]["ideology"] != "none")
    cmd = result["commands"]["randomize"]
    assert cmd[cmd.index("--catalogue") + 1] == str(catalogue)


def test_a_cells_subset_keeps_the_full_exports_rows_for_those_cells(tmp_path):
    full = E.export_study(pilot_spec(), root=ROOT, workspace=tmp_path / "ws")
    spec = pilot_spec()
    keys = ["immigration_enforcement/strong_left/open", "immigration_enforcement/none"]
    spec["randomization"]["cells"] = keys
    sub = E.export_study(spec, root=ROOT, workspace=tmp_path / "ws")
    assert sub["dir"] != full["dir"]
    assert sub["engine"] == "harness.randomize" and sub["commands"]["randomize"] is None
    assert sub["cells_filter"] == keys and sub["rows"] == 15
    rows = lines(sub["manifest"])
    assert rows == [r for r in lines(full["manifest"]) if S.cell_key(spec, r["condition"]) in keys]
    a = sidecar(sub)
    assert a["cells_filter"] == keys and a["rows"] == 15
    assert a["rows_per_cell"] == {"immigration_enforcement/strong_left/open": 10, "immigration_enforcement/none": 5}
    assert a["rows_per_mode"] == {"reinforced": 10, "once": 5}
    assert set(a["rows_per_variant"]) == {f"{keys[0]}/{v['id']}" for v in spec["nested"]["variants"]["strong_left"]}
    assert a["output"]["sha256"] == sha256_file(at(sub["manifest"]))
    assert sub["commands"]["check"] and sub["commands"]["run"]


# --- a generic study --------------------------------------------------------------------------------------

def test_a_generic_study_exports_through_sandbox_study(tmp_path):
    spec = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    out = tmp_path / "out"
    result = E.export_study(spec, root=ROOT, workspace=tmp_path / "ws", out_dir=out)
    assert result["dir"] == str(out)
    assert result["engine"] == "sandbox.study" and result["commands"]["randomize"] is None
    assert result["grid"] is None and result["catalogue"] is None
    assert result["unchanged_from_repo"] == {"grid": None, "catalogue": None, "batteries": False}
    assert result["rows"] == 40 and result["cells_filter"] is None
    assert json.loads((out / "study.json").read_text(encoding="utf-8")) == spec
    assert result["study"] == str(out / "study.json")
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["grid"] is None and config["batteries"] == str(out / "batteries.json")
    assert json.loads((out / "batteries.json").read_text(encoding="utf-8")) == spec["instrument"]
    rows = lines(result["manifest"])
    assert rows == S.compile_manifest(spec)[0]
    assert Path(result["manifest"]).name == "trust-dyads.jsonl"
    a = sidecar(result)
    assert Path(result["assignment"]).name == "trust-assignment.json"
    assert a["engine"] == "sandbox.study" and a["study"]["sha256"] == S.canonical_sha256(spec)
    assert a["output"] == {"path": result["manifest"], "sha256": sha256_file(out / "trust-dyads.jsonl")}
    assert (out / "trust-dyads.jsonl").read_text(encoding="utf-8") == "".join(
        json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def test_a_study_with_errors_is_refused_before_anything_is_written(tmp_path):
    spec = pilot_spec()
    spec["randomization"]["modes"] = []
    with pytest.raises(S.StudyError) as e:
        E.export_study(spec, root=ROOT, workspace=tmp_path / "ws")
    assert S.has_errors(e.value.issues)
    assert not (tmp_path / "ws").exists()


def test_a_config_without_model_urls_is_still_written_with_a_note(tmp_path):
    spec = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    spec["run"]["seeker"]["url"] = None
    result = E.export_study(spec, root=ROOT, workspace=tmp_path / "ws")
    assert (Path(result["dir"]) / "config.json").is_file()
    assert any("seeker.url" in n for n in result["notes"])


def test_paths_are_root_relative_inside_the_root_and_absolute_outside(tmp_path):
    assert E.display_path(ROOT / "prompts" / "grid.json", ROOT) == "prompts/grid.json"
    assert E.display_path("prompts/grid.json", ROOT) == "prompts/grid.json"
    assert E.display_path(tmp_path / "x.json", ROOT) == str(tmp_path / "x.json")


def test_default_export_dirs_are_named_after_the_study_and_never_reused(tmp_path):
    spec = {"name": "My Study: (v2)!"}
    a = E.default_export_dir(tmp_path, spec)
    b = E.default_export_dir(tmp_path, spec)
    assert a.parent == b.parent == tmp_path / "exports"
    assert a.name.startswith("my-study-v2-") and a.is_dir() and b.is_dir() and a != b
    assert E.default_export_dir(tmp_path, {"name": "!!!"}).name.startswith("study-")


# --- configs from a run -------------------------------------------------------------------------------------

def test_config_from_manifest_with_and_without_a_judge_override(tmp_path):
    data = tmp_path / "data"
    make_fake_run(data, "r1", scores=False)
    manifest = json.loads((data / "r1" / "manifest.json").read_text(encoding="utf-8"))
    ws = tmp_path / "ws"

    path = E.config_from_manifest(data, "r1", ws)
    assert path == ws / "configs" / "r1.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config == {**manifest["config"], "data_dir": str(data)}
    assert load_config(path)["judge"] == manifest["config"]["judge"]

    judge = {"url": "http://127.0.0.1:8098", "gguf_path": ""}
    jpath = E.config_from_manifest(data, "r1", ws, judge=judge)
    sha8 = hashlib.sha256(json.dumps({"url": "http://127.0.0.1:8098", "gguf_path": None},
                                     sort_keys=True).encode()).hexdigest()[:8]
    assert jpath == ws / "configs" / f"r1-judge-{sha8}.json"
    jconfig = json.loads(jpath.read_text(encoding="utf-8"))
    assert jconfig["judge"] == {"url": "http://127.0.0.1:8098", "gguf_path": None}
    assert {k: v for k, v in jconfig.items() if k != "judge"} == {k: v for k, v in config.items() if k != "judge"}
    assert E.config_from_manifest(data, "r1", ws, judge=judge) == jpath

    with pytest.raises(ValueError):
        E.config_from_manifest(data, "r1", ws, judge={"url": ""})
    with pytest.raises(RunNotFound):
        E.config_from_manifest(data, "nope", ws)
    with pytest.raises(ValueError):
        E.config_from_manifest(data, "..", ws)
    (data / "bare").mkdir()
    with pytest.raises(RunNotFound):
        E.config_from_manifest(data, "bare", ws)
