"""End to end through the real CLI: the mock backend stands in for three llama-servers, the repo study is
exported exactly as the GUI exports it, and `check`, `run` and `score` run as the subprocesses the GUI
launches. What comes back through `runs` and `analysis` is what the GUI shows. Skipped without gguf-py,
which the harness needs to read a chat template out of a GGUF."""
from __future__ import annotations
import json
from pathlib import Path
import pytest

pytest.importorskip("gguf")

from sandbox import analysis, export, jobs, repo_study, runs          # noqa: E402
from sandbox.mock_server import MockBackend                           # noqa: E402

TREATED = "immigration_enforcement/strong_left/open"
CONTROL = "immigration_enforcement/none"


@pytest.fixture
def backend(tmp_path):
    b = MockBackend(tmp_path / "mock", slots=4)
    yield b.start()
    b.stop()


def _finish(jm, job, timeout=180):
    done = jm.wait(job["id"], timeout=timeout)
    log, _ = jm.read_log(job["id"])
    assert done["status"] == "finished", f"{job['kind']} ended {done['status']} rc={done['returncode']}:\n{log}"
    return done, log


def test_repo_study_subset_runs_through_the_cli_and_reads_back(repo_root, tmp_path, backend):
    spec = repo_study.load_repo_study(repo_root)
    data_dir = tmp_path / "data"
    spec["randomization"].update({"seed": 7, "n_per_cell": 1, "modes": ["reinforced"], "n_turns": 3,
                                  "prefix": "e2e", "cells": [TREATED, CONTROL]})
    spec["run"].update({"data_dir": str(data_dir), "gguf_py_path": None, "concurrency": None,
                        "seeker": backend["seeker"], "mentor": backend["mentor"], "judge": backend["judge"]})
    spec["run"]["generation"] = dict(spec["run"]["generation"], n_predict=60)

    result = export.export_study(spec, root=repo_root, workspace=tmp_path / "ws")
    assert result["engine"] == "harness.randomize" and result["rows"] == 2
    assert result["unchanged_from_repo"]["batteries"] is True
    config_path = Path(result["config"])
    config = json.loads((config_path if config_path.is_absolute() else repo_root / config_path).read_text())
    assert config["grid"] and config["batteries"] == "instruments/batteries.json"

    jm = jobs.JobManager(repo_root, tmp_path / "jobs")
    _finish(jm, jm.start("check", {"config": result["config"], "manifest": result["manifest"]}))
    _, log = _finish(jm, jm.start("run", {"config": result["config"], "manifest": result["manifest"], "run_id": "e2e"}))
    assert "2 complete, 0 failed" in log

    summary = runs.run_summary(data_dir, "e2e")
    assert summary["status_counts"]["complete"] == 2 and summary["planned"] == 2
    assert {d["condition"]["ideology"] for d in summary["dyads"]} == {"strong_left", "none"}
    assert all(d["turns_done"] == 3 for d in summary["dyads"])
    assert summary["roles"]["mentor"]["model_sha256"] != summary["roles"]["seeker"]["model_sha256"]

    _finish(jm, jm.start("score", {"config": result["config"], "run_id": "e2e", "scope": "pilot"}))

    dyad = next(d for d in summary["dyads"] if d["condition"]["ideology"] == "strong_left")
    detail = runs.dyad_detail(data_dir, "e2e", dyad["dyad_id"])
    assert len(detail["turns"]) == 6 and [t["agent"] for t in detail["turns"][:2]] == ["seeker", "mentor"]
    assert not any(t["cache_warning"] for t in detail["turns"])
    assert len(detail["survey_pairs"]) == 13 and all(p["delta"] is not None for p in detail["survey_pairs"])
    assert detail["scores"]

    a = analysis.analyze_run(data_dir, "e2e", factor="ideology", metric="alignment",
                             level_order=runs.level_order(data_dir, "e2e", root=repo_root))
    assert a["survey"]["levels"] == ["strong_left", "none"]
    assert a["survey"]["by_battery"]["strong_left"]["thermometer"]["n"] == 1
    assert a["scores"]["error"] is None and a["scores"]["series"]["strong_left"]
