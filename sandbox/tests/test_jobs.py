"""Jobs: the argv is built only from validated fields, matching harness/run.py's options; a job's status
and meaning follow its exit code; stop is SIGINT then SIGKILL; records survive a restart."""
from __future__ import annotations
import json
import sys
import time
import pytest
from harness import run as R
from sandbox.jobs import EXIT_MEANINGS, JOB_KINDS, JobConflict, JobManager, build_argv, exit_meaning

PY = sys.executable
HR = [PY, "-m", "harness.run"]


# ------------------------------------------------------------------------------------------------ argv

def test_build_argv_for_every_kind():
    assert build_argv("check", {"config": "c.json"}) == HR + ["check", "--config", "c.json"]
    assert build_argv("check", {"config": "c.json", "manifest": "m.jsonl"}) == \
        HR + ["check", "--config", "c.json", "--manifest", "m.jsonl"]
    assert build_argv("run", {"config": "c.json", "manifest": "m.jsonl", "run_id": "w1-2026.10"}) == \
        HR + ["run", "--config", "c.json", "--run-id", "w1-2026.10", "--manifest", "m.jsonl"]
    assert build_argv("survey", {"config": "c.json", "run_id": "r", "phase": "pre"}) == \
        HR + ["survey", "--config", "c.json", "--run-id", "r", "--phase", "pre"]
    assert build_argv("survey", {"config": "c.json", "run_id": "r"})[-2:] == ["--phase", "post"]
    assert build_argv("score", {"config": "c.json", "run_id": "r"}) == \
        HR + ["score", "--config", "c.json", "--run-id", "r", "--scope", "pilot"]
    assert build_argv("score", {"config": "c.json", "run_id": "r", "scope": "stance", "subsample": "0.25"})[-4:] == \
        ["--scope", "stance", "--subsample", "0.25"]
    assert build_argv("flags", {"config": "c.json", "run_id": "r", "threshold": 0.4}) == \
        HR + ["flags", "--config", "c.json", "--run-id", "r", "--threshold", "0.4"]
    assert build_argv("flags", {"config": "c.json", "run_id": "r", "threshold": "1", "metric": "line_to_line",
                                "run_length": 4, "judge": "ABC123"})[-7:] == \
        ["1.0", "--metric", "line_to_line", "--run-length", "4", "--judge", "abc123"]
    assert build_argv("agreement", {"config": "c.json", "run_id": "r"}) == \
        HR + ["agreement", "--config", "c.json", "--run-id", "r"]
    assert build_argv("agreement", {"config": "c.json", "run_id": "r", "metric": "line_to_line"})[-2:] == \
        ["--metric", "line_to_line"]
    assert build_argv("check", {"config": "c.json", "manifest": ""}, python="/opt/py")[:1] == ["/opt/py"]


def test_every_built_argv_parses_with_the_harness_parser(monkeypatch):
    """The argv must be exactly what harness.run.main accepts: run each through its argparse."""
    seen = []
    monkeypatch.setattr(R, "load_config", lambda path: {"path": path})
    for name in ("cmd_check", "cmd_run", "cmd_survey", "cmd_score", "cmd_flags", "cmd_agreement"):
        monkeypatch.setattr(R, name, lambda *a, _n=name: seen.append((_n, a[1:])) or 0)
    fields = {"check": {"config": "c", "manifest": "m"}, "run": {"config": "c", "manifest": "m", "run_id": "r"},
              "survey": {"config": "c", "run_id": "r", "phase": "pre"},
              "score": {"config": "c", "run_id": "r", "scope": "main", "subsample": 0.5},
              "flags": {"config": "c", "run_id": "r", "threshold": 0.3, "metric": "prompt_to_line",
                        "run_length": 2, "judge": "beef"},
              "agreement": {"config": "c", "run_id": "r", "metric": "alignment"}}
    for kind in JOB_KINDS:
        assert R.main(build_argv(kind, fields[kind])[3:]) == 0
    assert seen == [("cmd_check", ("m",)), ("cmd_run", ("m", "r")), ("cmd_survey", ("r", "pre")),
                    ("cmd_score", ("r", "main", 0.5)), ("cmd_flags", ("r", 0.3, "prompt_to_line", 2, "beef")),
                    ("cmd_agreement", ("r", "alignment"))]


@pytest.mark.parametrize("kind, fields, field", [
    ("deploy", {"config": "c"}, "kind"),
    ("check", {}, "config"),
    ("check", {"config": "c", "extra": 1}, "extra"),
    ("check", {"config": "--help"}, "config"),
    ("check", {"config": "c\0x"}, "config"),
    ("check", {"config": 5}, "config"),
    ("run", {"config": "c", "manifest": "m"}, "run_id"),
    ("run", {"config": "c", "run_id": "r"}, "manifest"),
    ("run", {"config": "c", "manifest": "-x", "run_id": "r"}, "manifest"),
    ("run", {"config": "c", "manifest": "m", "run_id": "--x"}, "run_id"),
    ("run", {"config": "c", "manifest": "m", "run_id": "-r"}, "run_id"),
    ("run", {"config": "c", "manifest": "m", "run_id": "a/b"}, "run_id"),
    ("run", {"config": "c", "manifest": "m", "run_id": ".."}, "run_id"),
    ("run", {"config": "c", "manifest": "m", "run_id": "a b"}, "run_id"),
    ("survey", {"config": "c", "run_id": "r", "phase": "mid"}, "phase"),
    ("score", {"config": "c", "run_id": "r", "scope": "all"}, "scope"),
    ("score", {"config": "c", "run_id": "r", "subsample": 0}, "subsample"),
    ("score", {"config": "c", "run_id": "r", "subsample": 1.5}, "subsample"),
    ("score", {"config": "c", "run_id": "r", "subsample": "half"}, "subsample"),
    ("flags", {"config": "c", "run_id": "r"}, "threshold"),
    ("flags", {"config": "c", "run_id": "r", "threshold": "high"}, "threshold"),
    ("flags", {"config": "c", "run_id": "r", "threshold": "nan"}, "threshold"),
    ("flags", {"config": "c", "run_id": "r", "threshold": True}, "threshold"),
    ("flags", {"config": "c", "run_id": "r", "threshold": 2}, "threshold"),
    ("flags", {"config": "c", "run_id": "r", "threshold": 0.5, "metric": "alignment"}, "metric"),
    ("flags", {"config": "c", "run_id": "r", "threshold": 0.5, "run_length": 0}, "run_length"),
    ("flags", {"config": "c", "run_id": "r", "threshold": 0.5, "run_length": 2.5}, "run_length"),
    ("flags", {"config": "c", "run_id": "r", "threshold": 0.5, "judge": "--x"}, "judge"),
    ("flags", {"config": "c", "run_id": "r", "threshold": 0.5, "judge": "xyz"}, "judge"),
    ("agreement", {"config": "c", "run_id": "r", "metric": "vibes"}, "metric"),
])
def test_build_argv_refuses_bad_fields(kind, fields, field):
    with pytest.raises(ValueError, match=field):
        build_argv(kind, fields)


def test_exit_meanings():
    assert EXIT_MEANINGS[0] == "ok" and EXIT_MEANINGS[2] == "some dyads failed"
    assert exit_meaning(130).startswith("stopped") and exit_meaning(None) is None
    assert exit_meaning(-2) == "ended by SIGINT" and exit_meaning(-9) == "ended by SIGKILL"
    assert exit_meaning(3) == "exit code 3"


# ------------------------------------------------------------------------------------------------ jobs

@pytest.fixture
def jm(tmp_path):
    return JobManager(tmp_path, tmp_path / "jobs")


def _exit_argv(code: int) -> list[str]:
    return [PY, "-c", f"import sys; print('hi'); sys.exit({code})"]


def _wait_for_log(jm, job_id, needle, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if f"\n{needle}\n" in jm.read_log(job_id)[0]:       # a line of its own, not the "$ ..." header
            return
        time.sleep(0.02)
    raise AssertionError(f"{needle!r} never appeared in the log")


@pytest.mark.parametrize("code, status", [(0, "finished"), (1, "failed"), (2, "failed")])
def test_exit_code_gives_status_meaning_and_log(jm, tmp_path, code, status):
    job = jm.start_argv(_exit_argv(code), kind="check", label="probe")
    assert job["status"] == "running" and job["returncode"] is None and job["meaning"] is None
    assert job["kind"] == "check" and job["label"] == "probe" and job["argv"] == _exit_argv(code)
    done = jm.wait(job["id"], timeout=10)
    assert done["status"] == status and done["returncode"] == code and done["meaning"] == EXIT_MEANINGS[code]
    assert done["ended_at"] and done["started_at"]
    text, offset = jm.read_log(job["id"])
    assert "hi\n" in text and text.startswith("$ ") and f"exit {code}" in text
    assert offset == len(text.encode("utf-8"))
    on_disk = json.loads((tmp_path / "jobs" / f"{job['id']}.json").read_text())
    assert on_disk == done


def test_job_runs_in_root_with_unbuffered_python(jm, tmp_path):
    job = jm.start_argv([PY, "-c", "import os; print(os.getcwd()); print(os.environ['PYTHONUNBUFFERED'])"],
                        kind="check")
    jm.wait(job["id"], timeout=10)
    text = jm.read_log(job["id"])[0]
    assert str(tmp_path.resolve()) in text and "\n1\n" in text


def test_job_ids_are_safe_and_unique(jm):
    ids = [jm.start_argv(_exit_argv(0), kind="score")["id"] for _ in range(3)]
    for i in ids:
        jm.wait(i, timeout=10)
    assert len(set(ids)) == 3
    for i in ids:
        stamp_date, stamp_time, kind, n = i.split("-")
        assert len(stamp_date) == 8 and len(stamp_time) == 6 and kind == "score" and n.isdigit()
    assert [j["id"] for j in jm.list()] == ids[::-1]          # newest first
    with pytest.raises(KeyError):
        jm.get("nope")
    with pytest.raises(KeyError):
        jm.read_log("nope")


def test_stop_sends_sigint_and_ends_as_stopped(jm):
    job = jm.start_argv([PY, "-c", "import time; print('ready', flush=True); time.sleep(30)"], kind="run",
                        run_id="r1")
    _wait_for_log(jm, job["id"], "ready")
    assert jm.stop(job["id"])["stop_requested"] is True
    done = jm.wait(job["id"], timeout=10)
    assert done["status"] == "stopped" and done["returncode"] == -2 and done["meaning"] == "ended by SIGINT"
    assert jm.stop(job["id"]) == done                            # an ended job is left alone


def test_a_harness_style_interrupt_exits_130(jm):
    code = ("import sys, time\nprint('ready', flush=True)\ntry:\n    time.sleep(30)\n"
            "except KeyboardInterrupt:\n    print('finishing in-flight dyads')\n    sys.exit(130)\n")
    job = jm.start_argv([PY, "-c", code], kind="run", run_id="r2")
    _wait_for_log(jm, job["id"], "ready")
    jm.stop(job["id"])
    done = jm.wait(job["id"], timeout=10)
    assert done["status"] == "stopped" and done["returncode"] == 130
    assert done["meaning"] == "stopped - resume with the same run id"
    assert "finishing in-flight dyads" in jm.read_log(done["id"])[0]


def test_second_stop_kills_a_job_that_ignores_sigint(jm):
    code = ("import signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            "print('ready', flush=True)\ntime.sleep(30)\n")
    job = jm.start_argv([PY, "-c", code], kind="run", run_id="r3")
    _wait_for_log(jm, job["id"], "ready")
    jm.stop(job["id"])
    assert jm.wait(job["id"], timeout=0.3)["status"] == "running"
    jm.stop(job["id"])
    done = jm.wait(job["id"], timeout=10)
    assert done["status"] == "stopped" and done["returncode"] == -9


def test_a_second_job_of_the_same_kind_on_the_same_run_is_refused(jm):
    sleeper = [PY, "-c", "import time; time.sleep(30)"]
    job = jm.start_argv(sleeper, kind="run", run_id="dup")
    try:
        with pytest.raises(JobConflict):
            jm.start_argv(sleeper, kind="run", run_id="dup")
        other = jm.start_argv(_exit_argv(0), kind="score", run_id="dup")      # another kind is fine
        jm.wait(other["id"], timeout=10)
    finally:
        jm.stop(job["id"])
        jm.stop(job["id"])
        jm.wait(job["id"], timeout=10)


def test_start_builds_the_argv_from_fields(tmp_path):
    jm = JobManager(tmp_path, tmp_path / "jobs", python=PY)
    job = jm.start("check", {"kind": "check", "config": "missing-config.json", "label": "from the API"})
    assert job["argv"] == HR + ["check", "--config", "missing-config.json"] and job["run_id"] is None
    assert job["label"] == "from the API"
    with pytest.raises(ValueError, match="threshold"):
        jm.start("flags", {"config": "c", "run_id": "r"})
    jm.wait(job["id"], timeout=30)


def test_a_command_that_cannot_start_is_a_failed_job(jm):
    job = jm.start_argv(["/nonexistent/python-binary"], kind="check")
    assert job["status"] == "failed" and job["meaning"].startswith("could not start")
    assert "could not start" in jm.read_log(job["id"])[0]


def test_reload_keeps_records_and_marks_running_ones_lost(tmp_path):
    log_dir = tmp_path / "jobs"
    first = JobManager(tmp_path, log_dir)
    done = first.wait(first.start_argv(_exit_argv(2), kind="check")["id"], timeout=10)
    ghost = {"id": "20261007-120000-run-7", "kind": "run", "label": "", "argv": ["x"], "run_id": "w1",
             "status": "running", "returncode": None, "meaning": None, "started_at": "2026-10-07T12:00:00+0000",
             "ended_at": None, "log_path": str(log_dir / "20261007-120000-run-7.log"), "pid": 999999}
    (log_dir / f"{ghost['id']}.json").write_text(json.dumps(ghost))
    (log_dir / "garbage.json").write_text("{not json")
    second = JobManager(tmp_path, log_dir)
    assert second.get(done["id"]) == done
    lost = second.get(ghost["id"])
    assert lost["status"] == "lost" and "pid 999999" in lost["meaning"]
    assert json.loads((log_dir / f"{ghost['id']}.json").read_text())["status"] == "lost"
    assert second.stop(ghost["id"]) == lost and second.wait(ghost["id"], timeout=0.1) == lost
    assert second.read_log(ghost["id"]) == ("", 0)
    new = second.start_argv(_exit_argv(0), kind="check")
    assert int(new["id"].rsplit("-", 1)[1]) > 7
    second.wait(new["id"], timeout=10)


def test_read_log_offsets_and_utf8_boundaries(jm):
    job = jm.start_argv([PY, "-c", "print('héllo wörld ' * 3)"], kind="check")
    jm.wait(job["id"], timeout=10)
    full, end = jm.read_log(job["id"])
    assert "héllo wörld" in full and end == len(full.encode("utf-8"))
    assert jm.read_log(job["id"], end) == ("", end)
    assert jm.read_log(job["id"], end + 100) == ("", end)
    # Read in 5-byte pieces: the pieces join back to the whole text without a replacement character.
    pieces, offset = [], 0
    while offset < end:
        text, new = jm.read_log(job["id"], offset, max_bytes=5)
        assert new > offset
        pieces.append(text)
        offset = new
    assert "".join(pieces) == full and "�" not in full
    with pytest.raises(ValueError):
        jm.read_log(job["id"], -1)


def test_read_log_holds_back_a_split_character_while_running(jm, tmp_path):
    job = jm.start_argv([PY, "-c", "import sys, time; sys.stdout.buffer.write(b'ab\\xc3'); sys.stdout.flush(); "
                                   "time.sleep(30)"], kind="run", run_id="utf")
    try:
        log_path = tmp_path / "jobs" / f"{job['id']}.log"
        deadline = time.monotonic() + 5
        while not log_path.read_bytes().endswith(b"ab\xc3") and time.monotonic() < deadline:
            time.sleep(0.02)
        text, offset = jm.read_log(job["id"])
        assert text.endswith("ab") and "�" not in text
        assert offset == log_path.stat().st_size - 1                 # the lead byte waits for the next read
    finally:
        jm.stop(job["id"])
        jm.wait(job["id"], timeout=10)


def test_a_job_records_the_data_dir_its_config_names(tmp_path):
    """So the GUI can open the run a job writes without knowing which study started it."""
    jm = JobManager(tmp_path, tmp_path / "jobs", python=PY)
    (tmp_path / "with.json").write_text(json.dumps({"data_dir": "/scratch/runs"}))
    (tmp_path / "without.json").write_text(json.dumps({"run_seed": 1}))
    jobs = [jm.start("check", {"config": c}) for c in ("with.json", "without.json", "missing.json")]
    assert [j["data_dir"] for j in jobs] == ["/scratch/runs", "data", None]
    assert jm.wait(jobs[0]["id"], timeout=30)["data_dir"] == "/scratch/runs"
    for j in jobs[1:]:
        jm.wait(j["id"], timeout=30)
    assert JobManager(tmp_path, tmp_path / "jobs", python=PY).get(jobs[0]["id"])["data_dir"] == "/scratch/runs"
