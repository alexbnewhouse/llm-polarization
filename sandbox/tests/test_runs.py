"""sandbox.runs over run directories written by the harness's own code (tests/fakerun.py)."""
from __future__ import annotations
import json
import os
from pathlib import Path
import pytest
from harness import run as R
from harness.log import JsonlWriter, read_jsonl
from harness.transcript import message_order
from sandbox import runs
from sandbox.tests.fakerun import DEFAULT_CONDITIONS, make_fake_run

REPO = Path(__file__).resolve().parents[2]
ROLE_INFO = {"url", "alias", "model_path", "model_sha256", "family", "template_sha256", "build_info", "total_slots"}


@pytest.fixture
def data(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def retried_run(data):
    """d02 fails once and completes as attempt 2; d05 fails both attempts; d07 is in flight."""
    root = make_fake_run(data, "r1", fail={"d02": 1, "d05": 2})
    JsonlWriter(root / "dyads.jsonl").write({"run_id": "r1", "dyad_id": "d07", "attempt": 1,
                                              "condition": dict(DEFAULT_CONDITIONS[0]), "persona_text": "P",
                                              "persona_reminder": "R", "persona_mode": "reinforced", "seed": 7,
                                              "n_turns": 3, "ts": "2026-10-07T10:00:00+0000"})
    JsonlWriter(root / "status.jsonl").write({"run_id": "r1", "dyad_id": "d07", "attempt": 1, "status": "started",
                                               "ts": "2026-10-07T10:00:01+0000"})
    return root


def _set_mtime(root: Path, t: float):
    for p in [root, *root.iterdir()]:
        os.utime(p, (t, t))


# --- ids and paths -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["..", ".", "a/b", "-x", "", "a b", "../data", "x\\y", "r1\n", None, 3])
def test_bad_run_ids_are_refused(data, bad):
    (data / "r1").mkdir(parents=True)
    with pytest.raises(ValueError):
        runs.run_dir(data, bad)
    with pytest.raises(ValueError):
        runs.run_summary(data, bad)
    with pytest.raises(ValueError):
        runs.tail(data, bad, {})


def test_bad_dyad_ids_are_refused(data, retried_run):
    for bad in ("..", "a/b", "-x", ""):
        with pytest.raises(ValueError):
            runs.dyad_detail(data, "r1", bad)


def test_missing_run_is_not_found(data, retried_run):
    with pytest.raises(runs.RunNotFound):
        runs.run_dir(data, "nope")
    (data / "afile").write_text("x")
    with pytest.raises(runs.RunNotFound):
        runs.run_dir(data, "afile")
    assert runs.run_dir(data, "r1") == retried_run.resolve()


def test_a_run_dir_that_resolves_outside_data_dir_is_refused(tmp_path, data, retried_run):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "manifest.json").write_text("{}")
    (data / "escape").symlink_to(outside, target_is_directory=True)
    (data / "self").symlink_to(data, target_is_directory=True)
    for name in ("escape", "self"):
        with pytest.raises(ValueError):
            runs.run_dir(data, name)
    assert [r["run_id"] for r in runs.list_runs(data)] == ["r1"]


def test_resolve_data_dir(tmp_path):
    assert runs.resolve_data_dir(tmp_path) == (tmp_path / "data").resolve()
    assert runs.resolve_data_dir(tmp_path, "other/d") == (tmp_path / "other" / "d").resolve()
    assert runs.resolve_data_dir(tmp_path, str(tmp_path / "abs")) == (tmp_path / "abs").resolve()
    assert runs.resolve_data_dir(str(tmp_path), "x/../y") == (tmp_path / "y").resolve()


# --- list and summary -------------------------------------------------------------------------------------

def test_list_runs_counts_complete_failed_running_newest_first(data, retried_run):
    second = make_fake_run(data, "r2", n_turns=1, conditions=DEFAULT_CONDITIONS[:2], scores=False)
    bare = data / "bare"
    bare.mkdir()
    JsonlWriter(bare / "status.jsonl").write({"run_id": "bare", "dyad_id": "x", "attempt": 1, "status": "complete"})
    (data / "empty-dir").mkdir()                          # neither manifest nor status: not a run
    _set_mtime(retried_run, 2_000_000_000)
    _set_mtime(second, 1_000_000_000)
    _set_mtime(bare, 1_500_000_000)
    listed = runs.list_runs(data)
    assert [r["run_id"] for r in listed] == ["r1", "bare", "r2"]
    r1 = listed[0]
    assert set(r1) == {"run_id", "started_at", "mtime", "has_manifest", "has_scores", "has_flags", "n_dyads",
                       "planned", "status_counts", "seeker", "mentor"}
    assert r1["status_counts"] == {"complete": 5, "failed": 1, "running": 1}
    assert r1["n_dyads"] == 7 and r1["planned"] == 6
    assert r1["has_manifest"] and r1["has_scores"] and not r1["has_flags"]
    assert r1["seeker"] == {"alias": "fake-seeker", "model_path": "/fake/seeker.gguf",
                            "model_sha256": r1["seeker"]["model_sha256"]}
    assert r1["mentor"]["alias"] == "fake-mentor" and r1["started_at"]
    assert r1["mtime"] == 2_000_000_000
    b = listed[1]
    assert not b["has_manifest"] and b["seeker"] is None and b["mentor"] is None and b["planned"] is None
    assert b["status_counts"] == {"complete": 1, "failed": 0, "running": 0} and b["started_at"] is None
    assert listed[2]["status_counts"] == {"complete": 2, "failed": 0, "running": 0} and not listed[2]["has_scores"]
    json.dumps(listed)
    assert runs.list_runs(data / "missing") == []


def test_run_summary_shape_and_per_dyad_progress(data, retried_run):
    s = runs.run_summary(data, "r1")
    assert set(s) == {"run_id", "path", "planned", "status_counts", "has_scores", "has_flags", "manifest", "roles",
                      "judges", "dyads", "factors", "metrics", "judge_shas", "offsets"}
    assert s["run_id"] == "r1" and s["path"] == str(retried_run.resolve()) and s["planned"] == 6
    assert s["status_counts"] == {"complete": 5, "failed": 1, "running": 1}
    assert set(s["manifest"]) == {"started_at", "harness_commit", "harness_dirty", "input_manifest", "batteries",
                                  "environment", "config"}
    assert s["manifest"]["harness_commit"] == "fake"
    for role in ("seeker", "mentor"):
        assert set(s["roles"][role]) == ROLE_INFO              # template_source and the rest are stripped
        assert s["roles"][role]["alias"] == f"fake-{role}"
    assert s["roles"]["seeker"]["model_sha256"] != s["roles"]["mentor"]["model_sha256"]
    [judge] = s["judges"]
    assert set(judge) == {"file", "alias", "model_path", "model_sha256", "scope", "subsample", "ts"}
    assert judge["file"] == f"judge-{judge['model_sha256'][:12]}.json" and judge["scope"] == "pilot"
    assert judge["alias"] == "fake-judge" and judge["subsample"] is None
    assert s["judge_shas"] == [judge["model_sha256"]]
    assert s["metrics"] == ["prompt_to_line", "line_to_line", "alignment"]
    dyads = {d["dyad_id"]: d for d in s["dyads"]}
    assert [d["dyad_id"] for d in s["dyads"]] == ["d01", "d02", "d03", "d04", "d05", "d06", "d07"]
    for d in s["dyads"]:
        assert set(d) == {"dyad_id", "attempt", "attempts", "condition", "persona_mode", "n_turns", "status",
                          "turns_done", "reason", "last_ts"}
    assert dyads["d01"] | {"last_ts": None} == {
        "dyad_id": "d01", "attempt": 1, "attempts": 1, "condition": DEFAULT_CONDITIONS[0],
        "persona_mode": "reinforced", "n_turns": 3, "status": "complete", "turns_done": 3, "reason": None,
        "last_ts": None}
    assert dyads["d02"]["attempt"] == 2 and dyads["d02"]["attempts"] == 2 and dyads["d02"]["status"] == "complete"
    assert dyads["d02"]["turns_done"] == 3 and dyads["d02"]["reason"] is None
    assert dyads["d05"]["status"] == "failed" and dyads["d05"]["attempt"] == 2 and dyads["d05"]["attempts"] == 2
    assert "fake failure" in dyads["d05"]["reason"] and dyads["d05"]["turns_done"] == 1   # failed at turn 2
    assert dyads["d07"]["status"] == "running" and dyads["d07"]["turns_done"] == 0
    assert dyads["d07"]["last_ts"] == "2026-10-07T10:00:01+0000"
    assert dyads["d05"]["last_ts"] > dyads["d02"]["last_ts"]
    assert s["factors"] == {
        "topic": ["immigration_enforcement"],
        "ideology": ["strong_left", "moderate", "strong_right", "none"],
        "openness": ["open", "closed", "(none)"],
        "role": ["union_organizer_placeholder", "small_business_owner_placeholder", "retired_officer_placeholder",
                 "(none)"]}
    json.dumps(s)


def test_run_summary_without_a_manifest(data):
    root = data / "loose"
    root.mkdir(parents=True)
    JsonlWriter(root / "status.jsonl").write({"run_id": "loose", "dyad_id": "a", "attempt": 1, "status": "started"})
    s = runs.run_summary(data, "loose")
    assert s["manifest"] is None and s["roles"] == {"seeker": None, "mentor": None} and s["planned"] is None
    assert s["judges"] == [] and s["metrics"] == [] and s["factors"] == {}
    assert s["dyads"][0]["status"] == "running" and s["dyads"][0]["condition"] is None


def test_planned_reads_a_relative_input_manifest_against_root(tmp_path, data, retried_run):
    manifest = json.loads((retried_run / "manifest.json").read_text())
    manifest["input_manifest"]["path"] = "elsewhere/m.jsonl"
    (retried_run / "manifest.json").write_text(json.dumps(manifest))
    assert runs.run_summary(data, "r1")["planned"] is None              # not under the repo root
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "m.jsonl").write_text('{"dyad_id": "a"}\n\n{"dyad_id": "b"}\n')
    assert runs.run_summary(data, "r1", root=tmp_path)["planned"] == 2
    assert runs.list_runs(data, root=tmp_path)[0]["planned"] == 2


# --- one dyad ---------------------------------------------------------------------------------------------

def test_dyad_detail_uses_the_latest_complete_attempt_in_message_order(data, retried_run):
    # Scramble turns.jsonl: file order is thread timing, and the reader must sort by message_order.
    turns_path = retried_run / "turns.jsonl"
    lines = turns_path.read_text().splitlines(keepends=True)
    turns_path.write_text("".join(reversed(lines)))
    R.cmd_flags({"data_dir": str(data)}, "r1", threshold=0.5, metric="prompt_to_line", run_length=1, judge=None)
    d = runs.dyad_detail(data, "r1", "d02")
    assert set(d) == {"run_id", "dyad_id", "attempt", "attempts", "dyad", "status", "turns", "surveys",
                      "survey_pairs", "scores", "flag"}
    assert d["attempt"] == 2 and d["attempts"] == [1, 2]
    assert d["dyad"]["attempt"] == 2 and d["dyad"]["condition"] == DEFAULT_CONDITIONS[1]
    assert [s["status"] for s in d["status"]] == ["started", "failed", "started", "complete"]   # the history
    assert len(d["turns"]) == 6 and all(t["attempt"] == 2 for t in d["turns"])
    assert d["turns"] == sorted(d["turns"], key=message_order)
    assert [(t["turn"], t["agent"]) for t in d["turns"][:2]] == [(1, "seeker"), (1, "mentor")]
    assert len(d["surveys"]["pre"]) == 13 and len(d["surveys"]["post"]) == 13
    assert all(r["attempt"] == 2 for r in d["surveys"]["pre"] + d["surveys"]["post"])
    assert d["scores"] and all(r["attempt"] == 2 for r in d["scores"])
    assert len(d["scores"]) == 3 * 3                     # pilot: 3 turns x (2 seeker metrics + alignment)
    assert d["flag"]["dyad_id"] == "d02" and d["flag"]["attempt"] == 2
    assert runs.run_summary(data, "r1")["has_flags"] is True

    first = runs.dyad_detail(data, "r1", "d02", attempt=1)
    assert first["attempt"] == 1 and first["turns"][-1]["finish_reason"] == "error"
    assert first["surveys"]["post"] == [] and first["scores"] == [] and first["flag"] is None
    assert all(p["post"] is None and p["delta"] is None for p in first["survey_pairs"])

    failed = runs.dyad_detail(data, "r1", "d05")       # no complete attempt: the highest one
    assert failed["attempt"] == 2 and failed["flag"] is None

    with pytest.raises(runs.RunNotFound):
        runs.dyad_detail(data, "r1", "d02", attempt=3)
    with pytest.raises(runs.RunNotFound):
        runs.dyad_detail(data, "r1", "d99")
    json.dumps(d)


def test_survey_pairs_are_post_minus_pre_of_the_run_pass(data, retried_run):
    d = runs.dyad_detail(data, "r1", "d01")
    pre = {r["item_id"]: r["answer"] for r in d["surveys"]["pre"]}
    post = {r["item_id"]: r["answer"] for r in d["surveys"]["post"]}
    assert [p["item_id"] for p in d["survey_pairs"]] == [r["item_id"] for r in d["surveys"]["pre"]]
    for p in d["survey_pairs"]:
        assert set(p) == {"item_id", "battery", "scale", "pre", "post", "delta"}
        assert (p["pre"], p["post"], p["delta"]) == (pre[p["item_id"]], post[p["item_id"]],
                                                     post[p["item_id"]] - pre[p["item_id"]])
    assert any(p["delta"] != 0 for p in d["survey_pairs"])
    # A later `harness survey` pass shares every key but origin: it is listed, and it does not move the pairs.
    readmin = dict(d["surveys"]["post"][0], origin="readministered", answer=d["surveys"]["post"][0]["answer"] + 99)
    JsonlWriter(retried_run / "surveys.jsonl").write(readmin)
    again = runs.dyad_detail(data, "r1", "d01")
    assert len(again["surveys"]["post"]) == 14 and again["survey_pairs"] == d["survey_pairs"]
    pairs = runs.survey_pairs(again["surveys"]["pre"] + again["surveys"]["post"], origin="readministered")
    assert [(p["pre"], p["post"], p["delta"]) for p in pairs] == [(None, readmin["answer"], None)]


# --- tail --------------------------------------------------------------------------------------------------

def test_read_from_returns_whole_lines_only(tmp_path):
    p = tmp_path / "x.jsonl"
    assert runs.read_from(p, 0) == ([], 0) and runs.read_from(p, 17) == ([], 17)
    p.write_bytes(b'{"a": 1}\n{"a": 2}\n{"a": 3')
    rows, off = runs.read_from(p, 0)
    assert rows == [{"a": 1}, {"a": 2}] and off == len(b'{"a": 1}\n{"a": 2}\n')
    assert runs.read_from(p, off) == ([], off)
    with open(p, "ab") as f:
        f.write(b'}\n\n{"a": "\xc3\xa9"}\n')
    rows, off2 = runs.read_from(p, off)
    assert rows == [{"a": 3}, {"a": "é"}] and off2 == p.stat().st_size
    with pytest.raises(ValueError):
        runs.read_from(p, -1)


def test_read_from_max_bytes_resumes_and_never_splits_a_line(tmp_path):
    p = tmp_path / "x.jsonl"
    lines = [json.dumps({"i": i, "pad": "x" * 50}) + "\n" for i in range(20)]
    p.write_text("".join(lines))
    got, off, calls = [], 0, 0
    while True:
        rows, new = runs.read_from(p, off, max_bytes=100)
        calls += 1
        if new == off:
            break
        got += rows
        off = new
    assert [r["i"] for r in got] == list(range(20)) and calls > 2
    rows, _ = runs.read_from(p, 0, max_bytes=10)          # a line longer than the cap still comes back whole
    assert rows == [json.loads(lines[0])]


def test_tail_returns_whole_lines_and_resumes_from_offsets(data, retried_run):
    t = runs.tail(data, "r1", {})
    assert set(t) == {"rows", "offsets"} and set(t["rows"]) == set(t["offsets"]) == {"turns", "status", "surveys",
                                                                                         "scores"}
    for name in ("turns", "status", "surveys", "scores"):
        assert t["rows"][name] == read_jsonl(retried_run / f"{name}.jsonl")
        assert t["offsets"][name] == (retried_run / f"{name}.jsonl").stat().st_size
    again = runs.tail(data, "r1", t["offsets"])
    assert again["rows"] == {"turns": [], "status": [], "surveys": [], "scores": []}
    assert again["offsets"] == t["offsets"]
    row = {"run_id": "r1", "dyad_id": "d07", "attempt": 1, "status": "complete", "ts": "2026-10-07T10:05:00+0000"}
    line = (json.dumps(row) + "\n").encode()
    with open(retried_run / "status.jsonl", "ab") as f:
        f.write(line[:20])                                  # a writer caught mid-line
    partial = runs.tail(data, "r1", t["offsets"])
    assert partial["rows"]["status"] == [] and partial["offsets"] == t["offsets"]
    with open(retried_run / "status.jsonl", "ab") as f:
        f.write(line[20:])
    done = runs.tail(data, "r1", partial["offsets"])
    assert done["rows"]["status"] == [row] and done["rows"]["turns"] == []
    assert done["offsets"]["status"] == t["offsets"]["status"] + len(line)
    # Offsets may arrive as query-string text; a file that does not exist yet stays at its offset.
    assert runs.tail(data, "r1", {k: str(v) for k, v in done["offsets"].items()})["offsets"] == done["offsets"]
    no_scores = make_fake_run(data, "r3", n_turns=1, conditions=DEFAULT_CONDITIONS[:1], scores=False)
    assert not (no_scores / "scores.jsonl").exists()
    assert runs.tail(data, "r3", {"scores": 0})["rows"]["scores"] == []


# --- level order --------------------------------------------------------------------------------------------

GRID_ORDER = {"topic": ["immigration_enforcement", "decarbonization"],
              "ideology": ["strong_left", "lean_left", "moderate", "lean_right", "strong_right", "none"],
              "openness": ["open", "closed"]}


def _edit_manifest(root: Path, **config):
    m = json.loads((root / "manifest.json").read_text())
    m["config"].update(config)
    (root / "manifest.json").write_text(json.dumps(m))


def test_level_order_follows_the_grid(data, retried_run):
    assert runs.level_order(data, "r1") == GRID_ORDER
    _edit_manifest(retried_run, grid="prompts/grid.json")                # relative: against the repo root
    assert runs.level_order(data, "r1", root=REPO) == GRID_ORDER
    assert runs.level_order(data, "r1") == GRID_ORDER                    # root defaults to the repo
    assert runs.level_order(data, "nope") == {}


def test_level_order_falls_back_to_a_study_json_beside_the_input_manifest(data, retried_run):
    _edit_manifest(retried_run, grid=None)
    assert runs.level_order(data, "r1") == {}
    study = {"factors": [{"key": "trust", "levels": [{"id": "high"}, {"id": "low"}]},
                         {"key": "topic", "levels": [{"id": "b_topic"}, {"id": "a_topic"}]}],
             "nested": {"key": "role", "within": "trust",
                        "variants": {"high": [{"id": "judge"}, {"id": "nurse"}], "low": [{"id": "broker"}]}},
             "control": {"factor": "trust", "level": "none"}}
    (data / "study.json").write_text(json.dumps(study))                 # beside r1-dyads.jsonl
    assert runs.level_order(data, "r1") == {"trust": ["high", "low", "none"], "topic": ["b_topic", "a_topic"],
                                            "role": ["judge", "nurse", "broker"]}
    (data / "study.json").write_text("{not json")
    assert runs.level_order(data, "r1") == {}
    _edit_manifest(retried_run, grid="no/such/grid.json")                # a broken grid falls through, never raises
    assert runs.level_order(data, "r1") == {}
    (retried_run / "manifest.json").write_text("[")
    assert runs.level_order(data, "r1") == {}


# --- reading a wave without re-parsing it ----------------------------------------------------------------------

def test_run_summary_progress_follows_appends_and_a_replaced_file(data, retried_run):
    """Progress is read incrementally (a wave's turns.jsonl is hundreds of MB): appended rows are counted on
    the next call, a half-written line is not, and a file replaced by a shorter one is re-read from 0."""
    def d07():
        return next(d for d in runs.run_summary(data, "r1")["dyads"] if d["dyad_id"] == "d07")
    assert d07()["turns_done"] == 0
    turns = retried_run / "turns.jsonl"
    row = {"run_id": "r1", "dyad_id": "d07", "attempt": 1, "turn": 1, "agent": "mentor", "finish_reason": "stop",
           "text": "x", "ts": "2026-10-07T10:05:00+0000"}
    JsonlWriter(turns).write(row)
    assert d07()["turns_done"] == 1 and d07()["last_ts"] == "2026-10-07T10:05:00+0000"
    with open(turns, "a", encoding="utf-8") as f:
        f.write(json.dumps({**row, "turn": 2})[:20])                   # the writer is mid-line
    assert d07()["turns_done"] == 1
    with open(turns, "a", encoding="utf-8") as f:
        f.write(json.dumps({**row, "turn": 2})[20:] + "\n")
    assert d07()["turns_done"] == 2
    turns.write_text(json.dumps(row) + "\n", encoding="utf-8")       # replaced, shorter
    assert d07()["turns_done"] == 1


def test_read_rows_contains_is_a_prefilter_only(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"dyad_id": "d1", "text": "mentions \\"d10\\""}\n{"dyad_id": "d10"}\n{"dyad_id": "d1"}\n')
    rows = runs.read_rows(p, contains=b'"d1"')
    assert [r["dyad_id"] for r in rows] == ["d1", "d1"]


def test_run_summary_offsets_end_on_a_whole_line_and_tail_from_them(data, retried_run):
    """The live view starts tailing at the summary's offsets: never from 0, never inside a row."""
    turns = retried_run / "turns.jsonl"
    with open(turns, "a", encoding="utf-8") as f:
        f.write('{"dyad_id": "d07", "half')                        # the harness is mid-write
    s = runs.run_summary(data, "r1")
    assert s["offsets"]["turns"] == turns.stat().st_size - len('{"dyad_id": "d07", "half')
    assert set(s["offsets"]) == {"turns", "status", "surveys", "scores"}
    assert runs.tail(data, "r1", s["offsets"])["rows"] == {"turns": [], "status": [], "surveys": [], "scores": []}
    with open(turns, "a", encoding="utf-8") as f:
        f.write('": 1}\n')
    assert runs.tail(data, "r1", s["offsets"])["rows"]["turns"] == [{"dyad_id": "d07", "half": 1}]
    assert runs.line_end_offset(retried_run / "missing.jsonl") == 0


def test_read_from_streams_a_whole_file_with_contains(tmp_path):
    """A dyad's transcript out of a wave's turns.jsonl (hundreds of MB) must not hold the file in memory: the
    lines are streamed and only those holding the dyad's id are parsed."""
    import tracemalloc
    p = tmp_path / "turns.jsonl"
    pad = "x" * 480
    with open(p, "w", encoding="utf-8") as f:
        for i in range(40_000):                                        # about 20 MB
            f.write(json.dumps({"dyad_id": "w-target" if i % 10_000 == 7 else f"w-{i:05d}", "pad": pad}) + "\n")
        f.write(json.dumps({"dyad_id": "w-target", "half": True})[:-1])  # the harness is mid-line
    size = p.stat().st_size
    tracemalloc.start()
    try:
        rows, offset = runs.read_from(p, 0, contains=b'"w-target"')
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert [r["dyad_id"] for r in rows] == ["w-target"] * 4 and not any("half" in r for r in rows)
    assert offset == size - len(json.dumps({"dyad_id": "w-target", "half": True})[:-1])
    assert peak < size // 10, peak                                     # never the whole file at once
    # The semantics are unchanged: whole lines only, offsets past the last one, a filter that matches nothing.
    assert runs.read_from(p, 0, contains=b'"nobody"') == ([], offset)
    assert runs.read_from(p, offset, contains=b'"w-target"') == ([], offset)
    with open(p, "a", encoding="utf-8") as f:
        f.write("}\n")
    assert runs.read_from(p, offset, contains=b'"w-target"') == ([{"dyad_id": "w-target", "half": True}],
                                                                  p.stat().st_size)
    assert runs.read_from(p, size * 2, contains=b'"w-target"')[1] == p.stat().st_size   # replaced: from 0


def test_run_summary_reads_scores_incrementally(data, retried_run, monkeypatch):
    """metrics and judge_shas come from scores.jsonl, which a scored wave makes hundreds of MB: parsed once,
    then only from the last offset; reset when the file shrinks or is replaced."""
    scores = retried_run / "scores.jsonl"
    first = runs.run_summary(data, "r1")
    judge = first["judge_shas"][0]
    assert first["metrics"] and first["metrics"] == runs.metrics_present(read_jsonl(scores))

    reads = []
    real = runs.read_from
    monkeypatch.setattr(runs, "read_from", lambda path, offset, *a, **k: reads.append((Path(path).name, offset))
                        or real(path, offset, *a, **k))
    size = scores.stat().st_size
    row = {"run_id": "r1", "dyad_id": "d01", "attempt": 1, "turn": 1, "agent": "mentor", "metric": "zz_custom",
           "judge_sha256": "b" * 64, "score": 0.5}
    JsonlWriter(scores).write(row)
    again = runs.run_summary(data, "r1")
    assert again["metrics"] == first["metrics"] + ["zz_custom"] and again["judge_shas"] == [judge, "b" * 64]
    assert [o for name, o in reads if name == "scores.jsonl"] == [size]          # only the appended bytes
    reads.clear()
    assert runs.run_summary(data, "r1")["metrics"] == again["metrics"]
    assert [o for name, o in reads if name == "scores.jsonl"] == []               # nothing new: nothing read

    tmp = retried_run / "scores.new"                                              # replaced by a longer file
    tmp.write_text(json.dumps({**row, "metric": "alignment", "judge_sha256": "c" * 64}) + "\n" +
                   scores.read_text(encoding="utf-8"), encoding="utf-8")
    os.replace(tmp, scores)
    replaced = runs.run_summary(data, "r1")
    assert replaced["judge_shas"] == ["c" * 64, judge, "b" * 64]
    scores.write_text(json.dumps(row) + "\n", encoding="utf-8")                   # truncated in place, shorter
    assert runs.run_summary(data, "r1")["metrics"] == ["zz_custom"]
    assert runs.run_summary(data, "r1")["judge_shas"] == ["b" * 64]
    scores.unlink()
    assert runs.run_summary(data, "r1")["metrics"] == [] and runs.run_summary(data, "r1")["judge_shas"] == []
