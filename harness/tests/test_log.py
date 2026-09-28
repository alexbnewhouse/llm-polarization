"""Run paths, the JSONL writer, seed derivation, the write-once manifest and the resume index."""
import json, threading
from pathlib import Path
import pytest
from harness import log


def test_run_paths_creates_root(tmp_path):
    p = log.run_paths(tmp_path, "r1")
    assert p.root == tmp_path / "r1" and p.root.is_dir()
    assert p.turns.name == "turns.jsonl" and p.manifest.name == "manifest.json"
    assert p.dyads.name == "dyads.jsonl" and p.scores.name == "scores.jsonl"


def test_jsonl_writer_appends_and_reads_back(tmp_path):
    w = log.JsonlWriter(tmp_path / "t.jsonl")
    w.write({"a": 1}); w.write({"b": "x"})
    assert log.read_jsonl(tmp_path / "t.jsonl") == [{"a": 1}, {"b": "x"}]
    assert log.read_jsonl(tmp_path / "missing.jsonl") == []


def test_jsonl_writer_is_thread_safe(tmp_path):
    w = log.JsonlWriter(tmp_path / "t.jsonl")
    def work(i):
        for j in range(50):
            w.write({"i": i, "j": j})
    ths = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    [t.start() for t in ths]; [t.join() for t in ths]
    rows = log.read_jsonl(tmp_path / "t.jsonl")
    assert len(rows) == 400 and all(set(r) == {"i", "j"} for r in rows)


def test_derive_seed_is_stable_and_distinct():
    a = log.derive_seed(7, 42, "d1", 1, 3, "seeker")
    assert a == log.derive_seed(7, 42, "d1", 1, 3, "seeker")
    assert a != log.derive_seed(7, 42, "d1", 1, 3, "mentor")
    assert a != log.derive_seed(7, 42, "d1", 1, 4, "seeker")
    assert a != log.derive_seed(7, 42, "d1", 2, 3, "seeker")
    assert a != log.derive_seed(8, 42, "d1", 1, 3, "seeker")
    assert a != log.derive_seed(7, 43, "d1", 1, 3, "seeker")   # the per-dyad seed is live, not decorative
    assert 0 <= a < 2 ** 32


def test_sha256_helpers(tmp_path):
    assert log.sha256_text("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    f = tmp_path / "f.bin"; f.write_bytes(b"abc")
    assert log.sha256_file(f) == log.sha256_text("abc")


def test_write_manifest_refuses_changed_config(tmp_path):
    p = log.run_paths(tmp_path, "r1")
    cfg = {"run_seed": 1, "now": "2026-09-08", "concurrency": 8}
    log.write_manifest(p, {"run_id": "r1", "config": cfg})
    log.write_manifest(p, {"run_id": "r1", "config": cfg})        # same config: no-op
    with pytest.raises(log.ManifestMismatch):
        log.write_manifest(p, {"run_id": "r1", "config": {**cfg, "run_seed": 2}})
    assert json.loads(p.manifest.read_text())["config"] == cfg


def test_write_manifest_allows_operational_config_changes(tmp_path):
    # Dropping concurrency after an OOM and resuming the same run_id must be allowed: refusing it would
    # fragment one wave's data across two run ids. Only the run-affecting subset is compared.
    p = log.run_paths(tmp_path, "r1")
    cfg = {"run_seed": 1, "now": "2026-09-08", "concurrency": 8, "data_dir": "data"}
    log.write_manifest(p, {"run_id": "r1", "config": cfg})
    log.write_manifest(p, {"run_id": "r1", "config": {**cfg, "concurrency": 2, "data_dir": "/mnt/nas"}})
    assert json.loads(p.manifest.read_text())["config"]["concurrency"] == 8   # first write stands
    assert set(log.RUN_AFFECTING_CONFIG) == {"seeker", "mentor", "judge", "generation", "run_seed",
                                             "batteries", "now"}


def test_resume_index_and_next_attempt():
    rows = [
        {"dyad_id": "a", "attempt": 1, "status": "started"},
        {"dyad_id": "a", "attempt": 1, "status": "failed"},
        {"dyad_id": "a", "attempt": 2, "status": "started"},
        {"dyad_id": "b", "attempt": 1, "status": "started"},
        {"dyad_id": "b", "attempt": 1, "status": "complete"},
    ]
    idx = log.resume_index(rows)
    assert idx == {"a": {"attempt": 2, "status": "started"}, "b": {"attempt": 1, "status": "complete"}}
    assert log.next_attempt(idx, "a") == 3
    assert log.next_attempt(idx, "b") is None
    assert log.next_attempt(idx, "c") == 1


def test_resume_index_a_later_failed_row_does_not_undo_a_complete_attempt():
    # Parallelism H1: a second process's 'failed' row written after this attempt's 'complete' used to win
    # the tie and re-queue a completed dyad as attempt 2.
    rows = [{"dyad_id": "a", "attempt": 1, "status": "started"},
            {"dyad_id": "a", "attempt": 1, "status": "complete"},
            {"dyad_id": "a", "attempt": 1, "status": "failed"}]
    idx = log.resume_index(rows)
    assert idx == {"a": {"attempt": 1, "status": "complete"}} and log.next_attempt(idx, "a") is None
    rows.append({"dyad_id": "a", "attempt": 2, "status": "failed"})          # a higher attempt still wins
    assert log.resume_index(rows)["a"] == {"attempt": 2, "status": "failed"}


def test_run_lock_refuses_a_second_holder_and_is_released(tmp_path):
    p = log.run_paths(tmp_path, "r1")
    with log.run_lock(p, "run"):
        with pytest.raises(log.RunLocked, match="one harness process per run_id"):
            with log.run_lock(p, "score"):
                pass
        assert "(run," in (p.root / ".lock").read_text()
    with log.run_lock(p, "survey"):                                      # released on exit
        pass


def test_read_jsonl_names_the_file_and_line_and_diagnoses_a_torn_last_line(tmp_path):
    # Red-team M5, parallelism L1: the error named neither file nor line, and a torn tail blocked everything.
    f = tmp_path / "status.jsonl"
    f.write_text('{"a": 1}\n{"a": \n{"a": 3}\n')
    with pytest.raises(ValueError, match=r"status.jsonl line 2 is not valid JSON") as e:
        log.read_jsonl(f)
    assert not isinstance(e.value, log.TornLine)
    f.write_text('{"a": 1}\n\n{"a": 2, "b": "cut o')
    with pytest.raises(log.TornLine, match=r"status.jsonl line 3 .*--repair-torn-line"):
        log.read_jsonl(f)
    with pytest.raises(log.TornLine):
        log.check_tails(tmp_path)


def test_writer_refuses_to_glue_a_row_onto_a_torn_line_and_fsyncs_each_row(tmp_path, monkeypatch):
    synced = []
    monkeypatch.setattr(log.os, "fsync", lambda fd: synced.append(fd))
    f = tmp_path / "turns.jsonl"
    w = log.JsonlWriter(f)
    w.write({"a": 1}); w.write({"a": 2})
    assert len(synced) == 2
    f.write_text(f.read_text() + '{"a": 3, "cut')
    with pytest.raises(log.TornLine, match="line 3"):
        log.JsonlWriter(f).write({"a": 4})
    assert f.read_text().endswith('"cut')                     # nothing appended


def test_repair_drops_only_a_cut_off_last_line_after_a_backup(tmp_path):
    torn = tmp_path / "turns.jsonl"; torn.write_text('{"a": 1}\n{"a": 2}\n{"a": 3, "cu')
    whole = tmp_path / "status.jsonl"; whole.write_text('{"s": 1}\n{"s": 2}')
    fine = tmp_path / "surveys.jsonl"; fine.write_text('{"x": 1}\n')
    msgs = log.repair_torn_lines(tmp_path)
    assert len(msgs) == 2 and "dropped a cut-off last line of 12 bytes" in msgs[1]
    assert log.read_jsonl(torn) == [{"a": 1}, {"a": 2}] and log.read_jsonl(whole) == [{"s": 1}, {"s": 2}]
    assert whole.read_text().endswith("\n") and fine.read_text() == '{"x": 1}\n'
    backups = sorted(p.name.split(".torn-")[0] for p in tmp_path.glob("*.torn-*"))
    assert backups == ["status.jsonl", "turns.jsonl"]
    assert next(tmp_path.glob("turns.jsonl.torn-*")).read_text().endswith('"cu')
    log.check_tails(tmp_path)                                   # clean now
