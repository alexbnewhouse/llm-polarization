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
    log.write_manifest(p, {"run_id": "r1", "config": {"x": 1}})
    log.write_manifest(p, {"run_id": "r1", "config": {"x": 1}})   # same config: no-op
    with pytest.raises(log.ManifestMismatch):
        log.write_manifest(p, {"run_id": "r1", "config": {"x": 2}})
    assert json.loads(p.manifest.read_text())["config"] == {"x": 1}


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
