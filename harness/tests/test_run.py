import json
import os
import sys
from pathlib import Path
import jinja2
import pytest
import harness
from harness import log, run as R
from harness.client import ServerError
from harness.dialogue import AgentHandle, GenSettings, DyadSpec
from harness.templates import ChatTemplate
from harness.transcript import SEEKER, MENTOR
from harness.tests.fakes import FakeClient
from harness.tests.conftest import CHATML, REJECTS_TRAILING_SYSTEM

REPO = Path(__file__).resolve().parents[2]


def write_cfg(tmp_path, **over):
    cfg = {"data_dir": str(tmp_path / "data"), "gguf_py_path": None, "run_seed": 5, "now": "2026-09-08",
           "batteries": str(REPO / "instruments" / "batteries.json"),
           "seeker": {"url": "http://s"}, "mentor": {"url": "http://m"}, "judge": {"url": "http://j"}}
    cfg.update(over)
    p = tmp_path / "config.json"; p.write_text(json.dumps(cfg)); return p


def test_load_config_merges_defaults_and_requires_urls(tmp_path):
    cfg = R.load_config(write_cfg(tmp_path))
    assert cfg["generation"]["temperature"] == 0.7 and cfg["generation"]["n_predict"] == 300 and cfg["concurrency"] is None
    bad = tmp_path / "bad.json"; bad.write_text(json.dumps({"seeker": {"url": "x"}}))
    with pytest.raises(ValueError):
        R.load_config(bad)


def test_model_sha256_cached(tmp_path):
    f = tmp_path / "m.gguf"; f.write_bytes(b"abc")
    cache = tmp_path / "hashes.json"
    h1 = R.model_sha256_cached(str(f), cache)
    assert h1 == log.sha256_text("abc") and json.loads(cache.read_text())
    f.write_bytes(b"abcd")
    assert R.model_sha256_cached(str(f), cache) == log.sha256_text("abcd")


def fake_factory(url):
    return FakeClient(['{"answer": 3}'] if url in ("http://m", "http://j") else ["line"])


def test_build_agent_uses_props_model_path_and_template(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "read_template_from_gguf", lambda path, gguf_py_path=None: ChatTemplate.from_source(CHATML))
    monkeypatch.setattr(R, "model_sha256_cached", lambda path, cache_file=None: "HASH-" + path)
    cfg = R.load_config(write_cfg(tmp_path))
    handle, entry = R.build_agent(SEEKER, cfg["seeker"], 3, cfg, client_factory=fake_factory)
    assert handle.slot == 3 and handle.model_sha256 == "HASH-/fake/model.gguf" and handle.name == SEEKER
    assert entry["model_path"] == "/fake/model.gguf" and entry["total_slots"] == 4 and entry["template_sha256"] == handle.template.sha256


def test_check_agent_reports_parity_and_trailing_system(tmp_path):
    ok_tpl = ChatTemplate.from_source(CHATML)
    h = AgentHandle(SEEKER, FakeClient(), ok_tpl, "h", 0)
    results = R.check_agent(h, {"now": "2026-09-08"})
    assert all(ok for _, ok, _ in results) and {n for n, _, _ in results} >= {"health", "template_parity", "trailing_system"}
    bad = AgentHandle(SEEKER, FakeClient(), ChatTemplate.from_source(REJECTS_TRAILING_SYSTEM), "h", 0)
    results = dict((n, ok) for n, ok, _ in R.check_agent(bad, {"now": "2026-09-08"}))
    assert results["trailing_system"] is False and results["template_parity"] is False
    m = AgentHandle(MENTOR, FakeClient(), ChatTemplate.from_source(REJECTS_TRAILING_SYSTEM), "h", 0)
    assert "trailing_system" not in dict((n, ok) for n, ok, _ in R.check_agent(m, {"now": "2026-09-08"}))


def manifest_rows(n=3):
    return [{"dyad_id": f"d{i}", "condition": {"topic": "t"}, "persona_text": "P", "persona_reminder": "R",
             "persona_mode": "reinforced", "seed": i, "n_turns": 2} for i in range(n)]


def test_plan_work_resumes():
    status = [{"dyad_id": "d0", "attempt": 1, "status": "complete"}, {"dyad_id": "d1", "attempt": 1, "status": "failed"}]
    work = R.plan_work(manifest_rows(3), status)
    assert [(s.dyad_id, a) for s, a in work] == [("d1", 2), ("d2", 1)]


def make_ctx(tmp_path, seeker_client=None, mentor_client=None):
    tpl = ChatTemplate.from_source(CHATML)
    sc = seeker_client or FakeClient(["seeker line"])
    mc = mentor_client or FakeClient(['{"answer": 4}', "mentor line"])
    paths = log.run_paths(tmp_path, "r1")
    logs = {k: log.JsonlWriter(getattr(paths, k)) for k in ("dyads", "status", "turns", "surveys")}
    items = R.load_batteries(REPO / "instruments" / "batteries.json")[:2]
    ctx = R.RunContext("r1", 5, AgentHandle(SEEKER, sc, tpl, "S", 0), AgentHandle(MENTOR, mc, tpl, "M", 0),
                       GenSettings(), items, logs, clock=lambda: "T")
    return ctx, paths, sc, mc


def test_run_dyad_full_lifecycle(tmp_path):
    ctx, paths, sc, mc = make_ctx(tmp_path)
    spec = DyadSpec.from_row(manifest_rows(1)[0])
    assert R.run_dyad(2, spec, 1, ctx) == "complete"
    st = log.read_jsonl(paths.status)
    assert [s["status"] for s in st] == ["started", "complete"] and st[0]["attempt"] == 1
    d = log.read_jsonl(paths.dyads)[0]
    assert d["dyad_id"] == "d0" and d["attempt"] == 1 and d["persona_text"] == "P"
    turns = log.read_jsonl(paths.turns); surveys = log.read_jsonl(paths.surveys)
    assert len(turns) == 4 and len(surveys) == 4
    assert [s["phase"] for s in surveys] == ["pre", "pre", "post", "post"]
    assert all(c["id_slot"] == 2 for c in sc.calls + mc.calls)


def test_run_dyad_failure_marks_status(tmp_path):
    ctx, paths, sc, mc = make_ctx(tmp_path, seeker_client=FakeClient(["x"], fail_on=1))
    spec = DyadSpec.from_row(manifest_rows(1)[0])
    assert R.run_dyad(0, spec, 3, ctx) == "failed"
    st = log.read_jsonl(paths.status)
    assert st[-1]["status"] == "failed" and st[-1]["attempt"] == 3 and "fake failure" in st[-1]["reason"]


def test_main_run_and_score_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "read_template_from_gguf", lambda path, gguf_py_path=None: ChatTemplate.from_source(CHATML))
    monkeypatch.setattr(R, "model_sha256_cached", lambda path, cache_file=None: "HASH-" + path.split("/")[-1])
    clients = {}
    def factory(url, timeout=None):
        clients[url] = FakeClient(['{"answer": 2}', "line"] if url != "http://j" else ['{"score": 0.5, "rationale": "r"}'])
        clients[url].props = lambda: {"model_path": f"/{url[7:]}.gguf", "total_slots": 2, "build_info": "b", "model_alias": url[7:], "default_generation_settings": {}}
        clients[url].timeout = timeout
        return clients[url]
    monkeypatch.setattr(R, "LlamaClient", factory)
    cfg = write_cfg(tmp_path, concurrency=2)
    man = tmp_path / "dyads.jsonl"
    man.write_text("".join(json.dumps(r) + "\n" for r in manifest_rows(3)))
    assert R.main(["check", "--config", str(cfg)]) == 0
    assert R.main(["run", "--config", str(cfg), "--manifest", str(man), "--run-id", "r1"]) == 0
    assert clients["http://s"].timeout == 600
    paths = log.run_paths(tmp_path / "data", "r1")
    mf = json.loads(paths.manifest.read_text())
    assert mf["seeker"]["model_sha256"] == "HASH-s.gguf"
    assert mf["harness_commit"] and isinstance(mf["harness_dirty"], bool)
    assert mf["input_manifest"]["path"] == str(man) and mf["input_manifest"]["sha256"] == log.sha256_file(man)
    batteries = mf["batteries"]
    assert batteries["n_items"] == 13 and len(batteries["item_ids"]) == 13
    assert batteries["sha256"] == log.sha256_file(REPO / "instruments" / "batteries.json")
    assert all(s["batteries_sha256"] == batteries["sha256"] for s in log.read_jsonl(paths.surveys))
    env = mf["environment"]
    assert env["python"] == sys.version and env["jinja2"] == jinja2.__version__
    assert env["platform"] and env["harness_version"] == harness.__version__
    assert set(env) == {"python", "platform", "jinja2", "harness_version", "gguf_py_path", "gguf_py_commit", "gpu"}
    assert env["gpu"] is None or isinstance(env["gpu"], str)
    for role in ("seeker", "mentor"):
        assert mf[role]["template_source"] == CHATML          # archived in full, not only hashed
        assert "server_chat_template" in mf[role] and "model_ftype" in mf[role]
        assert mf[role]["build_info"] == "b" and mf[role]["total_slots"] == 2
    assert len(log.read_jsonl(paths.turns)) == 3 * 4
    assert sorted(s["status"] for s in log.read_jsonl(paths.status)).count("complete") == 3
    assert R.main(["run", "--config", str(cfg), "--manifest", str(man), "--run-id", "r1"]) == 0   # resume: nothing to do
    assert len(log.read_jsonl(paths.turns)) == 3 * 4
    assert R.main(["score", "--config", str(cfg), "--run-id", "r1", "--scope", "main"]) == 0
    scores = log.read_jsonl(paths.scores)
    assert len(scores) == 3 * 1 * 2 and all(s["score"] == 0.5 for s in scores)      # main: seeker, last turn (2), 2 metrics
    assert R.main(["survey", "--config", str(cfg), "--run-id", "r1", "--phase", "post"]) == 0
    post = [s for s in log.read_jsonl(paths.surveys) if s["phase"] == "post"]
    assert len(post) == 3 * 13 * 2
    # The re-administered rows share the in-run rows' key; `origin` is what tells them apart.
    assert sorted({s["origin"] for s in post}) == ["readministered", "run"]
    assert len([s for s in post if s["origin"] == "readministered"]) == 3 * 13
    judge_files = sorted(pth.name for pth in paths.root.glob("judge-*.json"))
    assert judge_files == ["judge-" + "HASH-j.gguf"[:12] + ".json"]
    judge = json.loads((paths.root / judge_files[0]).read_text())
    assert judge["scope"] == "main" and judge["n_predict"] == 160 and judge["temperature"] == 0.0
    assert judge["model_sha256"] == "HASH-j.gguf" and judge["template_source"] == CHATML
    assert judge["harness_commit"] and judge["judge_system"] and judge["judge_tasks"]
    assert all(s["harness_commit"] == judge["harness_commit"] for s in scores)


def test_main_check_reports_dead_server_without_traceback(tmp_path, monkeypatch, capsys):
    def factory(url, timeout=None):
        raise ServerError(f"connection refused: {url}")
    monkeypatch.setattr(R, "LlamaClient", factory)
    cfg = write_cfg(tmp_path)
    assert R.main(["check", "--config", str(cfg)]) == 1
    out = capsys.readouterr().out
    assert "FAIL health" in out


def test_main_run_manifest_mismatch_returns_1_without_raising(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "read_template_from_gguf", lambda path, gguf_py_path=None: ChatTemplate.from_source(CHATML))
    monkeypatch.setattr(R, "model_sha256_cached", lambda path, cache_file=None: "HASH-" + path.split("/")[-1])
    def factory(url, timeout=None):
        c = FakeClient(['{"answer": 2}', "line"])
        c.props = lambda: {"model_path": f"/{url[7:]}.gguf", "total_slots": 2, "build_info": "b",
                           "model_alias": url[7:], "default_generation_settings": {}}
        return c
    monkeypatch.setattr(R, "LlamaClient", factory)
    man = tmp_path / "dyads.jsonl"
    man.write_text("".join(json.dumps(r) + "\n" for r in manifest_rows(1)))
    cfg1 = write_cfg(tmp_path, run_seed=5)
    assert R.main(["run", "--config", str(cfg1), "--manifest", str(man), "--run-id", "r1"]) == 0
    cfg2 = write_cfg(tmp_path, run_seed=6)
    assert R.main(["run", "--config", str(cfg2), "--manifest", str(man), "--run-id", "r1"]) == 1


def test_model_sha256_cache_hit_is_keyed_by_path_size_and_mtime_ns(tmp_path):
    f = tmp_path / "m.gguf"; f.write_bytes(b"abc")
    cache = tmp_path / "hashes.json"
    assert R.model_sha256_cached(str(f), cache) == log.sha256_text("abc")
    key = next(iter(json.loads(cache.read_text())))
    assert key == f"{f}|3|{f.stat().st_mtime_ns}"
    # A cache hit does not re-hash: poison the entry and it is returned as it stands.
    cache.write_text(json.dumps({key: "POISONED"}))
    assert R.model_sha256_cached(str(f), cache) == "POISONED"
    # Same path, same size, different bytes: a changed mtime_ns is a cache miss even when the whole-second
    # mtime is unchanged (os.utime pins the nanoseconds, since some filesystems keep coarse timestamps).
    f.write_bytes(b"xyz")
    st = f.stat(); os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert R.model_sha256_cached(str(f), cache) == log.sha256_text("xyz")


def test_validate_manifest_rows_rejects_what_would_mislabel_or_break_a_wave():
    R.validate_manifest_rows(manifest_rows(2))                      # the good case stays good
    def rows(**over):
        r = manifest_rows(1)[0]; r.update(over); return [r]
    for bad in (rows(persona_mode="sometimes"), rows(n_turns=0),
                rows(persona_mode="reinforced", persona_reminder="  "), rows(persona_text=None)):
        if bad[0].get("persona_text") is None:
            bad[0].pop("persona_text")
        with pytest.raises(ValueError):
            R.validate_manifest_rows(bad)
    duplicate = manifest_rows(1) + manifest_rows(1)
    with pytest.raises(ValueError):
        R.validate_manifest_rows(duplicate)
    # plan_work validates before returning any work, so nothing is written for a bad manifest.
    with pytest.raises(ValueError):
        R.plan_work(rows(n_turns=0), [])


def test_main_run_reports_a_bad_manifest_as_an_error_line(tmp_path, monkeypatch, capsys):
    _fake_servers(tmp_path, monkeypatch)
    cfg = write_cfg(tmp_path)
    man = tmp_path / "dyads.jsonl"
    bad = manifest_rows(1)[0]; bad["persona_reminder"] = ""
    man.write_text(json.dumps(bad) + "\n")
    assert R.main(["run", "--config", str(cfg), "--manifest", str(man), "--run-id", "r1"]) == 1
    assert "persona_reminder" in capsys.readouterr().err


def test_run_dyad_contains_an_unexpected_exception(tmp_path):
    class Exploding(FakeClient):
        def complete(self, prompt, **kw):
            raise RuntimeError("something the harness never anticipated")
    ctx, paths, sc, mc = make_ctx(tmp_path, mentor_client=Exploding())
    spec = DyadSpec.from_row(manifest_rows(1)[0])
    assert R.run_dyad(0, spec, 1, ctx) == "failed"           # the pool survives; the dyad is marked
    st = log.read_jsonl(paths.status)
    assert st[-1]["status"] == "failed" and "RuntimeError" in st[-1]["reason"]


def _fake_servers(tmp_path, monkeypatch):
    """Point run.py at fake llama-servers: one FakeClient per url, no network, no GGUF."""
    monkeypatch.setattr(R, "read_template_from_gguf", lambda path, gguf_py_path=None: ChatTemplate.from_source(CHATML))
    monkeypatch.setattr(R, "model_sha256_cached", lambda path, cache_file=None: "HASH-" + path.split("/")[-1])
    clients = {}
    def factory(url, timeout=None):
        c = FakeClient(['{"answer": 2}', "line"])
        c.props = lambda: {"model_path": f"/{url[7:]}.gguf", "total_slots": 2, "build_info": "b",
                           "model_alias": url[7:], "default_generation_settings": {}}
        clients[url] = c
        return c
    monkeypatch.setattr(R, "LlamaClient", factory)
    return clients


def test_ctrl_c_stops_submitting_new_dyads_and_exits_130(tmp_path, monkeypatch, capsys):
    _fake_servers(tmp_path, monkeypatch)
    started = []
    def fake_run_dyad(slot, spec, attempt, ctx):
        started.append(spec.dyad_id)
        if len(started) == 2:
            raise KeyboardInterrupt          # stands in for the operator's Ctrl-C
        return "complete"
    monkeypatch.setattr(R, "run_dyad", fake_run_dyad)
    cfg = write_cfg(tmp_path, concurrency=1)
    man = tmp_path / "dyads.jsonl"
    man.write_text("".join(json.dumps(r) + "\n" for r in manifest_rows(3)))
    assert R.main(["run", "--config", str(cfg), "--manifest", str(man), "--run-id", "r1"]) == 130
    assert started == ["d0", "d1"]           # the third dyad is never started
    out = capsys.readouterr()
    assert "1 complete" in out.out and "not run" in out.out and "resume" in out.err


def test_run_exits_2_when_a_dyad_fails(tmp_path, monkeypatch):
    _fake_servers(tmp_path, monkeypatch)
    monkeypatch.setattr(R, "run_dyad", lambda slot, spec, attempt, ctx: "failed")
    cfg = write_cfg(tmp_path)
    man = tmp_path / "dyads.jsonl"
    man.write_text(json.dumps(manifest_rows(1)[0]) + "\n")
    assert R.main(["run", "--config", str(cfg), "--manifest", str(man), "--run-id", "r1"]) == 2


def test_run_refuses_when_seeker_and_mentor_share_a_server(tmp_path, monkeypatch, capsys):
    _fake_servers(tmp_path, monkeypatch)
    cfg = write_cfg(tmp_path, mentor={"url": "http://s"})
    man = tmp_path / "dyads.jsonl"
    man.write_text(json.dumps(manifest_rows(1)[0]) + "\n")
    assert R.main(["run", "--config", str(cfg), "--manifest", str(man), "--run-id", "r1"]) == 1
    assert "same" in capsys.readouterr().err.lower()


def test_resume_refuses_a_swapped_model_or_template(tmp_path, monkeypatch, capsys):
    _fake_servers(tmp_path, monkeypatch)
    cfg = write_cfg(tmp_path)
    man = tmp_path / "dyads.jsonl"
    man.write_text("".join(json.dumps(r) + "\n" for r in manifest_rows(2)))
    monkeypatch.setattr(R, "run_dyad", lambda slot, spec, attempt, ctx: "complete")
    assert R.main(["run", "--config", str(cfg), "--manifest", str(man), "--run-id", "r1"]) == 0
    # The GGUF at the same path is re-quantized: same config, different bytes.
    monkeypatch.setattr(R, "model_sha256_cached", lambda path, cache_file=None: "OTHER-" + path.split("/")[-1])
    assert R.main(["run", "--config", str(cfg), "--manifest", str(man), "--run-id", "r1"]) == 1
    assert "model_sha256" in capsys.readouterr().err


def test_survey_refuses_a_swapped_mentor(tmp_path, monkeypatch, capsys):
    _fake_servers(tmp_path, monkeypatch)
    cfg = write_cfg(tmp_path)
    man = tmp_path / "dyads.jsonl"
    man.write_text(json.dumps(manifest_rows(1)[0]) + "\n")
    assert R.main(["run", "--config", str(cfg), "--manifest", str(man), "--run-id", "r1"]) == 0
    monkeypatch.setattr(R, "read_template_from_gguf",
                        lambda path, gguf_py_path=None: ChatTemplate.from_source(CHATML + " "))
    assert R.main(["survey", "--config", str(cfg), "--run-id", "r1", "--phase", "pre"]) == 1
    assert "template_sha256" in capsys.readouterr().err
