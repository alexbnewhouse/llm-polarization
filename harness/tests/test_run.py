"""The CLI end to end against FakeClient servers: check, run, resume, survey, score, flags, agreement."""
import json
import os
import subprocess
import sys
from pathlib import Path
import jinja2
import pytest
import harness
from harness import log, run as R
from harness.client import ServerError
from harness.dialogue import AgentHandle, GenSettings, DyadSpec
from harness.templates import ChatTemplate, render as T_render
from harness.transcript import SEEKER, MENTOR
from harness.tests.fakes import FakeClient
from harness.tests.conftest import CHATML, REJECTS_TRAILING_SYSTEM

REPO = Path(__file__).resolve().parents[2]


def write_cfg(tmp_path, **over):
    cfg = {"data_dir": str(tmp_path / "data"), "gguf_py_path": None, "run_seed": 5, "now": "2026-09-08",
           "batteries": str(REPO / "instruments" / "batteries.json"), "grid": None,
           "seeker": {"url": "http://s"}, "mentor": {"url": "http://m", "family": "qwen"},
           "judge": {"url": "http://j", "family": "gemma"}}
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
    assert all(ok is not False for _, ok, _ in results)
    assert {n for n, _, _ in results} >= {"health", "template_parity", "template_parity_user_first",
                                          "server_chat_template", "trailing_system"}
    bad = AgentHandle(SEEKER, FakeClient(), ChatTemplate.from_source(REJECTS_TRAILING_SYSTEM), "h", 0)
    results = dict((n, ok) for n, ok, _ in R.check_agent(bad, {"now": "2026-09-08"}))
    assert results["trailing_system"] is False and results["template_parity"] is False
    m = AgentHandle(MENTOR, FakeClient(), ChatTemplate.from_source(REJECTS_TRAILING_SYSTEM), "h", 0)
    assert "trailing_system" not in dict((n, ok) for n, ok, _ in R.check_agent(m, {"now": "2026-09-08"}))


class DefaultSystemInjectingClient(FakeClient):
    """A server whose formatter inserts a default system block when the caller supplies none. This is the
    mentor's only message shape, and the system-first fixture cannot see the divergence."""
    def apply_template(self, messages):
        if not any(m["role"] == "system" for m in messages):
            messages = [{"role": "system", "content": "You are a helpful assistant."}] + list(messages)
        return T_render(self.tpl, messages)


def test_check_agent_catches_parity_that_only_breaks_without_a_system_message():
    h = AgentHandle(MENTOR, DefaultSystemInjectingClient(), ChatTemplate.from_source(CHATML), "h", 0)
    rows = dict((n, ok) for n, ok, _ in R.check_agent(h, {"now": "2026-09-08"}))
    assert rows["template_parity"] is True            # the system-first fixture sails through
    assert rows["template_parity_user_first"] is False


def test_check_agent_compares_the_template_the_server_reports():
    tpl = ChatTemplate.from_source(CHATML)
    same = FakeClient(); same.props = lambda: {**FakeClient().props(), "chat_template": CHATML}
    rows = dict((n, (ok, d)) for n, ok, d in R.check_agent(AgentHandle(MENTOR, same, tpl, "h", 0), {}))
    assert rows["server_chat_template"][0] is True
    # An arm served with --chat-template (Olmo) reports a different string: a warning, not a failure,
    # because template parity against that server is the gate.
    other = FakeClient(); other.props = lambda: {**FakeClient().props(), "chat_template": "{{ 'x' }}"}
    rows = dict((n, (ok, d)) for n, ok, d in R.check_agent(AgentHandle(MENTOR, other, tpl, "h", 0), {}))
    assert rows["server_chat_template"][0] is None and "differs" in rows["server_chat_template"][1]


def test_check_agent_context_budget():
    tpl = ChatTemplate.from_source(CHATML)
    c = FakeClient(); c.props = lambda: {**FakeClient().props(), "default_generation_settings": {"n_ctx": 8192}}
    cfg = {"now": "2026-09-08", "generation": {"n_predict": 300}}
    h = AgentHandle(MENTOR, c, tpl, "h", 0)
    rows = dict((n, ok) for n, ok, _ in R.check_agent(h, cfg, max_n_turns=10))     # 10*2*300+2048 = 8048
    assert rows["context_budget"] is True
    rows = dict((n, ok) for n, ok, _ in R.check_agent(h, cfg, max_n_turns=40))     # 40 turns does not fit
    assert rows["context_budget"] is False
    assert "context_budget" not in dict((n, ok) for n, ok, _ in R.check_agent(h, cfg))
    silent = FakeClient()                                                          # no n_ctx reported
    rows = dict((n, ok) for n, ok, _ in R.check_agent(AgentHandle(MENTOR, silent, tpl, "h", 0), cfg, max_n_turns=40))
    assert rows["context_budget"] is None


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
    assert batteries["n_items"] == 15 and len(batteries["item_ids"]) == 15
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
    assert len(post) == 3 * 15 * 2
    # The re-administered rows share the in-run rows' key; `origin` is what tells them apart.
    assert sorted({s["origin"] for s in post}) == ["readministered", "run"]
    assert len([s for s in post if s["origin"] == "readministered"]) == 3 * 15
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


def test_run_ignores_a_down_judge_but_check_still_reports_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(R, "read_template_from_gguf", lambda path, gguf_py_path=None: ChatTemplate.from_source(CHATML))
    monkeypatch.setattr(R, "model_sha256_cached", lambda path, cache_file=None: "HASH-" + path.split("/")[-1])
    def factory(url, timeout=None):
        c = FakeClient(['{"answer": 2}', "line"])
        if url == "http://j":
            def down():
                raise ServerError("judge is down")
            c.props = down
        else:
            c.props = lambda: {"model_path": f"/{url[7:]}.gguf", "total_slots": 2, "build_info": "b",
                               "model_alias": url[7:], "default_generation_settings": {}}
        return c
    monkeypatch.setattr(R, "LlamaClient", factory)
    cfg = write_cfg(tmp_path)
    man = tmp_path / "dyads.jsonl"
    man.write_text(json.dumps(manifest_rows(1)[0]) + "\n")
    # run only talks to seeker/mentor pre-flight, so a dead judge does not block it.
    assert R.main(["run", "--config", str(cfg), "--manifest", str(man), "--run-id", "r1"]) == 0
    # check on its own still covers the judge and reports it as down.
    assert R.main(["check", "--config", str(cfg)]) == 1
    assert "FAIL health judge" in capsys.readouterr().out


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


def _git_repo_with_one_tracked_harness_file(tmp_path):
    """A throwaway git repo, isolated from the operator's real git identity/signing config, with one
    commit tracking harness/x.py -- the fixture _git_dirty's path scoping is tested against."""
    repo = tmp_path / "repo"
    (repo / "harness").mkdir(parents=True)
    (repo / "harness" / "x.py").write_text("x = 1\n")
    run = lambda *args: subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    run("init", "-q")
    run("config", "user.email", "test@example.com")
    run("config", "user.name", "test")
    run("config", "commit.gpgsign", "false")
    run("add", "harness/x.py")
    run("commit", "-q", "-m", "init")
    return repo


def test_git_dirty_ignores_untracked_but_flags_tracked_source_changes(tmp_path, monkeypatch):
    repo = _git_repo_with_one_tracked_harness_file(tmp_path)
    monkeypatch.setattr(R, "HARNESS_DIR", repo)
    assert R._git_dirty() is False                              # clean tree
    (repo / "harness" / "manifest.json").write_text("{}")       # stands in for data/<run_id>/manifest.json
    assert R._git_dirty() is False                               # untracked: not flagged
    (repo / "harness" / "x.py").write_text("x = 2\n")            # a tracked source file, modified
    assert R._git_dirty() is True


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


def test_check_covers_the_judge_and_warns_when_two_roles_share_a_server(tmp_path, monkeypatch, capsys):
    _fake_servers(tmp_path, monkeypatch)
    assert R.main(["check", "--config", str(write_cfg(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert "[judge]" in out                        # the judge's prompt is checked before it scores anything
    assert R.main(["check", "--config", str(write_cfg(tmp_path, mentor={"url": "http://s"}))]) == 0
    assert "WARN seeker and mentor share" in capsys.readouterr().out


def test_check_reports_the_context_budget_for_the_manifest(tmp_path, monkeypatch, capsys):
    clients = _fake_servers(tmp_path, monkeypatch)
    man = tmp_path / "dyads.jsonl"
    rows = manifest_rows(1); rows[0]["n_turns"] = 40
    man.write_text(json.dumps(rows[0]) + "\n")
    cfg = write_cfg(tmp_path)
    assert R.main(["check", "--config", str(cfg), "--manifest", str(man)]) == 0
    assert "context_budget" in capsys.readouterr().out          # unknown n_ctx: reported, does not block
    def small_ctx(url, timeout=None):
        c = FakeClient()
        c.props = lambda: {"model_path": f"/{url[7:]}.gguf", "total_slots": 2, "build_info": "b",
                           "model_alias": url[7:], "default_generation_settings": {"n_ctx": 4096}}
        return c
    monkeypatch.setattr(R, "LlamaClient", small_ctx)
    assert R.main(["check", "--config", str(cfg), "--manifest", str(man)]) == 1
    assert "FAIL context_budget" in capsys.readouterr().out


def test_survey_refuses_when_the_instrument_changed(tmp_path, monkeypatch, capsys):
    _fake_servers(tmp_path, monkeypatch)
    cfg = write_cfg(tmp_path)
    man = tmp_path / "dyads.jsonl"
    man.write_text(json.dumps(manifest_rows(1)[0]) + "\n")
    assert R.main(["run", "--config", str(cfg), "--manifest", str(man), "--run-id", "r1"]) == 0
    original = json.loads((REPO / "instruments" / "batteries.json").read_text())
    original["items"][0]["text"] += " (edited wording)"
    modified = tmp_path / "batteries-modified.json"
    modified.write_text(json.dumps(original))
    cfg2 = write_cfg(tmp_path, batteries=str(modified))
    assert R.main(["survey", "--config", str(cfg2), "--run-id", "r1", "--phase", "post"]) == 1
    assert "error:" in capsys.readouterr().err
    # nothing was written for the refused re-administration
    assert not any(s["origin"] == "readministered" for s in log.read_jsonl(log.run_paths(tmp_path / "data", "r1").surveys))


def test_score_refuses_when_the_judge_fails_its_check(tmp_path, monkeypatch, capsys):
    _fake_servers(tmp_path, monkeypatch)
    cfg = write_cfg(tmp_path)
    man = tmp_path / "dyads.jsonl"
    man.write_text(json.dumps(manifest_rows(1)[0]) + "\n")
    monkeypatch.setattr(R, "run_dyad", lambda slot, spec, attempt, ctx: "complete")
    assert R.main(["run", "--config", str(cfg), "--manifest", str(man), "--run-id", "r1"]) == 0
    # The judge's GGUF template no longer matches what its server renders: grammar-forced scores would
    # still come back well-formed and be meaningless.
    monkeypatch.setattr(R, "read_template_from_gguf",
                        lambda path, gguf_py_path=None: ChatTemplate.from_source(
                            CHATML if "j" not in path else "{{ 'divergent' }}"))
    assert R.main(["score", "--config", str(cfg), "--run-id", "r1", "--scope", "pilot"]) == 1
    assert "not scoring" in capsys.readouterr().err


def test_check_agent_probes_kv_cache_reuse_on_the_slot():
    # Two completions on the same slot, the second extending the first: the server should prefill only the
    # new tokens. This is the pre-flight row that catches a server started without prompt caching, or a
    # template that rewrites the prefix between turns, before a wave spends a day finding out.
    tpl = ChatTemplate.from_source(CHATML)
    fc = FakeClient(["ok"])
    handle = AgentHandle(SEEKER, fc, tpl, "h", slot=3)
    cfg = R._merge(R.DEFAULT_CONFIG, {})
    results = dict((n, (ok, d)) for n, ok, d in R.check_agent(handle, cfg))
    ok, detail = results["cache_reuse"]
    assert ok is True, detail
    probes = fc.calls[-2:]
    assert all(c["id_slot"] == 3 and c["cache_prompt"] is True and c["n_predict"] == 1 for c in probes)
    assert probes[1]["prompt"].startswith(probes[0]["prompt"].rsplit("<|im_start|>assistant", 1)[0])


def test_check_agent_fails_cache_reuse_when_the_server_reprefills():
    tpl = ChatTemplate.from_source(CHATML)
    fc = FakeClient(["ok"])
    def cold(prompt, **kw):
        c = FakeClient.complete(fc, prompt, **kw)
        c.prompt_n = len(prompt.split())         # everything prefilled again: no cache hit at all
        return c
    fc.complete = cold
    handle = AgentHandle(SEEKER, fc, tpl, "h", slot=0)
    results = dict((n, (ok, d)) for n, ok, d in R.check_agent(handle, R._merge(R.DEFAULT_CONFIG, {})))
    ok, detail = results["cache_reuse"]
    assert ok is False and "prefilled" in detail


def test_cache_reuse_limit_comes_from_the_config_top_level(tmp_path):
    cfg = R.load_config(write_cfg(tmp_path))
    assert R._settings(cfg).cache_reuse_limit == 1000
    cfg = R.load_config(write_cfg(tmp_path, cache_reuse_limit=None))
    assert R._settings(cfg).cache_reuse_limit is None
    # operational, like concurrency: changing it must not make a resume refuse the run
    assert "cache_reuse_limit" not in log.RUN_AFFECTING_CONFIG


def _scored_run(tmp_path, monkeypatch):
    """A finished, scored two-dyad run with conditions and modes, for the flags/agreement commands."""
    _fake_servers(tmp_path, monkeypatch)
    cfg = write_cfg(tmp_path)
    rows = manifest_rows(2)
    rows[0]["condition"] = {"topic": "immigration_enforcement", "ideology": "strong_left", "openness": "open", "role": "x"}
    rows[1]["condition"] = {"topic": "immigration_enforcement", "ideology": "none", "openness": None, "role": None}
    rows[1]["persona_mode"] = "once"
    man = tmp_path / "dyads.jsonl"
    man.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert R.main(["run", "--config", str(cfg), "--manifest", str(man), "--run-id", "r1"]) == 0
    return cfg


def test_flags_command_writes_flags_and_reports_rates_by_ideology_and_mode(tmp_path, monkeypatch, capsys):
    cfg = _scored_run(tmp_path, monkeypatch)
    paths = log.run_paths(tmp_path / "data", "r1")
    w = log.JsonlWriter(paths.scores)
    for dyad, scores in (("d0", (0.2, 0.2)), ("d1", (0.9, 0.9))):
        for turn, s in zip((1, 2), scores):
            w.write({"run_id": "r1", "dyad_id": dyad, "attempt": 1, "turn": turn, "agent": SEEKER,
                     "metric": "prompt_to_line", "judge_sha256": "J", "score": s})
    # threshold is required: it is calibrated on pilot hand labels, never defaulted
    with pytest.raises(SystemExit):
        R.main(["flags", "--config", str(cfg), "--run-id", "r1"])
    assert R.main(["flags", "--config", str(cfg), "--run-id", "r1", "--threshold", "0.5", "--run-length", "2"]) == 0
    out = capsys.readouterr().out
    flags = log.read_jsonl(paths.flags)
    by = {r["dyad_id"]: r for r in flags}
    assert by["d0"]["flagged"] is True and by["d1"]["flagged"] is False
    assert by["d0"]["ideology"] == "strong_left" and by["d0"]["persona_mode"] == "reinforced"
    assert by["d0"]["threshold"] == 0.5 and by["d0"]["run_length"] == 2 and by["d0"]["metric"] == "prompt_to_line"
    assert by["d0"]["rule"] == "consecutive scored seeker turns"
    assert "strong_left" in out and "reinforced" in out and "1/1" in out
    # re-running replaces the file rather than appending to it: flags are derived, not a ledger
    assert R.main(["flags", "--config", str(cfg), "--run-id", "r1", "--threshold", "0.5", "--run-length", "2"]) == 0
    assert len(log.read_jsonl(paths.flags)) == 2


def test_agreement_command_reports_cross_judge_agreement(tmp_path, monkeypatch, capsys):
    cfg = _scored_run(tmp_path, monkeypatch)
    paths = log.run_paths(tmp_path / "data", "r1")
    w = log.JsonlWriter(paths.scores)
    for judge, scores in (("J1", (0.2, 0.8)), ("J2", (0.3, 0.7))):
        for turn, s in zip((1, 2), scores):
            w.write({"run_id": "r1", "dyad_id": "d0", "attempt": 1, "turn": turn, "agent": MENTOR,
                     "metric": "alignment", "judge_sha256": judge, "score": s})
    assert R.main(["agreement", "--config", str(cfg), "--run-id", "r1"]) == 0
    out = capsys.readouterr().out
    assert "J1" in out and "J2" in out and "n=2" in out


def test_score_stance_scope_with_a_subsample_and_the_judge_family_in_its_manifest(tmp_path, monkeypatch, capsys):
    cfg = _scored_run(tmp_path, monkeypatch)
    assert R.main(["score", "--config", str(cfg), "--run-id", "r1", "--scope", "stance", "--subsample", "1.0"]) == 0
    paths = log.run_paths(tmp_path / "data", "r1")
    scores = log.read_jsonl(paths.scores)
    assert scores and all(s["agent"] == MENTOR and s["metric"] == "alignment" for s in scores)
    judge = json.loads(next(paths.root.glob("judge-*.json")).read_text())
    assert judge["scope"] == "stance" and judge["subsample"] == 1.0 and "family" in judge
    manifest = json.loads(paths.manifest.read_text())
    assert "family" in manifest["mentor"] and "family" in manifest["seeker"]


def test_run_and_check_refuse_a_manifest_with_a_level_outside_the_grid(tmp_path, monkeypatch, capsys):
    # The grid gate (harness/grid.py): with config.grid set, `check --manifest` and `run` refuse a row whose
    # condition is not a cell of prompts/grid.json, before anything is written.
    _fake_servers(tmp_path, monkeypatch)
    cfg = write_cfg(tmp_path, grid=str(REPO / "prompts" / "grid.json"))
    bad = manifest_rows(1)[0]
    bad["condition"] = {"topic": "immigration_enforcement", "ideology": "centrist", "openness": "open", "role": "x"}
    man = tmp_path / "dyads.jsonl"; man.write_text(json.dumps(bad) + "\n")
    assert R.main(["check", "--config", str(cfg), "--manifest", str(man)]) == 1
    assert "centrist" in capsys.readouterr().out
    assert R.main(["run", "--config", str(cfg), "--manifest", str(man), "--run-id", "r1"]) == 1
    assert not (tmp_path / "data" / "r1" / "manifest.json").exists()
    good = manifest_rows(1)[0]
    good["condition"] = {"topic": "immigration_enforcement", "ideology": "moderate", "openness": "open", "role": "x"}
    man.write_text(json.dumps(good) + "\n")
    assert R.main(["run", "--config", str(cfg), "--manifest", str(man), "--run-id", "r2"]) == 0


def test_grid_defaults_to_the_repos_grid_and_null_disables_the_gate(tmp_path):
    cfg = R.load_config(write_cfg(tmp_path))
    assert cfg["grid"] is None                                  # write_cfg opts the unit tests out
    assert Path(R.DEFAULT_CONFIG["grid"]) == REPO / "prompts" / "grid.json"


def test_resume_compares_instrument_input_rows_build_and_commit(tmp_path, monkeypatch, capsys):
    # Red-team H2/M1, parallelism M4, gap audit F8: a resume that changes any of these would mix two
    # instruments, two stimulus sets, two builds or two versions of the code under one manifest.json.
    _fake_servers(tmp_path, monkeypatch)
    bat = tmp_path / "batteries.json"; bat.write_text((REPO / "instruments" / "batteries.json").read_text())
    cfg = write_cfg(tmp_path, batteries=str(bat))
    rows = manifest_rows(2)
    man = tmp_path / "dyads.jsonl"
    man.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(R, "run_dyad", lambda slot, spec, attempt, ctx: "failed")      # leaves work to resume
    run = lambda m=man, c=cfg: R.main(["run", "--config", str(c), "--manifest", str(m), "--run-id", "r1"])
    assert run() == 2
    paths = log.run_paths(tmp_path / "data", "r1")
    mf = json.loads(paths.manifest.read_text())
    assert mf["resume_compares"] == R.RESUME_COMPARES and "harness_diff_sha256" in mf
    assert log.sha256_file(paths.input_dyads) == mf["input_manifest"]["sha256"]
    assert run() == 2                                                   # nothing changed: resumes
    capsys.readouterr()

    def refused(expect, m=man, c=cfg):
        assert run(m, c) == 1
        err = capsys.readouterr().err
        assert expect in err and "not resuming" in err and err.count("\n") == 1, err
    # the instrument edited in place, at the same path
    original = bat.read_text()
    edited = json.loads(original); edited["items"][0]["text"] += " (reworded)"
    bat.write_text(json.dumps(edited))
    refused("now hashes to")
    bat.write_text(original)
    # a dyad row edited, and a dyad added; dropping one (a descope) is allowed
    other = tmp_path / "other.jsonl"
    changed = manifest_rows(2); changed[1]["persona_text"] = "EDITED"; changed[0]["n_turns"] = 3
    other.write_text("".join(json.dumps(r) + "\n" for r in changed))
    refused("d0 (n_turns), d1 (persona_text)", m=other)
    other.write_text("".join(json.dumps(r) + "\n" for r in manifest_rows(3)))
    refused("dyad_ids not in the input manifest this run started with: d2", m=other)
    other.write_text(json.dumps(rows[0]) + "\n")
    assert run(other) == 2
    # a server restarted on another llama.cpp build
    factory = R.LlamaClient
    def rebuilt(url, timeout=None):
        c = factory(url, timeout)
        c.props = (lambda p: lambda: {**p, "build_info": "b2"})(c.props())
        return c
    monkeypatch.setattr(R, "LlamaClient", rebuilt)
    refused("seeker.build_info is now 'b2', manifest.json records 'b'")
    monkeypatch.setattr(R, "LlamaClient", factory)
    # another harness commit; uncommitted changes that differ; a git state that cannot be read
    real_commit = R._git_commit()
    monkeypatch.setattr(R, "_git_commit", lambda: "0" * 40)
    refused("harness_commit is now 000000000000")
    monkeypatch.setattr(R, "_git_commit", lambda: real_commit)
    was_dirty = R._git_dirty()
    monkeypatch.setattr(R, "_git_dirty", lambda: True)
    monkeypatch.setattr(R, "_git_diff_sha256", lambda: "f" * 64)
    refused("uncommitted changes")
    monkeypatch.setattr(R, "_git_dirty", lambda: None)
    refused("cannot be confirmed unchanged")
    # an operational key is not compared
    monkeypatch.setattr(R, "_git_dirty", lambda: was_dirty)
    monkeypatch.setattr(R, "_git_diff_sha256", lambda: mf["harness_diff_sha256"])
    assert run(man, write_cfg(tmp_path, batteries=str(bat), concurrency=1)) == 2


def test_git_dirty_is_unknown_not_clean_when_git_cannot_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "HARNESS_DIR", tmp_path)               # not a git repository
    assert R._git_dirty() is None and R._git_diff_sha256() is None


def test_a_second_run_on_the_same_run_id_refuses(tmp_path, monkeypatch, capsys):
    # Red-team L7 / parallelism H1: two `run` processes on one run_id ran every dyad twice.
    _fake_servers(tmp_path, monkeypatch)
    cfg = write_cfg(tmp_path)
    man = tmp_path / "dyads.jsonl"
    man.write_text(json.dumps(manifest_rows(1)[0]) + "\n")
    paths = log.run_paths(tmp_path / "data", "r1")
    with log.run_lock(paths, "run"):
        for cmd in (["run", "--manifest", str(man)], ["survey"], ["score"]):
            assert R.main([cmd[0], "--config", str(cfg), "--run-id", "r1", *cmd[1:]]) == 1
            assert ".lock is held by pid" in capsys.readouterr().err
    assert not paths.status.exists()
    assert R.main(["run", "--config", str(cfg), "--manifest", str(man), "--run-id", "r1"]) == 0


def test_check_refuses_concurrency_above_the_slot_count_and_busy_slots(tmp_path, monkeypatch, capsys):
    # Parallelism M1 / red-team L3: llama.cpp wraps an out-of-range id_slot, so concurrency 3 on a 2-slot
    # server would put two dyads on one slot. Parallelism H1: a busy slot is another client's.
    clients = _fake_servers(tmp_path, monkeypatch)
    man = tmp_path / "dyads.jsonl"
    man.write_text(json.dumps(manifest_rows(1)[0]) + "\n")
    assert R.main(["check", "--config", str(write_cfg(tmp_path, concurrency=3))]) == 1
    assert "FAIL slots concurrency 3 exceeds the server's 2 slots" in capsys.readouterr().out
    assert R.main(["run", "--config", str(write_cfg(tmp_path, concurrency=3)), "--manifest", str(man),
                   "--run-id", "r1"]) == 1
    assert R.main(["check", "--config", str(write_cfg(tmp_path, concurrency=2))]) == 0
    capsys.readouterr()
    factory = R.LlamaClient
    def busy(url, timeout=None):
        c = factory(url, timeout)
        if url == "http://m":
            c.slots = lambda: [{"id": 0, "is_processing": False}, {"id": 1, "is_processing": True}]
        return c
    monkeypatch.setattr(R, "LlamaClient", busy)
    assert R.main(["check", "--config", str(write_cfg(tmp_path))]) == 1
    out = capsys.readouterr().out
    assert "FAIL slots slot(s) 1 of the 2 needed are busy" in out
    assert R.main(["check", "--config", str(write_cfg(tmp_path, concurrency=1))]) == 0   # slot 1 not needed
    # no slot count anywhere: concurrency null is refused, not silently 1
    def no_slots(url, timeout=None):
        c = factory(url, timeout)
        c.props = (lambda p: lambda: {**p, "total_slots": None})(c.props())
        c.slots = lambda: (_ for _ in ()).throw(ServerError("GET /slots: HTTP 501"))
        return c
    monkeypatch.setattr(R, "LlamaClient", no_slots)
    capsys.readouterr()
    assert R.main(["check", "--config", str(write_cfg(tmp_path))]) == 1
    assert "set `concurrency`" in capsys.readouterr().out
    assert R.main(["check", "--config", str(write_cfg(tmp_path, concurrency=2))]) == 0     # a warning only


def test_score_refuses_an_unknown_family_unless_the_config_states_it(tmp_path, monkeypatch, capsys):
    # Red-team H3: the fake servers' GGUFs (/m.gguf, /j.gguf) name no family, like an ollama blob.
    _fake_servers(tmp_path, monkeypatch)
    cfg = write_cfg(tmp_path, mentor={"url": "http://m"})
    man = tmp_path / "dyads.jsonl"
    man.write_text(json.dumps(manifest_rows(1)[0]) + "\n")
    assert R.main(["run", "--config", str(cfg), "--manifest", str(man), "--run-id", "r1"]) == 0
    mf = json.loads(log.run_paths(tmp_path / "data", "r1").manifest.read_text())
    assert mf["mentor"]["family"] is None and mf["mentor"]["family_source"] is None
    capsys.readouterr()
    assert R.main(["score", "--config", str(cfg), "--run-id", "r1", "--scope", "main"]) == 1
    assert "set mentor.family" in capsys.readouterr().err
    def score(judge):
        c = write_cfg(tmp_path, mentor={"url": "http://m", "family": "qwen"}, judge=judge)
        return R.main(["score", "--config", str(c), "--run-id", "r1", "--scope", "main"])
    assert score({"url": "http://j"}) == 1
    assert "set judge.family" in capsys.readouterr().err
    assert score({"url": "http://j", "family": "QwQ"}) == 1
    assert "mentor's model family (qwen)" in capsys.readouterr().err
    assert not list((tmp_path / "data" / "r1").glob("judge-*.json"))        # no record of a refused pass
    assert score({"url": "http://j", "family": "gemma"}) == 0
    judge = json.loads(next(log.run_paths(tmp_path / "data", "r1").root.glob("judge-*.json")).read_text())
    assert judge["family"] == "gemma" and judge["family_source"] == "config"


def test_with_slot_keeps_every_field_but_the_slot():
    h = AgentHandle("judge", FakeClient(), ChatTemplate.from_source(CHATML), "J", 0, alias="a",
                    family="gemma")
    c = R._with_slot(h, 5)
    assert c.slot == 5 and c.family == "gemma" and c.alias == "a" and h.slot == 0


def test_same_server_compares_resolved_endpoints_not_strings():
    # Red-team M4: a string compare let 127.0.0.1 vs localhost, and a trailing slash, through.
    same = R._same_server
    assert same("http://127.0.0.1:8201", "http://localhost:8201")
    assert same("http://127.0.0.1:8201", "http://127.0.0.1:8201/")
    assert same("http://[::1]:8201", "http://127.0.0.1:8201")
    assert not same("http://127.0.0.1:8201", "http://127.0.0.1:8202")
    assert not same("http://s", "http://m")


def test_run_refuses_two_urls_to_one_server(tmp_path, monkeypatch, capsys):
    _fake_servers(tmp_path, monkeypatch)
    man = tmp_path / "dyads.jsonl"
    man.write_text(json.dumps(manifest_rows(1)[0]) + "\n")
    run = lambda **over: R.main(["run", "--config", str(write_cfg(tmp_path, **over)), "--manifest", str(man),
                                 "--run-id", "r1"])
    assert run(seeker={"url": "http://127.0.0.1:1"},
               mentor={"url": "http://localhost:1/", "family": "qwen"}) == 1
    assert "are the same server" in capsys.readouterr().err
    # Two names the resolver cannot relate, served by one process: what the server reports gives it away.
    factory = R.LlamaClient
    one = {}
    def proxied(url, timeout=None):
        return one.setdefault("server", factory("http://one", timeout))
    monkeypatch.setattr(R, "LlamaClient", proxied)
    assert run() == 1
    assert "report the same model file" in capsys.readouterr().err
    assert not (tmp_path / "data" / "r1" / "manifest.json").exists()


def test_survey_checks_config_exits_2_on_failure_and_never_duplicates_rows(tmp_path, monkeypatch, capsys):
    # Red-team M7, gap audit F16/F17: survey took any run_seed/now/generation, exited 0 with failed items,
    # and a second pass duplicated every dyad's rows.
    _fake_servers(tmp_path, monkeypatch)
    cfg = write_cfg(tmp_path)
    man = tmp_path / "dyads.jsonl"
    man.write_text("".join(json.dumps(r) + "\n" for r in manifest_rows(2)))
    assert R.main(["run", "--config", str(cfg), "--manifest", str(man), "--run-id", "r1"]) == 0
    survey = lambda c=None: R.main(["survey", "--config", str(c or write_cfg(tmp_path)), "--run-id", "r1",
                                    "--phase", "post"])
    paths = log.run_paths(tmp_path / "data", "r1")
    readmin = lambda: [s for s in log.read_jsonl(paths.surveys) if s["origin"] == "readministered"]
    capsys.readouterr()
    assert survey(write_cfg(tmp_path, run_seed=6, now="2026-10-01")) == 1
    assert "config run_seed, now differ" in capsys.readouterr().err and not readmin()
    factory = R.LlamaClient
    def flaky(url, timeout=None):
        c = factory(url, timeout); c.fail_on = 2
        return c
    monkeypatch.setattr(R, "LlamaClient", flaky)
    assert survey() == 2                                         # one dyad's second item failed
    assert "1 failed" in capsys.readouterr().out
    ok = [s for s in readmin() if not s.get("error")]
    assert len(ok) == 1 + 13
    monkeypatch.setattr(R, "LlamaClient", factory)
    assert survey() == 0                                         # fills in the failed dyad's 12 items only
    ok = [s for s in readmin() if not s.get("error")]
    assert len(ok) == 2 * 13 and len({(s["dyad_id"], s["item_id"]) for s in ok}) == 2 * 13
    assert survey() == 0 and "2 already re-administered" in capsys.readouterr().out
    assert len([s for s in readmin() if not s.get("error")]) == 2 * 13


def test_score_exits_2_on_judge_failures_and_subsamples_on_the_manifests_run_seed(tmp_path, monkeypatch,
                                                                                    capsys):
    # Red-team M8 (a), (b): the second judge's config need not repeat run_seed, and a failed pass is exit 2.
    from harness.scorer import subsample_dyads
    _fake_servers(tmp_path, monkeypatch)
    cfg = write_cfg(tmp_path, run_seed=5)
    man = tmp_path / "dyads.jsonl"
    man.write_text("".join(json.dumps(r) + "\n" for r in manifest_rows(12)))
    assert R.main(["run", "--config", str(cfg), "--manifest", str(man), "--run-id", "r1"]) == 0
    judge2 = write_cfg(tmp_path, run_seed=0)
    assert R.main(["score", "--config", str(judge2), "--run-id", "r1", "--scope", "main",
                   "--subsample", "0.5"]) == 0
    paths = log.run_paths(tmp_path / "data", "r1")
    scored = {s["dyad_id"] for s in log.read_jsonl(paths.scores)}
    assert scored == subsample_dyads([f"d{i}" for i in range(12)], 0.5, 5) != subsample_dyads(
        [f"d{i}" for i in range(12)], 0.5, 0)
    s0 = log.read_jsonl(paths.scores)[0]
    assert s0["seed"] == log.derive_seed(5, int(s0["dyad_id"][1:]), s0["dyad_id"], 1, s0["turn"],
                                         f"judge:{s0['agent']}:{s0['metric']}")
    factory = R.LlamaClient
    def failing_judge(url, timeout=None):
        c = factory(url, timeout)
        if url == "http://j":
            real = c.complete
            c.complete = lambda prompt, **kw: (real(prompt, **kw) if kw.get("n_predict") == 1
                                               else (_ for _ in ()).throw(ServerError("judge down")))
        return c
    monkeypatch.setattr(R, "LlamaClient", failing_judge)
    capsys.readouterr()
    assert R.main(["score", "--config", str(cfg), "--run-id", "r1", "--scope", "main"]) == 2
    assert "judge calls failed" in capsys.readouterr().out


def test_judge_records_are_never_overwritten(tmp_path):
    # Red-team M8 (c), gap audit F14: a third differing pass used to overwrite judge-<sha12>-<scope>.json.
    paths = log.run_paths(tmp_path, "r1")
    entry = {"model_sha256": "abcdef1234567890", "url": "http://j"}
    names = [R.write_judge_manifest(paths, entry, scope, sub).name
             for scope, sub in (("pilot", None), ("stance", 0.2), ("stance", 0.5), ("stance", 0.2),
                                ("stance", 0.7))]
    assert names == ["judge-abcdef123456.json", "judge-abcdef123456-stance.json",
                     "judge-abcdef123456-stance-2.json", "judge-abcdef123456-stance.json",
                     "judge-abcdef123456-stance-3.json"]
    subs = [json.loads((paths.root / n).read_text())["subsample"] for n in sorted(set(names))]
    assert sorted(subs, key=str) == sorted([None, 0.2, 0.5, 0.7], key=str)


def test_setup_errors_exit_1_with_one_line_not_a_traceback(tmp_path, monkeypatch, capsys):
    # Gap audit F16: a wrong gguf_py_path or a missing GGUF (TemplateError), a missing file (OSError), a
    # manifest without a key (KeyError) escaped main as tracebacks.
    from harness.templates import TemplateError
    _fake_servers(tmp_path, monkeypatch)
    cfg = write_cfg(tmp_path)
    man = tmp_path / "dyads.jsonl"
    man.write_text(json.dumps(manifest_rows(1)[0]) + "\n")
    assert R.main(["run", "--config", str(cfg), "--manifest", str(man), "--run-id", "r1"]) == 0
    def no_gguf(path, gguf_py_path=None):
        raise TemplateError(f"GGUF not found: {path}")
    monkeypatch.setattr(R, "read_template_from_gguf", no_gguf)
    capsys.readouterr()
    assert R.main(["check", "--config", str(cfg)]) == 1
    assert "FAIL setup seeker" in capsys.readouterr().out
    assert R.main(["run", "--config", str(cfg), "--manifest", str(man), "--run-id", "r1"]) == 1
    assert "FAIL setup seeker" in capsys.readouterr().out
    for cmd in ("survey", "score"):
        assert R.main([cmd, "--config", str(cfg), "--run-id", "r1"]) == 1
        err = capsys.readouterr().err
        assert err.startswith("error: ") and err.count("\n") == 1 and "GGUF not found" in err
    _fake_servers(tmp_path, monkeypatch)
    missing = write_cfg(tmp_path, batteries=str(tmp_path / "nope.json"))
    assert R.main(["run", "--config", str(missing), "--manifest", str(man), "--run-id", "r2"]) == 1
    assert "nope.json" in capsys.readouterr().err
    paths = log.run_paths(tmp_path / "data", "r1")
    mf = json.loads(paths.manifest.read_text()); del mf["batteries"]
    paths.manifest.write_text(json.dumps(mf))
    assert R.main(["survey", "--config", str(write_cfg(tmp_path)), "--run-id", "r1"]) == 1
    assert "error: missing key 'batteries'" in capsys.readouterr().err
