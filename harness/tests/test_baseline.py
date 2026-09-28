"""The unconstrained survey check (survey --no-schema --sample), the temperature override, and the
no-dialogue baseline (gap audit F11, F12; docs/pap/pre-analysis-plan.md section 10)."""
import json
from pathlib import Path
from harness import log, run as R
from harness.client import ServerError
from harness.tests.fakes import FakeClient
from harness.tests.test_run import _fake_servers, manifest_rows, write_cfg

N_ITEMS = 15


class FreeTextMentor(FakeClient):
    """Answers {"answer": 2} under the schema and free text without it, so the two paths can differ.
    Dialogue turns (n_predict 300) and the cache probe (n_predict 1) reply as FakeClient does."""
    def complete(self, prompt, **kw):
        c = FakeClient.complete(self, prompt, **kw)
        if kw.get("n_predict") not in (1, 300):
            c.text = '{"answer": 2}' if kw.get("json_schema") else "I would say 2."
        return c


def _servers(tmp_path, monkeypatch, mentor_cls=FreeTextMentor):
    clients = _fake_servers(tmp_path, monkeypatch)
    factory = R.LlamaClient

    def with_mentor(url, timeout=None):
        base = factory(url, timeout)
        if url != "http://m":
            return base
        m = mentor_cls(['{"answer": 2}', "line"])
        m.props = base.props
        clients[url] = m
        return m
    monkeypatch.setattr(R, "LlamaClient", with_mentor)
    return clients


def _run(tmp_path, n=3):
    man = tmp_path / "dyads.jsonl"
    man.write_text("".join(json.dumps(r) + "\n" for r in manifest_rows(n)))
    assert R.main(["run", "--config", str(write_cfg(tmp_path)), "--manifest", str(man), "--run-id", "r1"]) == 0
    return log.run_paths(tmp_path / "data", "r1")


def test_survey_no_schema_on_a_sample_is_compared_with_the_constrained_answers(tmp_path, monkeypatch, capsys):
    clients = _servers(tmp_path, monkeypatch)
    paths = _run(tmp_path)
    survey = lambda *extra: R.main(["survey", "--config", str(write_cfg(tmp_path)), "--run-id", "r1",
                                    "--phase", "post", *extra])
    capsys.readouterr()
    assert survey("--no-schema", "--sample", "2") == 0
    out = capsys.readouterr().out
    free = [s for s in log.read_jsonl(paths.surveys) if s["schema"] is False]
    assert len(free) == 2 * N_ITEMS and len({s["dyad_id"] for s in free}) == 2
    assert {s["dyad_id"] for s in free} == set(R.sample_dyads(["d0", "d1", "d2"], 2, 5))
    assert all(s["origin"] == "readministered" and s["answer"] == 2 and s["answer_method"] == "labelled"
               for s in free)
    no_schema_calls = [c for c in clients["http://m"].calls if c.get("n_predict") == 32 and not c.get("json_schema")]
    assert len(no_schema_calls) == 2 * N_ITEMS
    assert "schema false" in out and f"{2 * N_ITEMS} of {2 * N_ITEMS} equal the run's constrained answer" in out
    # the run's own rows say they were constrained
    assert all(s["schema"] is True for s in log.read_jsonl(paths.surveys) if s["origin"] == "run")
    # idempotent, and a constrained pass is another measurement: it asks every dyad again
    assert survey("--no-schema", "--sample", "2") == 0
    assert "2 already re-administered" in capsys.readouterr().out
    assert survey() == 0
    rows = log.read_jsonl(paths.surveys)
    assert len([s for s in rows if s["origin"] == "readministered" and s["schema"]]) == 3 * N_ITEMS
    assert len([s for s in rows if s["schema"] is False]) == 2 * N_ITEMS


def test_survey_temperature_override_is_sent_and_recorded(tmp_path, monkeypatch, capsys):
    clients = _servers(tmp_path, monkeypatch)
    paths = _run(tmp_path, n=1)
    assert R.main(["survey", "--config", str(write_cfg(tmp_path)), "--run-id", "r1", "--phase", "pre",
                   "--temperature", "0.7", "--n-predict", "64"]) == 0
    sent = [c for c in clients["http://m"].calls if c.get("n_predict") == 64]      # survey's own client
    assert len(sent) == N_ITEMS and {c["temperature"] for c in sent} == {0.7}
    rows = [s for s in log.read_jsonl(paths.surveys) if s["origin"] == "readministered"]
    assert {(s["temperature"], s["n_predict"], s["schema"], s["top_p"]) for s in rows} == {(0.7, 64, True, 0.95)}
    assert "temperature 0.7, n_predict 64" in capsys.readouterr().out


def test_baseline_administers_the_pre_battery_k_times_with_derived_seeds(tmp_path, monkeypatch, capsys):
    clients = _servers(tmp_path, monkeypatch)
    cfg = write_cfg(tmp_path, concurrency=2)
    base = lambda *extra, c=cfg: R.main(["baseline", "--config", str(c), "--run-id", "b1", "--k", "3", *extra])
    assert base() == 0
    paths = log.run_paths(tmp_path / "data", "b1")
    rows = log.read_jsonl(paths.baseline)
    assert len(rows) == 3 * N_ITEMS and not paths.turns.exists() and not paths.surveys.exists()
    assert {r["administration"] for r in rows} == {1, 2, 3} and {r["origin"] for r in rows} == {"baseline"}
    assert {r["phase"] for r in rows} == {"pre"} and {r["temperature"] for r in rows} == {0.7}
    assert len({r["seed"] for r in rows}) == len(rows)
    r = rows[0]
    assert r["seed"] == log.derive_seed(5, r["administration"], f"baseline-{r['administration']:04d}", 1, 0,
                                        f"survey:pre:{r['item_id']}")
    # no dialogue, no system prompt: the pre battery's own prompt shape
    pre = [c for c in clients["http://m"].calls if c.get("json_schema")]
    assert pre and all(c["prompt"].count("<|im_start|>user") == 1 for c in pre)
    assert {c["id_slot"] for c in pre} <= {0, 1} and {c["temperature"] for c in pre} == {0.7}
    mf = json.loads(paths.manifest.read_text())
    assert mf["kind"] == "baseline" and mf["baseline"] == {"phase": "pre", "k": 3, "temperature": 0.7,
                                                           "top_p": 0.95, "n_predict": 32, "schema": True}
    assert mf["mentor"]["model_sha256"] == "HASH-m.gguf" and "seeker" not in mf
    assert mf["resume_compares"] == R.BASELINE_COMPARES
    # a second pass has nothing to do; other settings on the same run_id are refused
    assert base() == 0 and len(log.read_jsonl(paths.baseline)) == 3 * N_ITEMS
    capsys.readouterr()
    assert base("--temperature", "1.0") == 1
    assert "baseline.temperature is 1.0, manifest.json records 0.7" in capsys.readouterr().err
    # a dialogue run's run_id is not a baseline
    _run(tmp_path, n=1)
    assert R.main(["baseline", "--config", str(cfg), "--run-id", "r1", "--k", "1"]) == 1
    assert "is not a baseline run" in capsys.readouterr().err


def test_baseline_exits_2_on_failures_and_fills_them_in(tmp_path, monkeypatch, capsys):
    class Flaky(FreeTextMentor):
        """The fourth survey item this server is asked fails."""
        def complete(self, prompt, **kw):
            if kw.get("json_schema") and sum(1 for c in self.calls if c.get("json_schema")) == 3:
                self.calls.append({"prompt": prompt, **kw})
                raise ServerError("mentor down")
            return FreeTextMentor.complete(self, prompt, **kw)
    _servers(tmp_path, monkeypatch, mentor_cls=Flaky)
    cfg = write_cfg(tmp_path, concurrency=1)
    assert R.main(["baseline", "--config", str(cfg), "--run-id", "b1", "--k", "2"]) == 2
    paths = log.run_paths(tmp_path / "data", "b1")
    assert "administrations failed" in capsys.readouterr().out
    assert len([r for r in log.read_jsonl(paths.baseline) if r.get("error")]) == 1
    _servers(tmp_path, monkeypatch)
    assert R.main(["baseline", "--config", str(cfg), "--run-id", "b1", "--k", "2"]) == 0
    ok = [r for r in log.read_jsonl(paths.baseline) if not r.get("error")]
    assert len(ok) == 2 * N_ITEMS and len({(r["administration"], r["item_id"]) for r in ok}) == 2 * N_ITEMS


def test_baseline_under_the_study_lock_refuses_another_run_seed(tmp_path, monkeypatch, capsys):
    _servers(tmp_path, monkeypatch)
    lock = tmp_path / "study.json"
    man = tmp_path / "w1-dyads.jsonl"
    man.write_text(json.dumps(manifest_rows(1)[0]) + "\n")
    assert R.main(["study", "--config", str(write_cfg(tmp_path, study=str(lock))), "--manifest", str(man)]) == 0
    capsys.readouterr()
    assert R.main(["baseline", "--config", str(write_cfg(tmp_path, study=str(lock), run_seed=6)),
                   "--run-id", "b1", "--k", "1"]) == 1
    assert "run_seed is 6, study.json has 5" in capsys.readouterr().err
    assert R.main(["baseline", "--config", str(write_cfg(tmp_path, study=str(lock))), "--run-id", "b1",
                   "--k", "1"]) == 0
    mf = json.loads(log.run_paths(tmp_path / "data", "b1").manifest.read_text())
    assert mf["study"]["sha256"] == log.sha256_file(lock)
