"""The HTTP API (sandbox/server.py): every route of spec section 5 through App.handle (no sockets), the security
checks (X-Sandbox and Content-Type on writes, the Host header, path traversal, ids), and one real socket round
trip through make_server."""
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
import pytest
from sandbox import repo_study as R, study as S
from sandbox.server import App, make_server
from sandbox.tests.fakerun import dyad_ids, make_fake_run

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "studies" / "example-institutional-trust.study.json"
HOST = "127.0.0.1:8765"


@pytest.fixture
def app(tmp_path):
    studies = tmp_path / "studies"
    studies.mkdir()
    (studies / "trust.study.json").write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    a = App(ROOT, workspace=tmp_path / "ws", studies_dir=studies, mock_port=0)
    yield a
    a.close()


def call(app, method, path, body=None, query=None, headers=None, raw=None):
    """One request through App.handle with the headers the GUI sends; returns (status, parsed JSON or bytes)."""
    h = {"Host": HOST}
    if method != "GET":
        h.update({"X-Sandbox": "1", "Content-Type": "application/json"})
    h.update(headers or {})
    data = raw if raw is not None else (b"" if body is None else json.dumps(body).encode("utf-8"))
    status, out_headers, payload = app.handle(method, path, dict(query or {}), data, h)
    assert out_headers["Cache-Control"] == "no-store"
    if out_headers["Content-Type"].startswith("application/json"):
        assert out_headers["Content-Type"] == "application/json; charset=utf-8"
        return status, json.loads(payload)
    return status, payload


def example():
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


@pytest.fixture
def run_data(tmp_path):
    data = tmp_path / "data"
    make_fake_run(data, "r1")
    return data


# --- meta and studies -----------------------------------------------------------------------------------------

def test_meta(app):
    status, meta = call(app, "GET", "/api/meta")
    assert status == 200
    assert meta["root"] == str(ROOT) and meta["harness_version"] and meta["sandbox_version"]
    assert meta["data_dirs"][0] == "data"
    assert meta["randomization_presets"] == R.RANDOMIZATION_PRESETS
    assert meta["repo_sources"]["grid"] == "prompts/grid.json"
    assert isinstance(meta["gguf_importable"], bool)
    for key in ("python", "workspace", "studies_dir"):
        assert meta[key]


def test_studies_list_get_save_reload_and_delete(app, tmp_path):
    status, out = call(app, "GET", "/api/studies")
    assert status == 200
    assert [(s["name"], s["kind"]) for s in out["studies"]] == [("repo", "preset"), ("blank", "preset"),
                                                                ("trust", "saved")]
    assert out["studies"][2]["title"] == example()["name"] and out["studies"][2]["path"]

    status, repo = call(app, "GET", "/api/studies/repo")
    assert status == 200 and repo["kind"] == "preset" and repo["path"] is None
    assert not S.has_errors(S.validate_study(repo["spec"]))
    assert repo["spec"] == R.load_repo_study(ROOT)
    status, blank = call(app, "GET", "/api/studies/blank")
    assert status == 200 and blank["spec"] == S.blank_study()

    spec = example()
    spec["name"] = "Saved draft"
    spec["factors"][0]["levels"][0]["id"] = "bad id"            # drafts with errors may be saved
    status, saved = call(app, "PUT", "/api/studies/my-draft_2", {"spec": spec})
    assert status == 200 and S.has_errors(saved["issues"])
    assert (tmp_path / "studies" / "my-draft_2.study.json").is_file()
    status, back = call(app, "GET", "/api/studies/my-draft_2")
    assert status == 200 and back["spec"] == spec and back["kind"] == "saved"
    assert "my-draft_2" in [s["name"] for s in call(app, "GET", "/api/studies")[1]["studies"]]

    (tmp_path / "studies" / "broken.study.json").write_text("{not json", encoding="utf-8")
    titles = {s["name"]: s["title"] for s in call(app, "GET", "/api/studies")[1]["studies"]}
    assert titles["broken"] == "(unreadable)"

    assert call(app, "PUT", "/api/studies/repo", {"spec": spec})[0] == 409
    assert call(app, "PUT", "/api/studies/blank", {"spec": spec})[0] == 409
    assert call(app, "PUT", "/api/studies/Bad%20Name", {"spec": spec})[0] == 400
    assert call(app, "PUT", "/api/studies/-x", {"spec": spec})[0] == 400
    assert call(app, "PUT", "/api/studies/ok", {"spec": [1]})[0] == 400
    assert call(app, "PUT", "/api/studies/ok", {})[0] == 400
    assert call(app, "GET", "/api/studies/missing")[0] == 404

    assert call(app, "DELETE", "/api/studies/repo")[0] == 409
    assert call(app, "DELETE", "/api/studies/my-draft_2")[0] == 200
    assert call(app, "DELETE", "/api/studies/my-draft_2")[0] == 404
    assert not (tmp_path / "studies" / "my-draft_2.study.json").exists()


# --- the study endpoints --------------------------------------------------------------------------------------

def test_validate_reports_issues_summary_slots_and_shape_and_never_fails_on_garbage(app):
    status, out = call(app, "POST", "/api/study/validate", {"spec": R.load_repo_study(ROOT)})
    assert status == 200 and not S.has_errors(out["issues"])
    assert out["repo_shaped"] is True and out["repo_shape_reason"] is None
    assert out["summary"]["cells_treated"] == 20 and out["slots"]["treated"]
    status, out = call(app, "POST", "/api/study/validate", {"spec": example()})
    assert status == 200 and out["repo_shaped"] is False and out["repo_shape_reason"]
    for garbage in (None, [], "x", 3, {"factors": "nope", "nested": [1], "control": {"by": 5}}):
        status, out = call(app, "POST", "/api/study/validate", {"spec": garbage})
        assert status == 200 and S.has_errors(out["issues"]), garbage
        assert set(out) == {"issues", "summary", "slots", "repo_shaped", "repo_shape_reason"}
    assert call(app, "POST", "/api/study/validate", {})[0] == 200


def test_cells_render_manifest_and_repo_files(app):
    spec = example()
    status, out = call(app, "POST", "/api/study/cells", {"spec": spec})
    assert status == 200 and len(out["cells"]) == 10 and out["cells"][0]["selected"] is True

    cell = out["cells"][0]
    body = {"spec": spec, "kind": "treated", "condition": cell["condition"], "variant": cell["variants"][1]}
    status, out = call(app, "POST", "/api/study/render", body)
    assert status == 200 and "{" not in out["persona_text"] and out["slots"]["persona"] == cell["variants"][1]
    status, out = call(app, "POST", "/api/study/render",
                       {"spec": spec, "kind": "control", "condition": {"topic": "ai_regulation"}, "variant": None})
    assert status == 200 and out["persona_text"]
    status, out = call(app, "POST", "/api/study/render",
                       {"spec": spec, "kind": "treated", "condition": {"topic": "nope"}})
    assert status == 400 and out["error"] and out["issues"]

    status, out = call(app, "POST", "/api/study/manifest", {"spec": spec, "limit": 3})
    assert status == 200 and len(out["rows"]) == 3 and out["total"] == 40 and out["assignment"]["rows"] == 40
    assert out["rows"] == S.compile_manifest(spec)[0][:3]
    status, out = call(app, "POST", "/api/study/manifest", {"spec": spec})
    assert len(out["rows"]) == 20
    status, out = call(app, "POST", "/api/study/manifest", {"spec": spec, "limit": 10 ** 6})
    assert len(out["rows"]) == 40
    assert call(app, "POST", "/api/study/manifest", {"spec": spec, "limit": "x"})[0] == 400
    status, out = call(app, "POST", "/api/study/manifest", {"spec": {"name": "x"}})
    assert status == 400 and S.has_errors(out["issues"])

    repo = R.load_repo_study(ROOT)
    status, out = call(app, "POST", "/api/study/repo-files", {"spec": repo})
    assert status == 200 and (out["grid"], out["catalogue"]) == R.to_repo_files(repo)
    status, out = call(app, "POST", "/api/study/repo-files", {"spec": spec})
    assert status == 400 and "repo" in out["error"]


def test_export_writes_into_the_workspace_and_refuses_errors(app, tmp_path):
    status, out = call(app, "POST", "/api/study/export", {"spec": example()})
    assert status == 200 and out["engine"] == "sandbox.study" and out["rows"] == 40
    assert Path(out["dir"]).parent == tmp_path / "ws" / "exports"
    assert out["commands"]["check"][:3] == ["python", "-m", "harness.run"]
    status, out = call(app, "POST", "/api/study/export", {"spec": {"schema": S.SCHEMA}})
    assert status == 400 and S.has_errors(out["issues"])


# --- runs -----------------------------------------------------------------------------------------------------

def test_runs_list_summary_dyad_tail_analysis_and_config(app, run_data, tmp_path):
    q = {"data_dir": [str(run_data)]}                         # parse_qs shape: the first value counts
    status, out = call(app, "GET", "/api/runs", query=q)
    assert status == 200 and [r["run_id"] for r in out["runs"]] == ["r1"] and out["data_dir"] == str(run_data)

    status, out = call(app, "GET", "/api/runs/r1", query=q)
    assert status == 200 and out["run_id"] == "r1" and out["status_counts"]["complete"] == 6
    d = dyad_ids()[0]
    status, out = call(app, "GET", f"/api/runs/r1/dyads/{d}", query=q)
    assert status == 200 and out["dyad_id"] == d and out["turns"]
    assert call(app, "GET", f"/api/runs/r1/dyads/{d}", query={**q, "attempt": "1"})[0] == 200
    assert call(app, "GET", f"/api/runs/r1/dyads/{d}", query={**q, "attempt": "x"})[0] == 400
    assert call(app, "GET", f"/api/runs/r1/dyads/{d}", query={**q, "attempt": "9"})[0] == 404
    assert call(app, "GET", "/api/runs/r1/dyads/nobody", query=q)[0] == 404

    status, out = call(app, "GET", "/api/runs/r1/tail", query=q)
    assert status == 200 and out["rows"]["turns"] and all(v > 0 for v in out["offsets"].values())
    status, again = call(app, "GET", "/api/runs/r1/tail", query={**q, **{k: str(v) for k, v in out["offsets"].items()}})
    assert status == 200 and not any(again["rows"].values()) and again["offsets"] == out["offsets"]
    assert call(app, "GET", "/api/runs/r1/tail", query={**q, "turns": "-1"})[0] == 400

    status, out = call(app, "GET", "/api/runs/r1/analysis", query={**q, "factor": "ideology"})
    assert status == 200 and out["factor"] == "ideology" and out["survey"]["levels"]
    assert call(app, "GET", "/api/runs/r1/analysis", query={**q, "factor": "colour"})[0] == 400

    status, out = call(app, "POST", "/api/runs/r1/config", {"data_dir": str(run_data)})
    assert status == 200 and Path(out["path"]) == tmp_path / "ws" / "configs" / "r1.json"
    status, out = call(app, "POST", "/api/runs/r1/config",
                       {"data_dir": str(run_data), "judge": {"url": "http://127.0.0.1:8098", "gguf_path": ""}})
    assert status == 200 and Path(out["path"]).name.startswith("r1-judge-")
    assert call(app, "POST", "/api/runs/r1/config", {"data_dir": str(run_data), "judge": "x"})[0] == 400

    assert call(app, "GET", "/api/runs/nope", query=q)[0] == 404
    assert call(app, "GET", "/api/runs/..", query=q)[0] == 400
    assert call(app, "GET", "/api/runs/%2E%2E/tail", query=q)[0] == 400
    assert call(app, "POST", "/api/runs/nope/config", {"data_dir": str(run_data)})[0] == 404
    status, out = call(app, "GET", "/api/runs", query={"data_dir": str(tmp_path / "none")})
    assert status == 200 and out["runs"] == []


# --- jobs -----------------------------------------------------------------------------------------------------

def test_jobs_start_list_log_stop(app, tmp_path):
    status, job = call(app, "POST", "/api/jobs", {"kind": "check", "config": str(tmp_path / "no-such-config.json"),
                                                  "label": "bogus"})
    assert status == 201 and job["kind"] == "check" and job["label"] == "bogus" and job["id"]
    assert job["argv"][-2:] == ["--config", str(tmp_path / "no-such-config.json")]
    done = app.jobs.wait(job["id"], timeout=60)
    assert done["status"] == "failed" and done["returncode"] == 1

    status, out = call(app, "GET", "/api/jobs")
    assert status == 200 and [j["id"] for j in out["jobs"]] == [job["id"]]
    status, out = call(app, "GET", f"/api/jobs/{job['id']}")
    assert status == 200 and out["status"] == "failed"
    status, out = call(app, "GET", f"/api/jobs/{job['id']}/log", query={"offset": "0"})
    assert status == 200 and out["text"].startswith("$ ") and out["offset"] > 0 and out["job"]["id"] == job["id"]
    status, more = call(app, "GET", f"/api/jobs/{job['id']}/log", query={"offset": str(out["offset"])})
    assert status == 200 and more["text"] == "" and more["offset"] == out["offset"]
    assert call(app, "GET", f"/api/jobs/{job['id']}/log", query={"offset": "x"})[0] == 400
    status, out = call(app, "POST", f"/api/jobs/{job['id']}/stop", {})
    assert status == 200 and out["status"] == "failed"

    assert call(app, "GET", "/api/jobs/nope")[0] == 404
    assert call(app, "GET", "/api/jobs/nope/log")[0] == 404
    assert call(app, "POST", "/api/jobs/nope/stop", {})[0] == 404
    assert call(app, "POST", "/api/jobs", {"kind": "run", "config": "c.json", "manifest": "m.jsonl"})[0] == 400
    assert call(app, "POST", "/api/jobs", {"kind": "rm", "config": "c.json"})[0] == 400
    assert call(app, "POST", "/api/jobs", {"kind": "check", "config": "--evil"})[0] == 400
    assert call(app, "POST", "/api/jobs", {"config": "c.json"})[0] == 400


# --- the mock backend -----------------------------------------------------------------------------------------

def test_mock_start_info_stop(app):
    status, out = call(app, "GET", "/api/mock")
    assert status == 200 and out["running"] is False and out["seeker"] is None
    status, out = call(app, "POST", "/api/mock/start", {"slots": 2})
    assert status == 200 and out["running"] is True and out["slots"] == 2
    assert all(out[r]["url"].startswith("http://127.0.0.1:") and out[r]["gguf_path"] for r in ("seeker", "mentor",
                                                                                                "judge"))
    with urllib.request.urlopen(out["seeker"]["url"] + "/health", timeout=5) as resp:
        assert resp.status == 200
    status, again = call(app, "POST", "/api/mock/start", {"slots": 4})
    assert status == 200 and again == out                     # idempotent: the running backend
    assert call(app, "GET", "/api/mock")[1] == out
    status, out = call(app, "POST", "/api/mock/stop", {})
    assert status == 200 and out["running"] is False
    assert call(app, "POST", "/api/mock/start", {"slots": 0})[0] == 400
    assert call(app, "GET", "/api/mock")[1]["running"] is False


# --- security, static files and errors ------------------------------------------------------------------------

def test_writes_need_x_sandbox_and_json_and_every_request_a_local_host(app):
    body = {"spec": example()}
    assert call(app, "POST", "/api/study/validate", body, headers={"X-Sandbox": ""})[0] == 403
    assert call(app, "POST", "/api/study/validate", body, headers={"Content-Type": "text/plain"})[0] == 403
    assert call(app, "POST", "/api/study/validate", body,
                headers={"Content-Type": "application/json; charset=utf-8"})[0] == 200
    assert call(app, "PUT", "/api/studies/x", body, headers={"X-Sandbox": "0"})[0] == 403
    assert call(app, "DELETE", "/api/studies/trust", headers={"X-Sandbox": ""})[0] == 403
    for host in ("evil.example:8765", "evil.example", "127.0.0.1:9999", "127.0.0.1.evil.example:8765", ""):
        status, out = call(app, "GET", "/api/meta", headers={"Host": host})
        assert status == 403 and out["error"], host
    for host in ("127.0.0.1", "localhost:8765", "LOCALHOST", "[::1]:8765"):
        assert call(app, "GET", "/api/meta", headers={"Host": host})[0] == 200, host
    status, _, _ = app.handle("GET", "/api/meta", {}, b"", {})
    assert status == 403
    assert call(app, "POST", "/api/study/validate", raw=b"{not json")[0] == 400
    assert call(app, "POST", "/api/study/validate", raw=b"[1, 2]")[0] == 400
    assert call(app, "POST", "/api/study/validate", raw=b" " * (21 << 20))[0] == 413


def test_static_files_and_traversal(app, tmp_path):
    static = tmp_path / "static"
    (static / "js").mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><title>t</title>", encoding="utf-8")
    (static / "js" / "app.js").write_text("export {};", encoding="utf-8")
    (static / "app.css").write_text("body{}", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("no", encoding="utf-8")
    app.static_dir = static
    status, body = call(app, "GET", "/")
    assert status == 200 and body.startswith(b"<!doctype html>")
    headers = app.handle("GET", "/", {}, b"", {"Host": HOST})[1]
    assert headers["Content-Type"] == "text/html; charset=utf-8"
    assert app.handle("GET", "/static/js/app.js", {}, b"", {"Host": HOST})[1]["Content-Type"] == \
        "text/javascript; charset=utf-8"
    assert app.handle("GET", "/static/app.css", {}, b"", {"Host": HOST})[1]["Content-Type"] == "text/css; charset=utf-8"
    for path in ("/static/../secret.txt", "/static/%2e%2e/secret.txt", "/static/js/../../secret.txt",
                 "/static/..%2Fsecret.txt", "/static/", "/static/js", "/static/missing.js", "/static/a%00b"):
        assert call(app, "GET", path)[0] == 404, path
    app.static_dir = tmp_path / "nothing-here"
    assert call(app, "GET", "/")[0] == 404
    assert call(app, "GET", "/static/../server.py")[0] == 404


def test_unknown_routes_and_methods(app):
    assert call(app, "GET", "/api/nope")[0] == 404
    assert call(app, "GET", "/nope")[0] == 404
    status, out = call(app, "DELETE", "/api/meta")
    assert status == 405 and out["error"]
    assert call(app, "GET", "/api/study/validate")[0] == 405


# --- a real socket ----------------------------------------------------------------------------------------------

def test_a_real_socket_round_trip(tmp_path):
    server = make_server(ROOT, port=0, workspace=tmp_path / "ws", studies_dir=tmp_path / "studies")
    port = server.server_address[1]
    assert server.app.port == port and port != 0
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(base + "/api/meta", timeout=10) as resp:
            assert resp.status == 200 and resp.headers["Content-Type"] == "application/json; charset=utf-8"
            assert json.loads(resp.read())["root"] == str(ROOT)
        req = urllib.request.Request(base + "/api/study/validate", data=b'{"spec": null}', method="POST",
                                     headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=10)
        assert e.value.code == 403
        req = urllib.request.Request(base + "/api/study/validate", data=b'{"spec": null}', method="POST",
                                     headers={"Content-Type": "application/json", "X-Sandbox": "1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200 and json.loads(resp.read())["issues"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
