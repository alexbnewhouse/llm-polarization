"""The sandbox's HTTP server: the JSON API of docs/superpowers/specs/2026-10-07-sandbox-gui-design.md section 5
over study, repo_study, export, runs, analysis, jobs and mock_server, and the static app in sandbox/static/.

`App.handle(method, path, query, body, headers)` is the whole server as one function, so every route and every
check is tested without a socket; `make_server` puts it behind a stdlib ThreadingHTTPServer. There is no
authentication: the server binds 127.0.0.1 and is reached from another machine through an SSH tunnel. What
stands in for it is what a browser enforces: every non-GET request must carry `X-Sandbox: 1` and a JSON content
type, which a page on another site cannot send without a CORS preflight this server never answers, and every
request's Host must be this server's own name (a DNS-rebinding page arrives under its own host name). Nothing a
client sends reaches a shell: jobs are built from structured fields by sandbox/jobs.py."""
from __future__ import annotations
import argparse
import importlib.util
import ipaddress
import json
import math
import os
import platform
import re
import signal
import socket
import sys
import threading
import traceback
import urllib.parse
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from harness import __version__ as HARNESS_VERSION
from sandbox import __version__ as SANDBOX_VERSION
from sandbox import analysis, export, repo_study, runs, study
from sandbox.jobs import JobConflict, JobManager
from sandbox.mock_server import MockBackend

REPO = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_BODY = 20 << 20                    # a wave's study spec is tens of kB; anything near this is not one
STUDY_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
STUDY_SUFFIX = ".study.json"
PRESETS = {"repo": repo_study.REPO_STUDY_NAME, "blank": "Blank study"}
MANIFEST_LIMIT, MANIFEST_LIMIT_MAX = 20, 500
MOCK_BASE_PORT = 18201                 # stable mock URLs across restarts, so a mock run can be resumed; 0 if taken
MOCK_MAX_SLOTS = 256
JSON_TYPE = "application/json; charset=utf-8"
MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
        ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".svg": "image/svg+xml; charset=utf-8", ".json": "application/json; charset=utf-8",
        ".txt": "text/plain; charset=utf-8", ".map": "application/json; charset=utf-8", ".png": "image/png",
        ".ico": "image/x-icon", ".woff2": "font/woff2"}
LOCAL_NAMES = ("localhost", "127.0.0.1", "[::1]")
_EMPTY_SUMMARY = {"factors": [], "nested": None, "cells_treated": 0, "cells_control": 0, "cells_selected": 0,
                  "rows": 0, "rows_design": 0, "rows_per_mode": {}, "n_turns": None, "messages": 0,
                  "condition_keys": []}
_NO_MOCK = {"running": False, "seeker": None, "mentor": None, "judge": None, "slots": None, "delay": None}


class HTTPError(Exception):
    """An answer other than 200 that a handler decides on: status, message, and extra JSON fields."""

    def __init__(self, status: int, message: str, headers: dict | None = None, **extra):
        super().__init__(message)
        self.status, self.headers, self.extra = status, headers or {}, extra


@dataclass
class _Request:
    method: str
    args: list                         # the segments a route's "*" / "**" matched, unquoted
    query: dict                        # one value per name: the first
    raw: bytes
    _body: dict | None = field(default=None, repr=False)

    def body(self) -> dict:
        """The JSON object the request carries ({} when it carries nothing)."""
        if self._body is None:
            if not self.raw.strip():
                self._body = {}
            else:
                try:
                    obj = json.loads(self.raw.decode("utf-8"))
                except (UnicodeDecodeError, ValueError) as e:
                    raise HTTPError(400, f"the request body is not JSON: {e}") from None
                if not isinstance(obj, dict):
                    raise HTTPError(400, f"the request body must be a JSON object, got {type(obj).__name__}")
                self._body = obj
        return self._body


@dataclass
class _File:
    content_type: str
    data: bytes


def _finite(obj):
    """NaN and infinities as null: JSON has neither, and the browser's JSON.parse rejects them."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def json_response(status: int, obj, headers: dict | None = None) -> tuple[int, dict, bytes]:
    """A JSON answer, never cached (every view is live data)."""
    try:
        text = json.dumps(obj, ensure_ascii=False, allow_nan=False, default=str)
    except ValueError:
        text = json.dumps(_finite(obj), ensure_ascii=False, default=str)
    return status, {"Content-Type": JSON_TYPE, "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                    **(headers or {})}, text.encode("utf-8")


def _int_param(value, name: str, minimum: int = 0) -> int | None:
    """An integer query or body value (None when absent or empty); ValueError naming it otherwise."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    try:
        n = int(value.strip()) if isinstance(value, str) else value
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {value!r}") from None
    if not isinstance(n, int) or n < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return n


class App:
    """The API and the static app, as `handle()`. Holds the job manager (workspace/jobs) and at most one mock
    backend (workspace/mock), which close() stops."""

    def __init__(self, root, *, workspace=None, studies_dir=None, host: str = "127.0.0.1", port: int = 8765,
                 mock_port: int = MOCK_BASE_PORT):
        """`root` is the repository: the repo study is read from it and every job runs in it. `workspace`
        (default root/workspace) holds exports, derived configs, job logs and mock GGUFs; `studies_dir` (default
        root/studies) the saved studies. `host` and `port` are what the server is bound to, for the Host check
        (make_server sets the port after binding). `mock_port` is the mock seeker's port (mentor +1, judge +2)."""
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"root {self.root} is not a directory")
        self.workspace = Path(workspace).resolve() if workspace else self.root / "workspace"
        self.studies_dir = Path(studies_dir).resolve() if studies_dir else self.root / "studies"
        self.host, self.port, self.mock_port = host, port, mock_port
        self.static_dir = STATIC_DIR
        self.jobs = JobManager(self.root, self.workspace / "jobs")
        self._mock: MockBackend | None = None
        self._mock_lock = threading.Lock()
        get, post, put, delete = "GET", "POST", "PUT", "DELETE"
        self._routes: list[tuple[tuple[str, ...], dict]] = [
            (("",), {get: self._index}),
            (("static", "**"), {get: self._static}),
            (("api", "meta"), {get: self._meta}),
            (("api", "studies"), {get: self._list_studies}),
            (("api", "studies", "*"), {get: self._get_study, put: self._put_study, delete: self._delete_study}),
            (("api", "study", "validate"), {post: self._validate}),
            (("api", "study", "cells"), {post: self._cells}),
            (("api", "study", "render"), {post: self._render}),
            (("api", "study", "manifest"), {post: self._manifest}),
            (("api", "study", "export"), {post: self._export}),
            (("api", "study", "repo-files"), {post: self._repo_files}),
            (("api", "runs"), {get: self._list_runs}),
            (("api", "runs", "*"), {get: self._run_summary}),
            (("api", "runs", "*", "dyads", "*"), {get: self._dyad}),
            (("api", "runs", "*", "tail"), {get: self._tail}),
            (("api", "runs", "*", "analysis"), {get: self._analysis}),
            (("api", "runs", "*", "config"), {post: self._run_config}),
            (("api", "jobs"), {get: self._list_jobs, post: self._start_job}),
            (("api", "jobs", "*"), {get: self._get_job}),
            (("api", "jobs", "*", "log"), {get: self._job_log}),
            (("api", "jobs", "*", "stop"), {post: self._stop_job}),
            (("api", "mock"), {get: self._mock_info}),
            (("api", "mock", "start"), {post: self._mock_start}),
            (("api", "mock", "stop"), {post: self._mock_stop}),
        ]

    def close(self) -> None:
        """Stop the mock backend, if one runs. Jobs are left running: they are in their own sessions on
        purpose, so a restarted GUI never takes a multi-day run down with it."""
        with self._mock_lock:
            backend, self._mock = self._mock, None
        if backend is not None:
            backend.stop()

    # -- the one entry point

    def handle(self, method: str, path: str, query: dict, body: bytes, headers: dict) -> tuple[int, dict, bytes]:
        """Answer one request: (status, response headers, body bytes). `path` is the URL path still
        percent-encoded (each segment is unquoted after splitting, so an encoded '/' never makes a segment);
        `query` maps each name to a value or a list of values (the first counts). Never raises."""
        try:
            return self._handle(method.upper(), path, query, body or b"", headers or {})
        except HTTPError as e:
            return json_response(e.status, {"error": str(e), **e.extra}, e.headers)
        except study.StudyError as e:
            return json_response(400, {"error": str(e), "issues": e.issues})
        except JobConflict as e:
            return json_response(409, {"error": str(e)})
        except runs.RunNotFound as e:
            return json_response(404, {"error": str(e)})
        except ValueError as e:
            return json_response(400, {"error": str(e)})
        except Exception as e:
            print(f"sandbox: internal error on {method} {path}", file=sys.stderr)
            traceback.print_exc()
            return json_response(500, {"error": f"internal error: {type(e).__name__}: {e}"})

    def _handle(self, method, path, query, body, headers):
        h = {str(k).lower(): str(v) for k, v in headers.items()}
        if not self.host_allowed(h.get("host")):
            raise HTTPError(403, f"Host {h.get('host')!r} is not this server (use http://127.0.0.1:{self.port}/)")
        if method != "GET":
            if h.get("x-sandbox", "").strip() != "1":
                raise HTTPError(403, "a request that changes anything needs the header X-Sandbox: 1")
            if h.get("content-type", "").split(";")[0].strip().lower() != "application/json":
                raise HTTPError(403, "a request that changes anything needs Content-Type: application/json")
        if len(body) > MAX_BODY:
            raise HTTPError(413, f"the request body is larger than {MAX_BODY >> 20} MB")
        if not isinstance(path, str) or not path.startswith("/"):
            raise HTTPError(404, f"no such path {path!r}")
        segments = [urllib.parse.unquote(s) for s in path.split("/")[1:]]
        handlers, args = self._route(segments)
        if handlers is None:
            raise HTTPError(404, f"no such path {path}")
        handler = handlers.get(method)
        if handler is None:
            allowed = ", ".join(sorted(handlers))
            raise HTTPError(405, f"{method} is not allowed on {path} (allowed: {allowed})", {"Allow": allowed})
        q = {}
        for k, v in (query or {}).items():
            if isinstance(v, (list, tuple)):
                v = v[0] if v else None
            q[str(k)] = v
        req = _Request(method, args, q, body)
        if method in ("POST", "PUT", "DELETE"):
            req.body()                           # a malformed body is a 400 on every route, not only some
        out = handler(req)
        if isinstance(out, _File):
            return 200, {"Content-Type": out.content_type, "Cache-Control": "no-store",
                         "X-Content-Type-Options": "nosniff"}, out.data
        if isinstance(out, tuple):
            return json_response(*out)
        return json_response(200, out)

    def _route(self, segments: list[str]) -> tuple[dict | None, list]:
        for pattern, handlers in self._routes:
            if pattern and pattern[-1] == "**":
                head = pattern[:-1]
                if len(segments) >= len(head) and all(p == "*" or p == s for p, s in zip(head, segments)):
                    return handlers, segments[len(head):]
                continue
            if len(pattern) == len(segments) and all(p == "*" or p == s for p, s in zip(pattern, segments)):
                return handlers, [s for p, s in zip(pattern, segments) if p == "*"]
        return None, []

    def host_allowed(self, host) -> bool:
        """True when the Host header names this server: the bound host, localhost, 127.0.0.1 or [::1], with no
        port or with the bound port. A DNS-rebinding page reaches 127.0.0.1 under its own host name."""
        if not isinstance(host, str) or not host.strip():
            return False
        host = host.strip().lower()
        if host.startswith("["):
            end = host.find("]")
            if end < 0:
                return False
            name, rest = host[:end + 1], host[end + 1:]
        elif ":" in host:
            name, port = host.rsplit(":", 1)
            rest = ":" + port
        else:
            name, rest = host, ""
        if rest and rest != f":{self.port}":
            return False
        bound = self.host.lower()
        return name in {*LOCAL_NAMES, f"[{bound}]" if ":" in bound else bound}

    # -- helpers

    def show(self, path) -> str:
        """A path as the CLI run from the root sees it (sandbox.export.display_path)."""
        return export.display_path(path, self.root)

    def _data_dir(self, value):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"data_dir must be a path string, got {type(value).__name__}")
        if value and "\0" in value:
            raise ValueError("data_dir must not contain a NUL byte")
        return runs.resolve_data_dir(self.root, value)

    def _saved_studies(self) -> list[tuple[str, Path]]:
        if not self.studies_dir.is_dir():
            return []
        out = []
        for p in sorted(self.studies_dir.glob(f"*{STUDY_SUFFIX}")):
            name = p.name[:-len(STUDY_SUFFIX)]
            if STUDY_NAME_RE.match(name) and name not in PRESETS and p.is_file():
                out.append((name, p))
        return out

    @staticmethod
    def _read_spec(path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    def _study_path(self, name: str) -> Path:
        if not isinstance(name, str) or len(name) > 100 or not STUDY_NAME_RE.match(name):
            raise HTTPError(400, f"a study name matches {STUDY_NAME_RE.pattern} (at most 100 characters), "
                                 f"got {name!r}")
        return self.studies_dir / f"{name}{STUDY_SUFFIX}"

    def _repo_spec(self) -> dict:
        try:
            return repo_study.load_repo_study(self.root)
        except (OSError, ValueError) as e:
            raise HTTPError(500, f"cannot build the repo study from the repo's files: {e}") from None

    @staticmethod
    def _spec(req: _Request):
        return req.body().get("spec")

    # -- static

    def _index(self, req):
        return self._file(["index.html"])

    def _static(self, req):
        return self._file(req.args)

    def _file(self, parts: list[str]) -> _File:
        base = Path(self.static_dir).resolve()
        try:
            target = base.joinpath(*parts).resolve() if parts else base
            ok = base in target.parents and target.is_file()
        except (OSError, ValueError):
            ok = False
        if not ok:
            raise HTTPError(404, "no such file" + ("" if base.is_dir() else " (the app's static files are missing)"))
        return _File(MIME.get(target.suffix.lower(), "application/octet-stream"), target.read_bytes())

    # -- meta and studies

    def _meta(self, req):
        return {"root": str(self.root), "harness_version": HARNESS_VERSION, "sandbox_version": SANDBOX_VERSION,
                "python": sys.executable, "python_version": platform.python_version(),
                "workspace": self.show(self.workspace), "studies_dir": self.show(self.studies_dir),
                "data_dirs": self._data_dirs(), "randomization_presets": repo_study.RANDOMIZATION_PRESETS,
                "gguf_importable": importlib.util.find_spec("gguf") is not None,
                "repo_sources": {k: self.show(p) for k, p in repo_study.repo_sources(self.root).items()}}

    def _data_dirs(self) -> list[str]:
        """"data", then every other run.data_dir named by the repo study and the saved studies."""
        specs = []
        try:
            specs.append(repo_study.load_repo_study(self.root))
        except Exception:
            pass
        for _, path in self._saved_studies():
            try:
                specs.append(self._read_spec(path))
            except (OSError, ValueError):
                continue
        out = ["data"]
        for spec in specs:
            run = spec.get("run") if isinstance(spec, dict) else None
            value = run.get("data_dir") if isinstance(run, dict) else None
            if not isinstance(value, str) or not value.strip() or "\0" in value:
                continue
            try:
                shown = self.show(runs.resolve_data_dir(self.root, value))
            except (OSError, ValueError):
                continue
            if shown not in out:
                out.append(shown)
        return out

    def _list_studies(self, req):
        out = [{"name": name, "title": title, "kind": "preset", "path": None} for name, title in PRESETS.items()]
        for name, path in self._saved_studies():
            try:
                spec = self._read_spec(path)
                title = spec.get("name") if isinstance(spec, dict) and isinstance(spec.get("name"), str) else name
            except (OSError, ValueError):
                title = "(unreadable)"
            out.append({"name": name, "title": title, "kind": "saved", "path": self.show(path)})
        return {"studies": out}

    def _get_study(self, req):
        name = req.args[0]
        if name == "repo":
            return {"name": name, "kind": "preset", "path": None, "spec": self._repo_spec()}
        if name == "blank":
            return {"name": name, "kind": "preset", "path": None, "spec": study.blank_study()}
        path = self._study_path(name)
        if not path.is_file():
            raise HTTPError(404, f"no saved study {name!r} in {self.show(self.studies_dir)}")
        try:
            spec = self._read_spec(path)
        except (OSError, ValueError) as e:
            raise HTTPError(500, f"{self.show(path)} cannot be read as JSON: {e}") from None
        return {"name": name, "kind": "saved", "path": self.show(path), "spec": spec}

    def _put_study(self, req):
        name = req.args[0]
        if name in PRESETS:
            raise HTTPError(409, f"{name!r} is a preset and cannot be overwritten; save under another name")
        path = self._study_path(name)
        spec = self._spec(req)
        if not isinstance(spec, dict):
            raise HTTPError(400, "the body is {\"spec\": <the study object>}")
        self.studies_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)                    # a reader never sees half a file
        return {"name": name, "path": self.show(path), "issues": study.validate_study(spec)}

    def _delete_study(self, req):
        name = req.args[0]
        if name in PRESETS:
            raise HTTPError(409, f"{name!r} is a preset and cannot be deleted")
        path = self._study_path(name)
        try:
            path.unlink()
        except FileNotFoundError:
            raise HTTPError(404, f"no saved study {name!r}") from None
        return {"name": name, "path": self.show(path), "deleted": True}

    # -- the study

    def _validate(self, req):
        spec = self._spec(req)
        issues = study.validate_study(spec)                 # never raises
        try:
            summary = study.summarize(spec)
        except Exception:
            summary = dict(_EMPTY_SUMMARY)
        try:
            slots = study.slot_catalog(spec)
        except Exception:
            slots = {"treated": [], "control": []}
        try:
            reason = repo_study.repo_shape_reason(spec)
        except Exception as e:
            reason = f"the study's shape could not be checked ({type(e).__name__}: {e})"
        return {"issues": issues, "summary": summary, "slots": slots, "repo_shaped": reason is None,
                "repo_shape_reason": reason}

    def _cells(self, req):
        spec = self._spec(req)
        try:
            cells = study.enumerate_cells(spec)          # best effort on drafts already; a view never 500s
        except Exception:
            cells = []
        return {"cells": cells}

    def _render(self, req):
        b = req.body()
        return study.render_cell(b.get("spec"), b.get("kind"), b.get("condition"), b.get("variant"))

    def _manifest(self, req):
        b = req.body()
        limit = _int_param(b.get("limit"), "limit", minimum=0)
        limit = MANIFEST_LIMIT if limit is None else min(limit, MANIFEST_LIMIT_MAX)
        rows, assignment = study.compile_manifest(b.get("spec"))
        return {"rows": rows[:limit], "total": len(rows), "assignment": assignment}

    def _export(self, req):
        return export.export_study(self._spec(req), root=self.root, workspace=self.workspace, python="python")

    def _repo_files(self, req):
        try:
            grid, catalogue = repo_study.to_repo_files(self._spec(req))
        except repo_study.NotRepoShaped as e:
            raise HTTPError(400, str(e)) from None
        return {"grid": grid, "catalogue": catalogue}

    # -- runs

    def _list_runs(self, req):
        data_dir = self._data_dir(req.query.get("data_dir"))
        return {"data_dir": self.show(data_dir), "runs": runs.list_runs(data_dir, root=self.root)}

    def _run_summary(self, req):
        return runs.run_summary(self._data_dir(req.query.get("data_dir")), req.args[0], root=self.root)

    def _dyad(self, req):
        attempt = _int_param(req.query.get("attempt"), "attempt", minimum=1)
        return runs.dyad_detail(self._data_dir(req.query.get("data_dir")), req.args[0], req.args[1], attempt)

    def _tail(self, req):
        offsets = {}
        for name in runs.TAIL_FILES:
            n = _int_param(req.query.get(name), name, minimum=0)
            if n is not None:
                offsets[name] = n
        return runs.tail(self._data_dir(req.query.get("data_dir")), req.args[0], offsets)

    def _analysis(self, req):
        q = req.query
        return analysis.analyze_run(self._data_dir(q.get("data_dir")), req.args[0], factor=q.get("factor") or None,
                                    metric=q.get("metric") or "alignment", judge=q.get("judge") or None,
                                    root=self.root)

    def _run_config(self, req):
        b = req.body()
        path = export.config_from_manifest(self._data_dir(b.get("data_dir")), req.args[0], self.workspace,
                                           judge=b.get("judge"), root=self.root)
        return {"path": self.show(path)}

    # -- jobs

    def _job_call(self, fn, job_id, *args):
        try:
            return fn(job_id, *args)
        except KeyError:
            raise HTTPError(404, f"no job {job_id!r}") from None

    def _list_jobs(self, req):
        return {"jobs": self.jobs.list()}

    def _start_job(self, req):
        b = req.body()
        kind = b.get("kind")
        if not isinstance(kind, str) or not kind:
            raise HTTPError(400, "the body is {\"kind\": \"check\" | \"run\" | ..., <its fields>}")
        return 201, self.jobs.start(kind, b)

    def _get_job(self, req):
        return self._job_call(self.jobs.get, req.args[0])

    def _job_log(self, req):
        offset = _int_param(req.query.get("offset"), "offset", minimum=0) or 0
        text, new_offset = self._job_call(self.jobs.read_log, req.args[0], offset)
        return {"job": self._job_call(self.jobs.get, req.args[0]), "text": text, "offset": new_offset}

    def _stop_job(self, req):
        return self._job_call(self.jobs.stop, req.args[0])

    # -- the mock backend

    def _mock_info(self, req=None):
        with self._mock_lock:
            if self._mock is not None and self._mock.running:
                return self._mock.info()
        return dict(_NO_MOCK)

    def _mock_start(self, req):
        """Start the three mock servers, or return the running ones (idempotent: a second Start in another tab
        must not orphan the first backend's ports)."""
        b = req.body()
        slots = _int_param(b.get("slots"), "slots", minimum=1)
        slots = 8 if slots is None else slots
        if slots > MOCK_MAX_SLOTS:
            raise ValueError(f"slots must be at most {MOCK_MAX_SLOTS}, got {slots}")
        delay = b.get("delay")
        delay = 0.0 if delay is None or delay == "" else delay
        if isinstance(delay, bool) or not isinstance(delay, (int, float)):
            raise ValueError(f"delay must be seconds per word (a number), got {delay!r}")
        with self._mock_lock:
            if self._mock is not None and self._mock.running:
                return self._mock.info()
            directory = self.workspace / "mock"
            try:
                backend = MockBackend(directory, slots=slots, delay=float(delay), base_port=self.mock_port)
                info = backend.start()
            except OSError:
                if not self.mock_port:
                    raise
                backend = MockBackend(directory, slots=slots, delay=float(delay), base_port=0)
                info = backend.start()
            self._mock = backend
            return info

    def _mock_stop(self, req):
        self.close()
        return dict(_NO_MOCK)


# --- the socket ------------------------------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    server_version = f"sandbox/{SANDBOX_VERSION}"

    def log_message(self, fmt, *args):            # quiet: the GUI polls every second
        pass

    def log_error(self, fmt, *args):
        sys.stderr.write(f"sandbox: {self.address_string()} {fmt % args}\n")

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def _send(self, status: int, headers: dict, data: bytes) -> None:
        try:
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _dispatch(self, method: str) -> None:
        split = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(split.query, keep_blank_values=True)
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            self.close_connection = True
            return self._send(*json_response(411, {"error": "send the body with a Content-Length"}))
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0:
            self.close_connection = True
            return self._send(*json_response(400, {"error": "bad Content-Length"}))
        if length > MAX_BODY:
            self.close_connection = True                       # the unread body must not be read as a request
            return self._send(*json_response(413, {"error": f"the request body is larger than {MAX_BODY >> 20} MB"}))
        body = self.rfile.read(length) if length else b""
        self._send(*self.server.app.handle(method, split.path, query, body, dict(self.headers.items())))


class SandboxHTTPServer(ThreadingHTTPServer):
    """The stdlib threading server with the App attached; closing it stops the mock backend."""
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, app: App):
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        self.app = app
        super().__init__(address, handler)

    def server_close(self):
        super().server_close()
        self.app.close()


def make_server(root, *, host: str = "127.0.0.1", port: int = 8765, workspace=None, studies_dir=None,
                mock_port: int = MOCK_BASE_PORT) -> SandboxHTTPServer:
    """Bind the API to (host, port) and return the server, not yet serving (serve_forever). `server.app` is the
    App; port 0 binds an ephemeral port, and app.port is the port actually bound (the Host check needs it)."""
    app = App(root, workspace=workspace, studies_dir=studies_dir, host=host, port=port, mock_port=mock_port)
    server = SandboxHTTPServer((host, port), _Handler, app)
    app.port = server.server_address[1]
    return server


def _loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> int:
    """`python -m sandbox`: serve the GUI until Ctrl-C."""
    ap = argparse.ArgumentParser(prog="python -m sandbox",
                                 description="The sandbox GUI over the dyad harness (sandbox/README.md).")
    ap.add_argument("--host", default="127.0.0.1", help="interface to bind (default 127.0.0.1; there is no "
                                                        "authentication, so keep it on loopback and tunnel)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--root", default=str(REPO), help="the repository root (default: this checkout)")
    ap.add_argument("--workspace", default=None,
                    help="exports, configs, job logs, mock GGUFs (default <root>/workspace)")
    ap.add_argument("--studies", default=None, help="saved studies (default <root>/studies)")
    ap.add_argument("--open", action="store_true", help="open the GUI in a browser")
    a = ap.parse_args(argv)

    if threading.current_thread() is threading.main_thread():
        # Started under nohup or with `&` from a script, this process inherits SIGINT ignored, and so would every
        # job it starts (an ignored signal stays ignored across exec), so the first Stop -- a SIGINT -- would do
        # nothing. A handled signal is reset to the default in the child, so handle it here.
        signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        server = make_server(a.root, host=a.host, port=a.port, workspace=a.workspace, studies_dir=a.studies)
    except (OSError, ValueError) as e:
        print(f"error: cannot serve on {a.host}:{a.port}: {e}", file=sys.stderr)
        return 1
    port = server.app.port
    shown_host = "127.0.0.1" if a.host in ("", "0.0.0.0", "::") else (f"[{a.host}]" if ":" in a.host else a.host)
    url = f"http://{shown_host}:{port}/"
    print(f"sandbox {SANDBOX_VERSION}: {url}  (root {server.app.root}, workspace {server.app.workspace})")
    print(f"from another machine: ssh -L {port}:127.0.0.1:{port} <box>   then open http://127.0.0.1:{port}/")
    if not _loopback(a.host):
        print("\n" + "!" * 100 + f"\nWARNING: bound to {a.host!r}, not loopback. There is NO authentication, and this "
              "GUI starts harness runs\nand writes files. Anyone who can reach this port can use it. Bind 127.0.0.1 "
              "and use an SSH tunnel instead.\n(Requests must name the host as "
              f"{a.host}, localhost or 127.0.0.1.)\n" + "!" * 100 + "\n", file=sys.stderr)
    print("Ctrl-C to stop (jobs it started keep running; the mock backend stops)", flush=True)
    if a.open:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        server.server_close()
    return 0
