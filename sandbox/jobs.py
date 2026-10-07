"""Harness commands as background jobs for the GUI: `python -m harness.run <kind> ...` run as a subprocess,
so a sandbox run is a harness run with all of its guards and provenance.

The argv is built here, only from validated structured fields -- never from a string the client sent --
and the process is started without a shell. Output goes to `<log_dir>/<job_id>.log`, each job's record
to `<log_dir>/<job_id>.json`, so a restarted sandbox still lists earlier jobs. A job is stopped the way an
operator stops the CLI: SIGINT, so `run` finishes its in-flight dyads, starts no queued ones and exits 130
(resume with the same run id); a second stop while it is still running sends SIGKILL.

Jobs run in their own session (process group), so a Ctrl-C or a dropped SSH connection on the terminal
running the sandbox does not take a multi-day wave down with it. The price: a sandbox restarted while a
job runs can no longer follow it, and lists it as "lost" (its log and data/<run_id>/ keep growing)."""
from __future__ import annotations
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from harness.scorer import ADHERENCE_METRICS, METRICS, SCOPES
from harness.survey import PHASES

JOB_KINDS = ("check", "run", "survey", "score", "flags", "agreement")
EXIT_MEANINGS: dict[int, str] = {
    0: "ok",
    1: "refused or error (see log)",
    2: "some dyads failed",
    130: "stopped - resume with the same run id",
}
STATUSES = ("running", "finished", "failed", "stopped", "lost")

# The fields each kind takes: exactly the options of harness/run.py main() for that subcommand.
FIELDS: dict[str, tuple[str, ...]] = {
    "check": ("config", "manifest"),
    "run": ("config", "manifest", "run_id"),
    "survey": ("config", "run_id", "phase"),
    "score": ("config", "run_id", "scope", "subsample"),
    "flags": ("config", "run_id", "threshold", "metric", "run_length", "judge"),
    "agreement": ("config", "run_id", "metric"),
}
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
JOB_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_KIND_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_HEX_RE = re.compile(r"^[0-9a-fA-F]{1,64}$")


class JobConflict(ValueError):
    """A job of the same kind is already running on the same run id (two `run`s on one run id would run
    the same dyads twice)."""


# ------------------------------------------------------------------------------------------------ argv

def _absent(v) -> bool:
    """A form field left empty counts as not given."""
    return v is None or (isinstance(v, str) and v.strip() == "")


def _path(fields: dict, name: str, kind: str) -> str:
    """A path argument: a non-empty string that cannot be read as an option and holds no NUL."""
    v = fields.get(name)
    if _absent(v):
        raise ValueError(f"{kind}: {name} is required")
    if isinstance(v, os.PathLike):
        v = os.fspath(v)
    if not isinstance(v, str):
        raise ValueError(f"{kind}: {name} must be a path string, got {type(v).__name__}")
    if v.startswith("-"):
        raise ValueError(f"{kind}: {name} must not start with '-' (it would read as an option): {v!r}")
    if "\0" in v:
        raise ValueError(f"{kind}: {name} must not contain a NUL byte")
    return v


def _run_id(fields: dict, kind: str) -> str:
    """A run id: the harness writes data/<run_id>/, so it must be a plain directory name."""
    v = fields.get("run_id")
    if _absent(v):
        raise ValueError(f"{kind}: run_id is required")
    if not isinstance(v, str) or not RUN_ID_RE.match(v) or v.startswith("-") or set(v) == {"."}:
        raise ValueError(f"{kind}: run_id must match {RUN_ID_RE.pattern}, not start with '-' and not be "
                         f"'.' or '..', got {v!r}")
    return v


def _choice(fields: dict, name: str, choices, kind: str, default: str | None = None) -> str | None:
    """One of a fixed set of words, or the default when not given."""
    v = fields.get(name)
    if _absent(v):
        return default
    if v not in choices:
        raise ValueError(f"{kind}: {name} must be one of {', '.join(choices)}, got {v!r}")
    return v


def _float(fields: dict, name: str, kind: str) -> float | None:
    """A finite number (a JSON number or a numeric string), or None when not given."""
    v = fields.get(name)
    if _absent(v):
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise ValueError(f"{kind}: {name} must be a number, got {v!r}")
    try:
        x = float(v)
    except ValueError:
        raise ValueError(f"{kind}: {name} must be a number, got {v!r}") from None
    if not math.isfinite(x):
        raise ValueError(f"{kind}: {name} must be finite, got {v!r}")
    return x


def _int(fields: dict, name: str, kind: str) -> int | None:
    """A whole number (a JSON integer, an integral float or a digit string), or None when not given."""
    v = fields.get(name)
    if _absent(v):
        return None
    if isinstance(v, bool):
        raise ValueError(f"{kind}: {name} must be an integer, got {v!r}")
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and re.fullmatch(r"\s*\d+\s*", v):
        return int(v)
    raise ValueError(f"{kind}: {name} must be an integer, got {v!r}")


def build_argv(kind: str, fields: dict, python: str = sys.executable) -> list[str]:
    """The argv for `python -m harness.run <kind>`, from structured fields only, checked against the
    options harness/run.py's main() takes. Raises ValueError naming the field for an unknown kind, an
    unknown field, a missing required field, or a value that is malformed or would read as an option."""
    if kind not in JOB_KINDS:
        raise ValueError(f"kind must be one of {', '.join(JOB_KINDS)}, got {kind!r}")
    if not isinstance(fields, dict):
        raise ValueError(f"{kind}: fields must be an object")
    unknown = sorted(str(k) for k in fields if k not in FIELDS[kind])
    if unknown:
        raise ValueError(f"{kind}: unknown field(s) {', '.join(unknown)}; {kind} takes {', '.join(FIELDS[kind])}")
    if not isinstance(python, str) or not python:
        raise ValueError("python must be the path of an interpreter")
    argv = [python, "-m", "harness.run", kind, "--config", _path(fields, "config", kind)]
    if kind != "check":
        argv += ["--run-id", _run_id(fields, kind)]
    if kind == "check":
        if not _absent(fields.get("manifest")):
            argv += ["--manifest", _path(fields, "manifest", kind)]
    elif kind == "run":
        argv += ["--manifest", _path(fields, "manifest", kind)]
    elif kind == "survey":
        argv += ["--phase", _choice(fields, "phase", PHASES, kind, default="post")]
    elif kind == "score":
        argv += ["--scope", _choice(fields, "scope", SCOPES, kind, default="pilot")]
        subsample = _float(fields, "subsample", kind)
        if subsample is not None:
            if not 0 < subsample <= 1:
                raise ValueError(f"{kind}: subsample must be in (0, 1], got {subsample}")
            argv += ["--subsample", repr(subsample)]
    elif kind == "flags":
        threshold = _float(fields, "threshold", kind)
        if threshold is None:
            raise ValueError(f"{kind}: threshold is required (it is calibrated on the pilot's hand labels; "
                             "there is no default on purpose)")
        if not 0 <= threshold <= 1:
            raise ValueError(f"{kind}: threshold must be in [0, 1], the judge's score scale, got {threshold}")
        argv += ["--threshold", repr(threshold)]
        metric = _choice(fields, "metric", ADHERENCE_METRICS, kind)
        if metric:
            argv += ["--metric", metric]
        run_length = _int(fields, "run_length", kind)
        if run_length is not None:
            if run_length < 1:
                raise ValueError(f"{kind}: run_length must be at least 1, got {run_length}")
            argv += ["--run-length", str(run_length)]
        judge = fields.get("judge")
        if not _absent(judge):
            if not isinstance(judge, str) or not _HEX_RE.match(judge):
                raise ValueError(f"{kind}: judge must be a hex sha256 prefix, got {judge!r}")
            argv += ["--judge", judge.lower()]
    elif kind == "agreement":
        metric = _choice(fields, "metric", tuple(METRICS), kind)
        if metric:
            argv += ["--metric", metric]
    return argv


def exit_meaning(returncode: int | None) -> str | None:
    """What an exit code means for a harness command; None while the job runs."""
    if returncode is None:
        return None
    if returncode in EXIT_MEANINGS:
        return EXIT_MEANINGS[returncode]
    if returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except ValueError:
            name = f"signal {-returncode}"
        return f"ended by {name}"
    return f"exit code {returncode}"


# ------------------------------------------------------------------------------------------------ jobs

def _now() -> str:
    """The harness's timestamp format (harness.log.now_iso): local time with its UTC offset."""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _utf8_complete(data: bytes) -> int:
    """Length of the longest prefix of `data` that does not end inside a UTF-8 sequence."""
    n = len(data)
    i = n - 1
    while i >= 0 and n - i <= 4 and 0x80 <= data[i] < 0xC0:       # continuation bytes
        i -= 1
    if i < 0 or n - i > 4:
        return n
    lead = data[i]
    need = 2 if 0xC0 <= lead < 0xE0 else 3 if 0xE0 <= lead < 0xF0 else 4 if 0xF0 <= lead < 0xF8 else 1
    return i if n - i < need else n


@dataclass
class _Job:
    record: dict
    seq: int
    proc: subprocess.Popen | None = None
    done: threading.Event = field(default_factory=threading.Event)
    stops: int = 0


class JobManager:
    """Starts, follows, stops and remembers harness jobs. Thread-safe: the HTTP server calls it from
    many request threads at once."""

    def __init__(self, root, log_dir, python: str = sys.executable):
        """`root` is the working directory of every job (the repo root, so `-m harness.run` and the
        config's relative paths resolve); `log_dir` holds each job's log and record. Records left by an
        earlier process are loaded; one still "running" becomes "lost", since nothing follows it now."""
        self.root = Path(root).resolve()
        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.python = python
        self._lock = threading.RLock()
        self._jobs: dict[str, _Job] = {}
        self._seq = 0
        self._load()

    # -- persistence

    def _persist(self, record: dict) -> None:
        """Write the record atomically, so a crash mid-write never leaves half a JSON file behind."""
        path = self.log_dir / f"{record['id']}.json"
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    @staticmethod
    def _seq_of(job_id: str) -> int:
        m = re.search(r"-(\d+)$", job_id)
        return int(m.group(1)) if m else 0

    def _load(self) -> None:
        for path in sorted(self.log_dir.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(record, dict) or record.get("id") != path.stem or not JOB_ID_RE.match(path.stem):
                continue
            if record.get("status") == "running":
                pid = record.get("pid")
                record["status"] = "lost"
                record["meaning"] = ("lost: started by an earlier sandbox process, which can no longer follow it; "
                                     f"it may still be running{f' (pid {pid})' if pid else ''} -- see its log "
                                     "and the run's data directory")
                try:
                    self._persist(record)
                except OSError:
                    pass
            seq = self._seq_of(record["id"])
            job = _Job(record, seq)
            job.done.set()
            self._jobs[record["id"]] = job
            self._seq = max(self._seq, seq)

    # -- starting

    def _new_id(self, kind: str) -> str:
        while True:
            self._seq += 1
            job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}-{self._seq}"
            if job_id not in self._jobs and not (self.log_dir / f"{job_id}.log").exists():
                return job_id

    def start(self, kind: str, fields: dict, label: str = "") -> dict:
        """Start `python -m harness.run <kind>` from structured fields (build_argv). A `kind` key in
        fields is accepted when it matches, and a `label` key is used when no label is given, so an API
        body can be passed as it came. Raises ValueError for bad fields, JobConflict for a duplicate."""
        fields = dict(fields) if isinstance(fields, dict) else fields
        if isinstance(fields, dict):
            if "kind" in fields:
                if fields.pop("kind") != kind:
                    raise ValueError("fields.kind does not match kind")
            body_label = fields.pop("label", None)
            label = label or (body_label if isinstance(body_label, str) else "")
        argv = build_argv(kind, fields, python=self.python)
        run_id = fields.get("run_id") if kind != "check" else None
        job = self.start_argv(argv, kind=kind, label=label, run_id=run_id)
        data_dir = self._config_data_dir(fields.get("config"))
        with self._lock:
            record = self._jobs[job["id"]].record
            record["data_dir"] = data_dir
            self._persist(record)
            return self._copy(record)

    def _config_data_dir(self, config) -> str | None:
        """The data_dir the job's config names, as written there (relative to the root the CLI runs in;
        the harness default "data" when it names none), so the GUI can open the run the job writes.
        None when the config cannot be read: the harness will say why in the job's log."""
        try:
            path = Path(config)
            cfg = json.loads((path if path.is_absolute() else self.root / path).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        value = cfg.get("data_dir") if isinstance(cfg, dict) else None
        return value if isinstance(value, str) and value else "data"

    def start_argv(self, argv: list[str], *, kind: str, label: str = "", run_id: str | None = None) -> dict:
        """Start any argv as a job (no shell; cwd root; PYTHONUNBUFFERED=1 so the log is live). For the
        tests and internal callers: the API goes through start(), which builds the argv itself."""
        if not isinstance(argv, (list, tuple)) or not argv or not all(isinstance(a, str) for a in argv):
            raise ValueError("argv must be a non-empty list of strings")
        if not isinstance(kind, str) or not _KIND_RE.match(kind):
            raise ValueError(f"kind must match {_KIND_RE.pattern}, got {kind!r}")
        if not isinstance(label, str):
            raise ValueError("label must be a string")
        label = label.strip()[:200]
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        with self._lock:
            if run_id is not None:
                for other in self._jobs.values():
                    r = other.record
                    if r["status"] == "running" and r["kind"] == kind and r.get("run_id") == run_id:
                        raise JobConflict(f"a {kind} job on run_id {run_id} is already running ({r['id']}); "
                                          "stop it or wait for it to end")
            job_id = self._new_id(kind)
            log_path = self.log_dir / f"{job_id}.log"
            record = {"id": job_id, "kind": kind, "label": label, "argv": list(argv), "run_id": run_id,
                      "status": "running", "returncode": None, "meaning": None, "started_at": _now(),
                      "ended_at": None, "log_path": str(log_path), "pid": None, "stop_requested": False,
                      "data_dir": None}
            job = _Job(record, self._seq)
            with open(log_path, "wb") as log_file:
                log_file.write(f"$ {shlex.join(argv)}\n".encode("utf-8"))
                log_file.flush()
                try:
                    job.proc = subprocess.Popen(list(argv), cwd=self.root, stdin=subprocess.DEVNULL,
                                                stdout=log_file, stderr=subprocess.STDOUT, env=env,
                                                start_new_session=True)
                except (OSError, ValueError) as e:
                    log_file.write(f"could not start: {e}\n".encode("utf-8"))
                    record.update(status="failed", meaning=f"could not start: {e}", ended_at=_now())
                    job.done.set()
            self._jobs[job_id] = job
            if job.proc is not None:
                record["pid"] = job.proc.pid
            self._persist(record)
            snapshot = self._copy(record)
        if job.proc is not None:
            threading.Thread(target=self._watch, args=(job,), name=f"job-{job_id}", daemon=True).start()
        return snapshot

    def _watch(self, job: _Job) -> None:
        """Wait for the process and record how it ended. rc 0 is finished even after a stop request (it
        had nothing left to stop); any other code is stopped if a stop was requested, failed otherwise."""
        rc = job.proc.wait()
        with self._lock:
            r = job.record
            r["returncode"] = rc
            r["ended_at"] = _now()
            r["status"] = "finished" if rc == 0 else ("stopped" if job.stops else "failed")
            r["meaning"] = exit_meaning(rc)
            try:
                with open(r["log_path"], "ab") as f:
                    f.write(f"\n[{r['status']}: exit {rc} -- {r['meaning']}]\n".encode("utf-8"))
            except OSError:
                pass
            self._persist(r)
        job.done.set()

    # -- reading

    @staticmethod
    def _copy(record: dict) -> dict:
        return {**record, "argv": list(record.get("argv") or [])}

    def _job(self, job_id: str) -> _Job:
        job = self._jobs.get(job_id) if isinstance(job_id, str) else None
        if job is None:
            raise KeyError(job_id)
        return job

    def list(self) -> list[dict]:
        """Every job this manager knows, newest first."""
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: (j.record.get("started_at") or "", j.seq), reverse=True)
            return [self._copy(j.record) for j in jobs]

    def get(self, job_id: str) -> dict:
        """One job's record; KeyError for an unknown id."""
        with self._lock:
            return self._copy(self._job(job_id).record)

    def read_log(self, job_id: str, offset: int = 0, max_bytes: int = 65536) -> tuple[str, int]:
        """Log text from byte `offset`, at most `max_bytes` of it, and the offset to ask from next. While
        the job runs, a UTF-8 sequence cut off at the end is left for the next read rather than mangled."""
        with self._lock:
            job = self._job(job_id)
            path = Path(job.record["log_path"])
            running = job.record["status"] == "running"
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError(f"offset must be a non-negative integer, got {offset!r}")
        max_bytes = max(int(max_bytes), 4)
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return "", 0
        if offset >= size:
            return "", size
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read(max_bytes)
        cut = _utf8_complete(data)
        if cut < len(data) and (running or offset + len(data) < size):
            data = data[:cut]
        return data.decode("utf-8", errors="replace"), offset + len(data)

    # -- stopping and waiting

    def stop(self, job_id: str) -> dict:
        """First call: SIGINT to the job's process group, as Ctrl-C in a terminal would (`run` finishes
        in-flight dyads and exits 130). A second call while it still runs: SIGKILL. A job that has ended,
        or was lost, is returned unchanged."""
        with self._lock:
            job = self._job(job_id)
            r = job.record
            if job.proc is None or r["status"] != "running" or job.proc.returncode is not None:
                return self._copy(r)
            job.stops += 1
            r["stop_requested"] = True
            sig = signal.SIGINT if job.stops == 1 else signal.SIGKILL
            try:
                os.killpg(job.proc.pid, sig)
            except ProcessLookupError:
                pass
            except PermissionError:
                job.proc.send_signal(sig)
            self._persist(r)
            return self._copy(r)

    def wait(self, job_id: str, timeout: float | None = None) -> dict:
        """Block until the job has ended and its record is written (or `timeout` seconds pass), then
        return the record -- still "running" if the timeout came first."""
        with self._lock:
            job = self._job(job_id)
        job.done.wait(timeout)
        return self.get(job_id)
