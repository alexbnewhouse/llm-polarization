"""Read a run's files under data/<run_id>/ for the GUI: list the runs in a data directory, summarise one,
one dyad's transcript, survey and scores, tail new rows by byte offset, and the order of each axis's levels.

Read-only: the harness writes these files (data/README.md is their dictionary) and nothing here ever does.
Runs are read while the harness is still appending to them, so every reader takes whole lines only: a row
the writer is halfway through is left for the next read instead of failing the request. Run ids and dyad
ids are checked against RUN_ID_RE and every path is resolved and kept inside the data directory, because
both arrive from an HTTP request."""
from __future__ import annotations
import json
import os
import re
import threading
from pathlib import Path
from harness.grid import load_grid
from harness.log import resume_index
from harness.scorer import METRICS, latest_complete_attempts
from harness.transcript import message_order

REPO = Path(__file__).resolve().parents[1]
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
TAIL_FILES = ("turns", "status", "surveys", "scores")
# What the GUI shows of a manifest role block. template_source, server_chat_template and the server's
# default_generation_settings stay in manifest.json: they are kilobytes per role and only provenance.
ROLE_INFO_KEYS = ("url", "alias", "model_path", "model_sha256", "family", "template_sha256", "build_info",
                  "total_slots")
JUDGE_INFO_KEYS = ("alias", "model_path", "model_sha256", "scope", "subsample", "ts")
MANIFEST_KEYS = ("started_at", "harness_commit", "harness_dirty", "input_manifest", "batteries", "environment",
                 "config")
NONE_LEVEL = "(none)"
ALL_LEVEL = "all"
# A tail returns at most about this much of each file per call; the client resumes from the offsets. A wave's
# turns.jsonl runs to hundreds of MB, and one response must not try to carry all of it.
TAIL_MAX_BYTES = 8 << 20


class RunNotFound(LookupError):
    """No such run, dyad or attempt in the data directory: a 404, not an operator error."""


def check_id(value, what: str = "run_id") -> str:
    """Return `value` if it is a safe run or dyad id, else raise ValueError naming the field. `.` and `..`
    match the character class but name directories, and a leading `-` reads as an option to a CLI."""
    if (not isinstance(value, str) or not RUN_ID_RE.fullmatch(value) or value in (".", "..")
            or value.startswith("-")):
        raise ValueError(f"{what} {value!r} must match {RUN_ID_RE.pattern}, not be '.' or '..', "
                         "and not start with '-'")
    return value


def resolve_data_dir(root, data_dir=None) -> Path:
    """The data directory as an absolute, resolved path: `root/data` by default, a relative `data_dir`
    against the repo root (where the harness CLI runs, so where its config's data_dir points), an absolute
    one as it is."""
    p = Path(data_dir) if data_dir not in (None, "") else Path("data")
    if not p.is_absolute():
        p = Path(root) / p
    return p.resolve()


def run_dir(data_dir, run_id) -> Path:
    """The resolved directory of one run. ValueError for a bad id or one that resolves (through a symlink)
    outside the data directory; RunNotFound when there is no such directory."""
    check_id(run_id)
    base = Path(data_dir).resolve()
    path = (base / run_id).resolve()
    if base not in path.parents:
        raise ValueError(f"run_id {run_id!r} resolves outside the data directory {base}")
    if not path.is_dir():
        raise RunNotFound(f"no run {run_id!r} in {base}")
    return path


def read_from(path, offset, max_bytes: int | None = None, contains: bytes | None = None) -> tuple[list[dict], int]:
    """The JSONL rows of `path` from byte `offset` on, and the offset to resume from. Whole lines only: a
    trailing line without its newline is a row the harness is still writing and is left for next time. A
    missing file gives ([], offset). An offset past the end means the file was replaced, so it reads from
    the start. With `max_bytes` it stops after about that many bytes, but always on a line end and never
    with nothing when a whole line is there. A line that is not a JSON object is skipped, and so, with
    `contains`, is a line without those bytes: a cheap prefilter that spares parsing a wave's other rows."""
    offset = int(offset)
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    try:
        f = open(path, "rb")
    except FileNotFoundError:
        return [], offset
    with f:
        if offset > os.fstat(f.fileno()).st_size:
            offset = 0
        f.seek(offset)
        if max_bytes is None:
            data = f.read()
        else:
            data = f.read(max(1, int(max_bytes)))
            if data and not data.endswith(b"\n"):
                data += f.readline()          # finish the line the cap cut through, if it is finished
    end = data.rfind(b"\n") + 1
    rows = []
    for line in data[:end].split(b"\n"):
        if not line.strip() or (contains is not None and contains not in line):
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows, offset + end


def read_rows(path, contains: bytes | None = None) -> list[dict]:
    """Every whole row of a JSONL file (with `contains`, a superset of the rows that hold those bytes; the
    caller still filters on the parsed field). harness.log.read_jsonl would raise on the half-written last
    line of a run in progress, which is exactly the run the GUI is watching."""
    return read_from(path, 0, contains=contains)[0]


def _id_bytes(dyad_id: str) -> bytes:
    """The bytes a JSONL row of this dyad contains, however the writer spaced it: the quoted id. Ids are
    checked against RUN_ID_RE, so they need no JSON escaping."""
    return json.dumps(dyad_id).encode("utf-8")


class _TurnProgress:
    """Per-run progress from turns.jsonl, read incrementally: the run summary is refreshed every few seconds
    while a wave runs, and a wave's turns.jsonl is hundreds of MB, so it is parsed once and then only from
    the last offset. Holds (dyad_id, attempt) -> completed mentor rows and dyad_id -> latest row ts."""
    CHUNK = 16 << 20

    def __init__(self):
        self._lock = threading.Lock()
        self._state: dict[str, dict] = {}

    def get(self, path: Path) -> tuple[dict, dict]:
        key = str(path)
        with self._lock:
            st = self._state.get(key)
            try:
                size = path.stat().st_size
            except OSError:
                self._state.pop(key, None)
                return {}, {}
            if st is None or size < st["offset"]:          # new, or replaced by a shorter file
                st = self._state[key] = {"offset": 0, "done": {}, "last_ts": {}}
            while st["offset"] < size:
                rows, new = read_from(path, st["offset"], max_bytes=self.CHUNK)
                if new == st["offset"]:
                    break                                   # only a half-written line is left
                st["offset"] = new
                for r in rows:
                    d = r.get("dyad_id")
                    if r.get("ts") and r["ts"] > st["last_ts"].get(d, ""):
                        st["last_ts"][d] = r["ts"]
                    if r.get("agent") == "mentor" and r.get("finish_reason") != "error":
                        k = (d, _attempt(r))
                        st["done"][k] = st["done"].get(k, 0) + 1
            return dict(st["done"]), dict(st["last_ts"])


_TURN_PROGRESS = _TurnProgress()


def _read_json(path) -> dict | None:
    """A JSON object from a file, or None when it is missing or not a JSON object."""
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _attempt(row: dict) -> int:
    """A row's attempt number; rows written before attempts existed are attempt 1, as in the harness."""
    try:
        return int(row.get("attempt", 1))
    except (TypeError, ValueError):
        return 1


def _status_rows(rows: list[dict]) -> list[dict]:
    """Status rows resume_index can read (it indexes dyad_id and attempt without a default)."""
    return [r for r in rows if "dyad_id" in r and "attempt" in r and "status" in r]


def status_label(status: str | None) -> str:
    """complete and failed as they are; `started` (and anything else) is a dyad still running."""
    return status if status in ("complete", "failed") else "running"


def status_counts(status_rows: list[dict]) -> dict[str, int]:
    """{complete, failed, running}: dyads by the status of their latest attempt (harness.log.resume_index)."""
    counts = {"complete": 0, "failed": 0, "running": 0}
    for v in resume_index(_status_rows(status_rows)).values():
        counts[status_label(v["status"])] += 1
    return counts


def level_label(value) -> str:
    """A condition value as a level key: JSON null is "(none)" (the control's openness and role)."""
    return NONE_LEVEL if value is None else str(value)


def order_levels(seen: list[str], order=None) -> list[str]:
    """Level labels in `order` (a grid's or study's level list) when given -- labels not in it follow in
    the order they were seen, "(none)" last -- else in the order they were seen."""
    if not order:
        return list(seen)
    out = []
    for label in [level_label(v) for v in order] + list(seen):
        if label in seen and label != NONE_LEVEL and label not in out:
            out.append(label)
    return out + ([NONE_LEVEL] if NONE_LEVEL in seen else [])


def condition_levels(dyad_rows: list[dict], level_order: dict | None = None) -> dict[str, list[str]]:
    """Each condition key seen in dyads.jsonl -> the levels it takes, ordered by `level_order`."""
    seen: dict[str, list[str]] = {}
    for r in dyad_rows:
        cond = r.get("condition")
        if not isinstance(cond, dict):
            continue
        for key, value in cond.items():
            levels = seen.setdefault(key, [])
            if level_label(value) not in levels:
                levels.append(level_label(value))
    return {k: order_levels(v, (level_order or {}).get(k)) for k, v in seen.items()}


def metrics_present(score_rows: list[dict]) -> list[str]:
    """The metrics scores.jsonl holds, in harness.scorer.METRICS order (any others after)."""
    present = []
    for r in score_rows:
        m = r.get("metric")
        if m is not None and m not in present:
            present.append(m)
    return [m for m in METRICS if m in present] + [m for m in present if m not in METRICS]


def judge_shas(score_rows: list[dict]) -> list[str]:
    """The judge model hashes in scores.jsonl, in the order they first scored."""
    out = []
    for r in score_rows:
        j = r.get("judge_sha256")
        if j and j not in out:
            out.append(j)
    return out


def _manifest(path: Path) -> dict | None:
    return _read_json(path / "manifest.json")


def _input_manifest_path(manifest: dict | None, root) -> Path | None:
    """The dyad manifest a run was started from. A relative path is relative to where `harness run` ran,
    which for the sandbox (and the documented CLI) is the repository root."""
    try:
        p = Path((manifest or {}).get("input_manifest", {}).get("path") or "")
    except (AttributeError, TypeError):
        return None
    if not str(p) or str(p) == ".":
        return None
    return p if p.is_absolute() else Path(root or REPO) / p


def _planned(manifest: dict | None, root) -> int | None:
    """Rows in the run's input manifest: how many dyads the run set out to do. None when unreadable."""
    p = _input_manifest_path(manifest, root)
    if p is None:
        return None
    try:
        with open(p, "rb") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return None


def _role_info(block) -> dict | None:
    if not isinstance(block, dict):
        return None
    return {k: block.get(k) for k in ROLE_INFO_KEYS}


def _nonempty(path: Path) -> bool:
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def _mtime(path: Path) -> float:
    """When a run last changed: the newest of its directory and files (appends move a file's mtime, not
    the directory's)."""
    times = [path.stat().st_mtime]
    for p in path.iterdir():
        try:
            times.append(p.stat().st_mtime)
        except OSError:
            continue
    return max(times)


def list_runs(data_dir, root=None) -> list[dict]:
    """Every run in a data directory (a directory with a manifest.json or a status.jsonl), newest first,
    with its progress counts and models. A missing data directory has no runs."""
    base = Path(data_dir).resolve()
    if not base.is_dir():
        return []
    out = []
    for entry in base.iterdir():
        try:
            path = run_dir(base, entry.name)
        except (ValueError, RunNotFound):
            continue
        has_manifest = (path / "manifest.json").is_file()
        if not has_manifest and not (path / "status.jsonl").is_file():
            continue
        manifest = _manifest(path)
        status = read_rows(path / "status.jsonl")
        out.append({"run_id": entry.name, "started_at": (manifest or {}).get("started_at"), "mtime": _mtime(path),
                    "has_manifest": has_manifest, "has_scores": _nonempty(path / "scores.jsonl"),
                    "has_flags": _nonempty(path / "flags.jsonl"),
                    "n_dyads": len(resume_index(_status_rows(status))), "planned": _planned(manifest, root),
                    "status_counts": status_counts(status),
                    **{role: ({k: manifest[role].get(k) for k in ("alias", "model_path", "model_sha256")}
                              if isinstance((manifest or {}).get(role), dict) else None)
                       for role in ("seeker", "mentor")}})
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


def _judges(path: Path) -> list[dict]:
    """The judge-*.json provenance files `score` wrote beside the run, by file name."""
    out = []
    for p in sorted(path.glob("judge-*.json")):
        j = _read_json(p)
        if j is not None:
            out.append({"file": p.name, **{k: j.get(k) for k in JUDGE_INFO_KEYS}})
    return out


def run_summary(data_dir, run_id, root=None) -> dict:
    """One run for the Runs view: its manifest (minus the bulky role fields), models, judges, per-dyad
    progress, the axes and their levels, and what has been scored. A dyad's status is that of its latest
    attempt; turns_done counts that attempt's mentor rows that did not fail."""
    path = run_dir(data_dir, run_id)
    manifest = _manifest(path)
    status = _status_rows(read_rows(path / "status.jsonl"))
    dyad_rows = [r for r in read_rows(path / "dyads.jsonl") if "dyad_id" in r]
    done, turn_ts = _TURN_PROGRESS.get(path / "turns.jsonl")
    scores = read_rows(path / "scores.jsonl")
    index = resume_index(status)

    order: list[str] = []
    attempts: dict[str, set[int]] = {}
    rows: dict[tuple[str, int], dict] = {}
    for r in dyad_rows:
        d = r["dyad_id"]
        if d not in attempts:
            order.append(d)
        attempts.setdefault(d, set()).add(_attempt(r))
        rows[(d, _attempt(r))] = r
    last_ts: dict[str, str] = {}
    reasons: dict[tuple[str, int], str | None] = {}
    for r in status:
        d = r["dyad_id"]
        if d not in attempts:
            order.append(d)
        attempts.setdefault(d, set()).add(_attempt(r))
        reasons[(d, _attempt(r))] = r.get("reason")
        if r.get("ts") and r["ts"] > last_ts.get(d, ""):
            last_ts[d] = r["ts"]
    for d, ts in turn_ts.items():
        if d in attempts and ts > last_ts.get(d, ""):
            last_ts[d] = ts

    dyads = []
    for d in order:
        cur = index.get(d)
        attempt = cur["attempt"] if cur else max(attempts[d])
        label = status_label(cur["status"]) if cur else "running"
        row = rows.get((d, attempt)) or next((rows[(d, a)] for a in sorted(attempts[d], reverse=True)
                                              if (d, a) in rows), {})
        dyads.append({"dyad_id": d, "attempt": attempt, "attempts": len(attempts[d]),
                      "condition": row.get("condition"), "persona_mode": row.get("persona_mode"),
                      "n_turns": row.get("n_turns"), "status": label, "turns_done": done.get((d, attempt), 0),
                      "reason": reasons.get((d, attempt)) if label == "failed" else None,
                      "last_ts": last_ts.get(d)})

    return {"run_id": run_id, "path": str(path), "planned": _planned(manifest, root),
            "status_counts": status_counts(status), "has_scores": _nonempty(path / "scores.jsonl"),
            "has_flags": _nonempty(path / "flags.jsonl"),
            "manifest": {k: manifest.get(k) for k in MANIFEST_KEYS} if manifest is not None else None,
            "roles": {role: _role_info((manifest or {}).get(role)) for role in ("seeker", "mentor")},
            "judges": _judges(path), "dyads": dyads,
            "factors": condition_levels(dyad_rows, level_order(data_dir, run_id, root=root)),
            "metrics": metrics_present(scores), "judge_shas": judge_shas(scores)}


def survey_pairs(rows: list[dict], origin: str = "run") -> list[dict]:
    """One dyad attempt's survey rows as pre/post pairs per item, in first-appearance order. Only rows of
    `origin` count (a row without one is from the run itself); a repeated (phase, item) takes the last row;
    `delta` is post minus pre when both answers are numbers."""
    items: dict[str, dict] = {}
    for r in rows:
        if r.get("origin", "run") != origin or r.get("phase") not in ("pre", "post"):
            continue
        it = items.setdefault(r.get("item_id"), {"item_id": r.get("item_id"), "battery": r.get("battery"),
                                                 "scale": r.get("scale"), "pre": None, "post": None})
        it[r["phase"]] = r.get("answer")
    out = []
    for it in items.values():
        pre, post = it["pre"], it["post"]
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (pre, post))
        out.append({**it, "delta": post - pre if numeric else None})
    return out


def dyad_detail(data_dir, run_id, dyad_id, attempt=None) -> dict:
    """One dyad for the transcript view. `attempt` None is the latest complete attempt (what analysis
    uses), else the highest attempt there is. Turns are in message_order (file order is thread timing);
    `status` is the dyad's whole ledger, every attempt; the rest is the chosen attempt's."""
    path = run_dir(data_dir, run_id)
    check_id(dyad_id, "dyad_id")
    needle = _id_bytes(dyad_id)
    status = [r for r in _status_rows(read_rows(path / "status.jsonl", needle)) if r["dyad_id"] == dyad_id]
    dyad_rows = [r for r in read_rows(path / "dyads.jsonl", needle) if r.get("dyad_id") == dyad_id]
    attempts = sorted({_attempt(r) for r in status + dyad_rows})
    if not attempts:
        raise RunNotFound(f"no dyad {dyad_id!r} in run {run_id!r}")
    if attempt is None:
        attempt = latest_complete_attempts(status).get(dyad_id, attempts[-1])
    else:
        attempt = int(attempt)
        if attempt not in attempts:
            raise RunNotFound(f"dyad {dyad_id!r} has no attempt {attempt} (attempts: {attempts})")

    def mine(rows):
        return [r for r in rows if r.get("dyad_id") == dyad_id and _attempt(r) == attempt]

    surveys = mine(read_rows(path / "surveys.jsonl", needle))
    metric_rank = {m: i for i, m in enumerate(METRICS)}
    scores = sorted(mine(read_rows(path / "scores.jsonl", needle)),
                    key=lambda r: (*message_order(r), metric_rank.get(r.get("metric"), len(metric_rank))))
    return {"run_id": run_id, "dyad_id": dyad_id, "attempt": attempt, "attempts": attempts,
            "dyad": next((r for r in reversed(dyad_rows) if _attempt(r) == attempt), None),
            "status": status, "turns": sorted(mine(read_rows(path / "turns.jsonl", needle)), key=message_order),
            "surveys": {phase: [r for r in surveys if r.get("phase") == phase] for phase in ("pre", "post")},
            "survey_pairs": survey_pairs(surveys), "scores": scores,
            "flag": next(iter(reversed(mine(read_rows(path / "flags.jsonl", needle)))), None)}


def tail(data_dir, run_id, offsets: dict, max_bytes: int | None = TAIL_MAX_BYTES) -> dict:
    """New whole rows of turns, status, surveys and scores since the given byte offsets (missing ones are
    0), and the offsets to pass next time. This is how the GUI follows a run in progress."""
    path = run_dir(data_dir, run_id)
    rows, new = {}, {}
    for name in TAIL_FILES:
        rows[name], new[name] = read_from(path / f"{name}.jsonl", (offsets or {}).get(name) or 0,
                                          max_bytes=max_bytes)
    return {"rows": rows, "offsets": new}


def _dedupe(values) -> list:
    out = []
    for v in values:
        if v not in out:
            out.append(v)
    return out


def _grid_order(manifest: dict, root) -> dict[str, list]:
    """Level order from the grid the run's config names: each factor's levels, then the control's level
    (ideology `none`) after the factor it sets."""
    grid_path = Path((manifest.get("config") or {}).get("grid") or "")
    if not str(grid_path) or str(grid_path) == ".":
        return {}
    grid = load_grid(grid_path if grid_path.is_absolute() else Path(root or REPO) / grid_path)
    order = {k: _dedupe(v) for k, v in grid["factors"].items()}
    for key, level in (grid.get("control") or {}).items():
        if key in order and level is not None and level not in order[key]:
            order[key].append(level)
    return order


def _study_order(manifest: dict, root) -> dict[str, list]:
    """Level order from a sandbox study.json beside the run's input manifest (an export directory): each
    factor's level ids, the control level after its factor, and the nested key's variant ids."""
    p = _input_manifest_path(manifest, root)
    spec = _read_json(p.parent / "study.json") if p is not None else None
    if spec is None:
        return {}
    order = {f["key"]: _dedupe(lv["id"] for lv in f["levels"]) for f in spec.get("factors") or []}
    control = spec.get("control")
    if control:
        levels = order.setdefault(control["factor"], [])
        if control["level"] not in levels:
            levels.append(control["level"])
    nested = spec.get("nested")
    if nested:
        order[nested["key"]] = _dedupe(v["id"] for variants in nested["variants"].values() for v in variants)
    return order


def level_order(data_dir, run_id, root=None) -> dict[str, list]:
    """The order of each axis's levels for this run: from the grid its config names, else from the
    study.json beside its input manifest, else {} (first appearance). Never raises on a missing or
    malformed file -- ordering is cosmetic and must not take a view down."""
    try:
        path = run_dir(data_dir, run_id)
    except RunNotFound:
        return {}
    manifest = _manifest(path) or {}
    for source in (_grid_order, _study_order):
        try:
            order = source(manifest, root)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
        if order:
            return order
    return {}
