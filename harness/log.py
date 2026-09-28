"""Append-only JSONL logging, run manifest, resume index, seeds and hashing."""
from __future__ import annotations
import contextlib, fcntl, hashlib, json, os, re, shutil, threading, time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    """Where one run's files live. Every path a run writes is named here and nowhere else."""
    root: Path
    manifest: Path
    dyads: Path
    status: Path
    turns: Path
    surveys: Path
    scores: Path
    flags: Path
    input_dyads: Path       # a verbatim copy of the input dyad manifest, written when the run starts
    assignment: Path        # a copy of the randomizer's assignment log for that manifest, when there is one
    baseline: Path          # `baseline` rows: the pre battery administered K times, no dialogue


# What a run_id or a dyad_id may be: it names a directory, and a dyad_id goes into every derived seed with
# `|` as the separator, so no path separators, no `..`, no `|`, and not empty.
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def check_id(value, what: str) -> str:
    """Return `value` if it is a string made of letters, digits, `.`, `_` and `-` that starts with a letter
    or digit; raise ValueError naming `what` otherwise."""
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise ValueError(f"{what} must be a non-empty string of letters, digits, '.', '_' and '-' starting "
                         f"with a letter or digit, got {value!r}")
    return value


def run_paths(data_dir: str | Path, run_id: str, create: bool = True) -> RunPaths:
    """The paths for one run_id, creating data/<run_id>/ if it does not exist yet and `create` is true.
    Commands that only read a run pass create=False, so a mistyped run_id does not leave a directory."""
    root = Path(data_dir) / check_id(run_id, "run_id")
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return RunPaths(root, root / "manifest.json", root / "dyads.jsonl", root / "status.jsonl",
                    root / "turns.jsonl", root / "surveys.jsonl", root / "scores.jsonl", root / "flags.jsonl",
                    root / "input-dyads.jsonl", root / "assignment.json", root / "baseline.jsonl")


class TornLine(ValueError):
    """A JSONL file ends in a line with no newline: a write cut off by a crash, a kill or a full disk."""


def _torn_message(path: Path, lineno: int) -> str:
    return (f"{path} line {lineno} has no newline at the end of the file: a row cut off by a crash, a kill "
            "or a full disk. Re-run `run`, `survey` or `score` with --repair-torn-line to back the file up "
            "and drop that one line (or finish it, if it is a whole row)")


def _ends_torn(path: Path) -> bool:
    """True when the file exists, is not empty, and its last byte is not a newline."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                return False
            f.seek(-1, os.SEEK_END)
            return f.read(1) != b"\n"
    except FileNotFoundError:
        return False


def _line_count(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


class JsonlWriter:
    """Append-only writer for one JSONL file, safe to share across the run's worker threads."""

    def __init__(self, path: Path):
        """Take the path and create a lock private to this instance, serialising writes to this one file
        only -- a second JsonlWriter on a different path has its own lock and is unaffected. The file
        itself is opened per write, and only when there is a line to add."""
        self.path = Path(path)
        self._lock = threading.Lock()
        self._tail_checked = False

    def write(self, obj: dict) -> None:
        """Append one row, flush it and fsync it, so a row that is written is on disk before the next one
        (a 'complete' status row never outlives the turns it vouches for). Refuses to append to a file
        whose last line is torn: the new row would be glued onto the fragment and lost with it."""
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        with self._lock:
            if not self._tail_checked:
                if _ends_torn(self.path):
                    raise TornLine(_torn_message(self.path, _line_count(self.path)))
                self._tail_checked = True
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict]:
    """Every row of a JSONL file in file order; an empty list when the file does not exist yet. A line that
    does not parse raises ValueError naming the file and the line; a cut-off last line raises TornLine."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as e:
                if not line.endswith("\n"):
                    raise TornLine(_torn_message(path, n)) from None
                raise ValueError(f"{path} line {n} is not valid JSON: {e}") from None
    return rows


def check_tails(root: Path) -> None:
    """Raise TornLine for the first *.jsonl in a run directory whose last line is cut off."""
    for path in sorted(Path(root).glob("*.jsonl")):
        if _ends_torn(path):
            raise TornLine(_torn_message(path, _line_count(path)))


def repair_torn_lines(root: Path) -> list[str]:
    """For each *.jsonl in a run directory whose last line has no newline: copy the file to
    <name>.torn-<time>, then drop that last line, or add the newline when the line is a whole row. Only the
    last line is ever touched. Returns one message per file changed."""
    out = []
    for path in sorted(Path(root).glob("*.jsonl")):
        if not _ends_torn(path):
            continue
        backup = path.with_name(f"{path.name}.torn-{time.strftime('%Y%m%dT%H%M%S')}")
        shutil.copy2(path, backup)
        data = path.read_bytes()
        cut = data.rfind(b"\n") + 1
        tail = data[cut:]
        try:
            json.loads(tail.decode("utf-8"))
            with open(path, "ab") as f:
                f.write(b"\n"); f.flush(); os.fsync(f.fileno())
            out.append(f"{path}: the last line was a whole row without its newline; added it "
                       f"(backup {backup.name})")
        except ValueError:
            with open(path, "r+b") as f:
                f.truncate(cut); f.flush(); os.fsync(f.fileno())
            out.append(f"{path}: dropped a cut-off last line of {len(tail)} bytes (backup {backup.name})")
    return out


def sha256_text(s: str) -> str:
    """SHA-256 of a string, UTF-8 encoded. This is what pins every prompt in the study."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes, read in 1 MB chunks so a 20 GB GGUF does not go through memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def derive_seed(run_seed: int, dyad_seed: int, dyad_id: str, attempt: int, turn: int, agent: str) -> int:
    """The seed for one generation. Derived, not random: the same
    (run_seed, dyad_seed, dyad_id, attempt, turn, agent) always gives the same seed, so a run is
    re-runnable from its run_seed and its dyad manifest alone. Takes the first 8 hex digits of the
    SHA-256 (32 bits), which is the range llama-server accepts."""
    return int(sha256_text(f"{run_seed}|{dyad_seed}|{dyad_id}|{attempt}|{turn}|{agent}")[:8], 16)


def now_iso() -> str:
    """The timestamp written on every row: local time with a UTC offset, so the box's clock is readable
    as it stood. Every run in this study is on one machine, in one timezone."""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class ManifestMismatch(Exception):
    """Raised when a run's recorded identity and the live one disagree -- the config that produced the
    rows, or the model and template now being served. Always fatal, never patched over."""


# The config keys that decide what a run produces. Everything else in the config -- concurrency,
# data_dir, gguf_py_path, cache_reuse_limit, grid -- is operational: dropping concurrency after an OOM
# and resuming the same run_id must not be refused, because the alternative is fragmenting one wave's
# data across two run ids. `batteries` is compared as a path here; `run` and `survey` also compare its
# content hash (harness/run.py resume_changes).
RUN_AFFECTING_CONFIG = ("seeker", "mentor", "judge", "generation", "run_seed", "batteries", "now")


def run_affecting(config: dict | None) -> dict:
    """The part of a config two runs must agree on to be the same run. The full config is still stored."""
    return {k: (config or {}).get(k) for k in RUN_AFFECTING_CONFIG}


def write_manifest(paths: RunPaths, manifest: dict) -> None:
    """Write the run manifest once, or verify that an existing one describes the same run. This is what
    makes a resume safe and an accidental overwrite impossible: an existing manifest is never rewritten.
    Assumes one `run` process per run_id (the check and the write are not atomic across processes)."""
    if paths.manifest.exists():
        existing = json.loads(paths.manifest.read_text(encoding="utf-8"))
        if run_affecting(existing.get("config")) != run_affecting(manifest.get("config")):
            raise ManifestMismatch(f"{paths.manifest} exists with a different config; use a new run_id")
        return
    paths.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


class RunLocked(Exception):
    """Another harness process holds this run's lock file."""


@contextlib.contextmanager
def run_lock(paths: RunPaths, command: str):
    """Hold an exclusive lock on data/<run_id>/.lock for the life of one command, or raise RunLocked. Two
    processes appending to one run would run every dyad twice on the same slots. The lock is an
    fcntl.flock, so the kernel releases it when the process dies, however it dies."""
    path = paths.root / ".lock"
    f = open(path, "a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.seek(0)
            holder = f.read().strip() or "another process"
            raise RunLocked(f"{path} is held by {holder}; one harness process per run_id at a time") from None
        f.seek(0); f.truncate()
        f.write(f"pid {os.getpid()} ({command}, since {now_iso()})"); f.flush()
        yield
    finally:
        f.close()


def resume_index(status_rows: list[dict]) -> dict[str, dict]:
    """Latest status per dyad, read from status.jsonl. A higher attempt wins; within one attempt a later
    row wins, so a 'complete' written after a 'started' is what counts -- except that nothing after a
    'complete' undoes it: a stray 'failed' for an attempt that completed does not re-queue it."""
    idx: dict[str, dict] = {}
    for r in status_rows:
        cur = idx.get(r["dyad_id"])
        if cur is None or r["attempt"] > cur["attempt"] or (r["attempt"] == cur["attempt"]
                                                           and cur["status"] != "complete"):
            idx[r["dyad_id"]] = {"attempt": r["attempt"], "status": r["status"]}
    return idx


def next_attempt(index: dict, dyad_id: str) -> int | None:
    """Which attempt number to run next for this dyad: 1 if never seen, None if its latest attempt
    completed (nothing to do), otherwise one past the latest. A retry is a NEW attempt: it draws new
    seeds and the earlier attempt's rows are kept."""
    cur = index.get(dyad_id)
    if cur is None:
        return 1
    if cur["status"] == "complete":
        return None
    return cur["attempt"] + 1
