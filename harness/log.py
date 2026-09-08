"""Append-only JSONL logging, run manifest, resume index, seeds and hashing."""
from __future__ import annotations
import hashlib, json, threading, time
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


def run_paths(data_dir: str | Path, run_id: str) -> RunPaths:
    """The paths for one run_id, creating data/<run_id>/ if it does not exist yet."""
    root = Path(data_dir) / run_id
    root.mkdir(parents=True, exist_ok=True)
    return RunPaths(root, root / "manifest.json", root / "dyads.jsonl", root / "status.jsonl",
                    root / "turns.jsonl", root / "surveys.jsonl", root / "scores.jsonl")


class JsonlWriter:
    """Append-only writer for one JSONL file, safe to share across the run's worker threads."""

    def __init__(self, path: Path):
        """Take the path and the lock that serialises this process's writers; the file itself is opened
        per write, and only when there is a line to add."""
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, obj: dict) -> None:
        """Append one row and flush it. A crashed run keeps every row written before the crash."""
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()


def read_jsonl(path: Path) -> list[dict]:
    """Every row of a JSONL file in file order; an empty list when the file does not exist yet."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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
# data_dir -- is operational: dropping concurrency after an OOM and resuming the same run_id must not
# be refused, because the alternative is fragmenting one wave's data across two run ids.
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


def resume_index(status_rows: list[dict]) -> dict[str, dict]:
    """Latest status per dyad, read from status.jsonl. Later rows win ties (`>=`), so a 'complete'
    written after a 'started' for the same attempt is what counts."""
    idx: dict[str, dict] = {}
    for r in status_rows:
        cur = idx.get(r["dyad_id"])
        if cur is None or r["attempt"] >= cur["attempt"]:
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
