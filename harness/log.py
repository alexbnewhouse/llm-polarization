"""Append-only JSONL logging, run manifest, resume index, seeds and hashing."""
from __future__ import annotations
import hashlib, json, threading, time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    root: Path
    manifest: Path
    dyads: Path
    status: Path
    turns: Path
    surveys: Path
    scores: Path


def run_paths(data_dir: str | Path, run_id: str) -> RunPaths:
    root = Path(data_dir) / run_id
    root.mkdir(parents=True, exist_ok=True)
    return RunPaths(root, root / "manifest.json", root / "dyads.jsonl", root / "status.jsonl",
                    root / "turns.jsonl", root / "surveys.jsonl", root / "scores.jsonl")


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def derive_seed(run_seed: int, dyad_seed: int, dyad_id: str, attempt: int, turn: int, agent: str) -> int:
    return int(sha256_text(f"{run_seed}|{dyad_seed}|{dyad_id}|{attempt}|{turn}|{agent}")[:8], 16)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class ManifestMismatch(Exception):
    pass


# The config keys that decide what a run produces. Everything else in the config -- concurrency,
# data_dir -- is operational: dropping concurrency after an OOM and resuming the same run_id must not
# be refused, because the alternative is fragmenting one wave's data across two run ids.
RUN_AFFECTING_CONFIG = ("seeker", "mentor", "judge", "generation", "run_seed", "batteries", "now")


def run_affecting(config: dict | None) -> dict:
    """The part of a config two runs must agree on to be the same run. The full config is still stored."""
    return {k: (config or {}).get(k) for k in RUN_AFFECTING_CONFIG}


def write_manifest(paths: RunPaths, manifest: dict) -> None:
    if paths.manifest.exists():
        existing = json.loads(paths.manifest.read_text(encoding="utf-8"))
        if run_affecting(existing.get("config")) != run_affecting(manifest.get("config")):
            raise ManifestMismatch(f"{paths.manifest} exists with a different config; use a new run_id")
        return
    paths.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def resume_index(status_rows: list[dict]) -> dict[str, dict]:
    idx: dict[str, dict] = {}
    for r in status_rows:
        cur = idx.get(r["dyad_id"])
        if cur is None or r["attempt"] >= cur["attempt"]:
            idx[r["dyad_id"]] = {"attempt": r["attempt"], "status": r["status"]}
    return idx


def next_attempt(index: dict, dyad_id: str) -> int | None:
    cur = index.get(dyad_id)
    if cur is None:
        return 1
    if cur["status"] == "complete":
        return None
    return cur["attempt"] + 1
