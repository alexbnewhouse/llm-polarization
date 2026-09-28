"""The study lock: one study.json that every arm's `run`, `baseline` and `score` must match, so the three arms
share run_seed, now, the wave manifest, the instrument, the grid and the judge (gap audit F9;
docs/pap/pre-analysis-plan.md sections 11 and 12). The path is the config's `study`; null turns it off.

`python -m harness.run study --config config.json --manifest w1-dyads.jsonl` writes it once, from the config,
the manifest and the live judge. Afterwards the same command, `check`, `run`, `baseline` and `score` compare
against it and refuse a difference. A descoped manifest from `harness.randomize --subset-of` counts as the
study's manifest when its assignment log names the study's manifest as its parent."""
from __future__ import annotations
import json
from pathlib import Path
from harness.log import now_iso, sha256_file

STUDY_VERSION = 1
# What study.json pins. `input_manifest`, `batteries` and `grid` are compared by sha256, `judge` by model
# sha256 and family, the other two by value.
STUDY_KEYS = ("run_seed", "now", "input_manifest", "batteries", "grid", "judge")


class StudyMismatch(Exception):
    """This run, baseline or scoring pass differs from study.json, or study.json is missing."""


def _file(path) -> dict | None:
    """{path, sha256} of a file, or None for no path."""
    if not path:
        return None
    p = Path(path)
    return {"path": str(p), "sha256": sha256_file(p)}


def subset_parent(manifest_path) -> str | None:
    """The parent manifest's sha256 when `manifest_path` is a subset written by `harness.randomize
    --subset-of` and still hashes to what its assignment log records; None otherwise."""
    from harness.randomize import assignment_log_path
    log_path = assignment_log_path(manifest_path)
    if not log_path.exists():
        return None
    try:
        a = json.loads(log_path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    parent, output = a.get("parent") or {}, a.get("output") or {}
    if a.get("kind") != "subset" or output.get("sha256") != sha256_file(Path(manifest_path)):
        return None
    return parent.get("sha256")


def input_values(manifest_path) -> dict:
    """{path, sha256, parent_sha256} of an input dyad manifest; parent_sha256 is null unless a subset."""
    return {**_file(manifest_path), "parent_sha256": subset_parent(manifest_path)}


def judge_values(entry: dict) -> dict:
    """The judge fields study.json pins, from a manifest entry (harness.run.build_agent)."""
    return {"model_sha256": entry.get("model_sha256"), "family": entry.get("family"),
            "model_path": entry.get("model_path")}


def live_values(cfg: dict, manifest_path=None, judge_entry: dict | None = None) -> dict:
    """The values this config (and manifest, and judge) would lock or be compared on. Keys that cannot be
    known here (no manifest, no judge) are left out, and so are not compared."""
    out = {"run_seed": cfg.get("run_seed"), "now": cfg.get("now"), "batteries": _file(cfg.get("batteries")),
           "grid": _file(cfg.get("grid"))}
    if manifest_path:
        out["input_manifest"] = input_values(manifest_path)
    if judge_entry:
        out["judge"] = judge_values(judge_entry)
    return out


def run_values(manifest: dict) -> dict:
    """The values a run's manifest.json recorded when it started, for `score` to compare. A key the
    manifest does not have (a run started before the lock existed) is left out."""
    cfg = manifest.get("config") or {}
    out = {"run_seed": cfg.get("run_seed"), "now": cfg.get("now")}
    b = manifest.get("batteries") or {}
    if b.get("sha256"):
        out["batteries"] = {"path": b.get("path"), "sha256": b["sha256"]}
    if "grid" in manifest:
        out["grid"] = manifest["grid"]
    im = manifest.get("input_manifest") or {}
    if im.get("sha256"):
        out["input_manifest"] = {"path": im.get("path"), "sha256": im["sha256"],
                                 "parent_sha256": im.get("parent_sha256")}
    return out


def load(path) -> dict:
    """Read study.json, or raise StudyMismatch naming how to create it."""
    p = Path(path)
    if not p.exists():
        raise StudyMismatch(f"study {p} does not exist; create it once with `python -m harness.run study "
                            "--config <config> --manifest <wave manifest>`, or set `study` to null")
    return json.loads(p.read_text(encoding="utf-8"))


def _sha(v) -> str | None:
    return (v or {}).get("sha256")


def changes(study: dict, live: dict) -> list[str]:
    """One line per value in `live` that differs from study.json; empty when they agree."""
    out = []
    for key in ("run_seed", "now"):
        if key in live and live[key] != study.get(key):
            out.append(f"{key} is {live[key]!r}, study.json has {study.get(key)!r}")
    for key in ("batteries", "grid"):
        if key in live and _sha(live[key]) != _sha(study.get(key)):
            now = _sha(live[key])
            out.append(f"{key} {'is null' if live[key] is None else 'hashes to ' + now[:12]}, study.json has "
                       + (str(_sha(study.get(key)))[:12] if study.get(key) else "null"))
    if "input_manifest" in live:
        want, im = _sha(study.get("input_manifest")), live["input_manifest"] or {}
        if want not in (im.get("sha256"), im.get("parent_sha256")):
            out.append(f"input manifest {im.get('path')} hashes to {str(im.get('sha256'))[:12]} and is not a "
                       f"subset of the study's ({str(want)[:12]}); descope with "
                       "`harness.randomize --subset-of`")
    if "judge" in live:
        was, now = study.get("judge") or {}, live["judge"] or {}
        for k in ("model_sha256", "family"):
            if now.get(k) != was.get(k):
                v, w = now.get(k), was.get(k)
                out.append(f"judge.{k} is {str(v)[:12] if k == 'model_sha256' else repr(v)}, study.json has "
                           f"{str(w)[:12] if k == 'model_sha256' else repr(w)}")
    return out


def check(path, live: dict) -> None:
    """Raise StudyMismatch unless study.json at `path` agrees with `live`."""
    diff = changes(load(path), live)
    if diff:
        raise StudyMismatch(f"study {path}: " + "; ".join(diff))


def write(path, values: dict, harness_commit: str) -> dict:
    """Write study.json once. Every key in STUDY_KEYS must be known; the judge's family too, since the
    study rule refuses a judge from any arm's family and unknown is not different."""
    missing = [k for k in STUDY_KEYS if values.get(k) is None and k != "grid"]
    if missing:
        raise StudyMismatch(f"cannot lock the study without {', '.join(missing)}")
    if not values["judge"].get("family"):
        raise StudyMismatch("the judge's model family is unknown; set judge.family in the config")
    p = Path(path)
    if p.exists():
        raise StudyMismatch(f"{p} exists; it is written once")
    record = {"study_version": STUDY_VERSION, "created_at": now_iso(), "harness_commit": harness_commit,
              **{k: values.get(k) for k in STUDY_KEYS}}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return record
