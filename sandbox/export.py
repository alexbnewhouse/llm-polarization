"""An export directory: everything `python -m harness.run check|run|score` needs for one study, written where a
terminal can use it too (docs/superpowers/specs/2026-10-07-sandbox-gui-design.md sections 2, 4 and 5).

A repo-shaped study (sandbox/repo_study.py) is randomized by `harness.randomize.build_manifest` and written by
`harness.randomize.write_manifest` themselves, and every file whose content equals the repo's (grid, catalogue,
batteries) is referenced at the repo's path rather than copied, so the manifest and the assignment log carry the
hashes a CLI-made manifest carries; the printed `randomize` command writes the same bytes. Any other study is
compiled by `sandbox.study.compile_manifest`, which follows the same rules. Either way the directory also holds
the config (the study's `run` block plus `batteries` and `grid`), the batteries when they differ from the repo's,
and `study.json`, the spec as exported (sandbox/runs.py reads level order from it).

Path strings in the result, the config and the commands are relative to the repository root when they lie
inside it and absolute otherwise: the CLI, and every job the GUI starts, runs from the root."""
from __future__ import annotations
import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from harness.log import sha256_file
from harness.randomize import build_manifest, write_manifest
from harness.run import load_config
from sandbox import repo_study, runs, study
from sandbox.jobs import build_argv

ENGINE_REPO = "harness.randomize"
ENGINE_STUDY = "sandbox.study"
RUN_ID_PLACEHOLDER = "RUN_ID"          # the printed run/score commands; the GUI asks for the run id itself
_URL_RE = re.compile(r"^https?://\S+$")


def display_path(path, root) -> str:
    """`path` as the CLI run from `root` should see it: root-relative (POSIX separators) when it lies inside
    the root, absolute otherwise. A relative `path` is taken as relative to the root."""
    root = Path(root).resolve()
    p = Path(path)
    p = (p if p.is_absolute() else root / p).resolve()
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return str(p)


def _dump(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def _write_json(path: Path, obj) -> Path:
    path.write_text(_dump(obj), encoding="utf-8")
    return path


def _source(spec, key: str, root: Path) -> Path | None:
    """The repo file the spec was built from (spec.repo.sources[key]), if it exists inside the root. A source
    outside the root is never referenced: the spec arrives from a browser, and only the repo's own files are
    worth pointing at."""
    repo = spec.get("repo")
    sources = repo.get("sources") if isinstance(repo, dict) else None
    s = sources.get(key) if isinstance(sources, dict) else None
    if not isinstance(s, str) or not s or "\0" in s:
        return None
    p = Path(s)
    p = (p if p.is_absolute() else root / p).resolve()
    if root not in p.parents or not p.is_file():
        return None
    return p


def _reuse_or_write(obj, source: Path | None, target: Path) -> tuple[Path, bool]:
    """The repo's file when its content equals `obj` (True), else `obj` written to `target` (False)."""
    if source is not None and repo_study.same_json(obj, source):
        return source, True
    return _write_json(target, obj), False


def _slug(name) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48].strip("-") if isinstance(name, str) else ""
    return s or "study"


def default_export_dir(workspace, spec) -> Path:
    """A new, empty directory workspace/exports/<slug of the study name>-<YYYYmmdd-HHMMSS>, with -2, -3 ...
    appended when that exists. Created here (mkdir, not exists-then-create), so two exports in the same second
    never share a directory."""
    base = Path(workspace) / "exports"
    base.mkdir(parents=True, exist_ok=True)
    stem = f"{_slug(spec.get('name') if isinstance(spec, dict) else None)}-{time.strftime('%Y%m%d-%H%M%S')}"
    n = 1
    while True:
        path = base / (stem if n == 1 else f"{stem}-{n}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            n += 1


def _subset(spec, rows: list[dict], assignment: dict, cells: list) -> tuple[list[dict], dict]:
    """randomization.cells applied to harness.randomize's rows after the shuffle (spec 3.2): the kept rows keep
    their seeds and order, and the counts are taken over them, keys in the full assignment's order."""
    keep = set(cells)
    keys: dict[str, str | None] = {}

    def key(row):
        c = json.dumps(row.get("condition"), sort_keys=True)
        if c not in keys:
            keys[c] = study.cell_key(spec, row.get("condition"))
        return keys[c]

    kept = [row for row in rows if key(row) in keep]
    per_cell: dict[str, int] = {}
    per_variant: dict[str, int] = {}
    per_mode: dict[str, int] = {}
    for row in kept:
        k = key(row)
        per_cell[k] = per_cell.get(k, 0) + 1
        role = (row.get("condition") or {}).get("role")
        if role is not None:
            per_variant[f"{k}/{role}"] = per_variant.get(f"{k}/{role}", 0) + 1
        per_mode[row["persona_mode"]] = per_mode.get(row["persona_mode"], 0) + 1
    a = dict(assignment)
    a["rows"] = len(kept)
    a["rows_per_cell"] = {k: per_cell[k] for k in assignment["rows_per_cell"] if k in per_cell}
    a["rows_per_variant"] = {k: per_variant[k] for k in assignment["rows_per_variant"] if k in per_variant}
    a["rows_per_mode"] = {m: per_mode[m] for m in assignment["rows_per_mode"] if m in per_mode}
    a["cells_filter"] = list(cells)
    return kept, a


def export_study(spec, *, root, workspace, out_dir=None, python: str = "python") -> dict:
    """Write the export directory for a valid study and return the export result of spec section 5: the
    paths written or reused, the engine, which repo files were reused unchanged, the row count, the cells
    filter, notes, and the CLI argv of each step (`python` names the interpreter in them; the GUI prints them as
    shell lines run from the root). Raises study.StudyError (with the issues) when validation finds errors,
    before anything is written. `out_dir` None is default_export_dir(workspace, spec)."""
    issues = study.validate_study(spec)
    if study.has_errors(issues):
        raise study.StudyError(issues)
    root = Path(root).resolve()
    created = out_dir is None
    out = default_export_dir(workspace, spec) if created else Path(out_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
        return _export(spec, root, out.resolve(), python)
    except BaseException:
        if created:                     # a half-written export is worse than none; a given out_dir is the caller's
            shutil.rmtree(out, ignore_errors=True)
        raise


def _export(spec: dict, root: Path, out: Path, python: str) -> dict:
    def show(p):
        return display_path(p, root)

    r = spec["randomization"]
    cells = r.get("cells")
    manifest = out / f"{r['prefix']}-dyads.jsonl"
    notes: list[str] = []

    batteries, same_batteries = _reuse_or_write(spec["instrument"], _source(spec, "batteries", root),
                                                out / "batteries.json")
    randomize_cmd = None
    if repo_study.repo_shape_reason(spec) is None:
        engine = ENGINE_REPO
        grid, catalogue = repo_study.to_repo_files(spec)
        grid_path, same_grid = _reuse_or_write(grid, _source(spec, "grid", root), out / "grid.json")
        catalogue_path, same_catalogue = _reuse_or_write(catalogue, _source(spec, "catalogue", root),
                                                         out / "catalogue.json")
        if not same_catalogue:
            # The edited catalogue still carries the repo's version string (the spec keeps it in repo.catalogue),
            # so the assignment log's version alone would read as the repo's catalogue.
            notes.append(f"the catalogue differs from the repo's but keeps version {catalogue.get('version')}; "
                         "the assignment log's catalogue sha256 tells them apart")
        kwargs = repo_study.build_manifest_kwargs(spec)
        rows, assignment = build_manifest(grid, catalogue, **kwargs)
        if cells is not None:
            rows, assignment = _subset(spec, rows, assignment, cells)
        # write_manifest hashes grid and catalogue by the paths it is given, relative to the process's CWD,
        # which for the GUI server is not necessarily the root: hand it absolute paths, then record the paths
        # the printed command uses, so the log reads as the CLI run from the root would write it.
        sidecar = write_manifest(manifest, rows, assignment, grid_path=grid_path, catalogue_path=catalogue_path)
        a = json.loads(sidecar.read_text(encoding="utf-8"))
        a["grid"]["path"], a["catalogue"]["path"] = show(grid_path), show(catalogue_path)
        a["output"]["path"] = show(manifest)
        sidecar.write_text(json.dumps(a, indent=2, ensure_ascii=False), encoding="utf-8")   # write_manifest's format
        if cells is None:
            randomize_cmd = [python, "-m", "harness.randomize", "--grid", show(grid_path),
                             "--catalogue", show(catalogue_path), "--out", show(manifest),
                             "--seed", str(kwargs["seed"]), "--n-per-cell", str(kwargs["n_per_cell"]),
                             "--n-control", str(kwargs["n_control"]), "--modes", ",".join(kwargs["modes"]),
                             "--n-turns", str(kwargs["n_turns"]), "--prefix", kwargs["prefix"]]
        else:
            notes.append(f"a subset of {len(cells)} cell(s): harness.randomize built the full design and the "
                         "export keeps those cells' rows, so there is no single randomize command for it")
    else:
        engine = ENGINE_STUDY
        grid_path = catalogue_path = same_grid = same_catalogue = None
        rows, assignment = study.compile_manifest(spec)
        with open(manifest, "w", encoding="utf-8") as f:              # as harness.randomize.write_manifest does
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        sidecar = out / f"{r['prefix']}-assignment.json"
        assignment["output"] = {"path": show(manifest), "sha256": sha256_file(manifest)}
        sidecar.write_text(json.dumps(assignment, indent=2, ensure_ascii=False), encoding="utf-8")

    study_path = _write_json(out / "study.json", spec)
    config = {**spec["run"], "batteries": show(batteries), "grid": show(grid_path) if grid_path else None}
    config_path = _write_json(out / "config.json", config)
    try:
        load_config(config_path)
    except (ValueError, OSError, TypeError, AttributeError) as e:
        notes.append(f"config.json: {e}; `check` and `run` refuse it until that is set (Models view)")

    config_s, manifest_s = show(config_path), show(manifest)
    return {
        "dir": show(out), "manifest": manifest_s, "assignment": show(sidecar), "config": config_s,
        "batteries": show(batteries), "grid": show(grid_path) if grid_path else None,
        "catalogue": show(catalogue_path) if catalogue_path else None, "study": show(study_path),
        "engine": engine,
        "unchanged_from_repo": {"grid": same_grid, "catalogue": same_catalogue, "batteries": same_batteries},
        "rows": len(rows), "cells_filter": list(cells) if cells is not None else None, "notes": notes,
        "commands": {
            "randomize": randomize_cmd,
            "check": build_argv("check", {"config": config_s, "manifest": manifest_s}, python=python),
            "run": build_argv("run", {"config": config_s, "manifest": manifest_s, "run_id": RUN_ID_PLACEHOLDER},
                              python=python),
            "score": build_argv("score", {"config": config_s, "run_id": RUN_ID_PLACEHOLDER, "scope": "pilot"},
                                python=python),
        },
    }


def _judge(judge) -> dict:
    """A judge override {url, gguf_path}: url an http(s) URL, gguf_path a path or empty/null (read from the
    server's /props then)."""
    if not isinstance(judge, dict):
        raise ValueError(f"judge must be an object {{url, gguf_path}}, got {type(judge).__name__}")
    unknown = sorted(str(k) for k in judge if k not in ("url", "gguf_path"))
    if unknown:
        raise ValueError(f"judge takes url and gguf_path, not {', '.join(unknown)}")
    url = judge.get("url")
    if not isinstance(url, str) or not _URL_RE.fullmatch(url.strip()):
        raise ValueError(f"judge.url must be an http(s) URL, got {url!r}")
    gguf = judge.get("gguf_path")
    if isinstance(gguf, str) and not gguf.strip():
        gguf = None
    if gguf is not None and (not isinstance(gguf, str) or "\0" in gguf):
        raise ValueError(f"judge.gguf_path must be a path or empty, got {gguf!r}")
    return {"url": url.strip(), "gguf_path": gguf.strip() if gguf else None}


def config_from_manifest(data_dir, run_id, workspace, judge=None, *, root=None) -> Path:
    """A config for `score`, `flags`, `survey` and `agreement` on an existing run: the config its manifest.json
    records, with data_dir set to where the run was found (data_dir is operational, not run-affecting, in the
    harness; root-relative when `root` is given and it lies inside), and, with `judge` {url, gguf_path}, that
    judge instead of the recorded one (a second judge for the two-judge design). Written to
    workspace/configs/<run_id>.json, or <run_id>-judge-<sha256 of the judge JSON, 8 hex>.json; returns the path.
    RunNotFound when the run or its manifest.json is missing, ValueError for a bad id, judge or manifest."""
    path = runs.run_dir(data_dir, run_id)
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise runs.RunNotFound(f"run {run_id!r} has no manifest.json: `harness run` has not started it")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise ValueError(f"{manifest_path} is not valid JSON: {e}") from None
    config = manifest.get("config") if isinstance(manifest, dict) else None
    if not isinstance(config, dict):
        raise ValueError(f"{manifest_path} has no config object")
    config = json.loads(json.dumps(config))
    config["data_dir"] = display_path(data_dir, root) if root is not None else str(Path(data_dir).resolve())
    name = run_id
    if judge is not None:
        j = _judge(judge)
        config["judge"] = j
        name = f"{run_id}-judge-{hashlib.sha256(json.dumps(j, sort_keys=True).encode('utf-8')).hexdigest()[:8]}"
    out = Path(workspace) / "configs" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    return _write_json(out, config)
