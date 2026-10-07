"""Descriptive views of a run for the sandbox, by the level of any condition key: the mentor's survey shift
(post minus pre) per item and per battery, and a judge's score over turns.

Every view uses the latest complete attempt of each dyad (harness.scorer.latest_complete_attempts), so
the rows of a failed attempt that was retried never count. A level that is JSON null is "(none)"; with no
factor there is one level, "all". Standard errors are the sample SD over sqrt(n), null below n = 2. These
are what the GUI shows while a study is being tried out; the paper's estimands live in analysis/."""
from __future__ import annotations
import math
import statistics
from harness.scorer import METRICS, latest_complete_attempts
from sandbox import runs
from sandbox.runs import ALL_LEVEL, level_label, order_levels


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _se(values: list[float]) -> float | None:
    """Sample SD over sqrt(n); None below two values, where there is no spread to speak of."""
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) >= 2 else None


def _attempt(row: dict) -> int:
    try:
        return int(row.get("attempt", 1))
    except (TypeError, ValueError):
        return 1


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _levels_of(dyads: list[dict], status: list[dict], factor: str | None) -> dict[tuple[str, int], str]:
    """{(dyad_id, attempt): level} for each dyad's latest complete attempt, in ledger order. A dyad without a
    dyads.jsonl row for that attempt has no condition, so it lands in "(none)"."""
    chosen = latest_complete_attempts([r for r in status if "dyad_id" in r and "attempt" in r and "status" in r])
    conditions = {(r.get("dyad_id"), _attempt(r)): r.get("condition") for r in dyads}
    out = {}
    for dyad_id, attempt in chosen.items():
        cond = conditions.get((dyad_id, attempt))
        out[(dyad_id, attempt)] = (ALL_LEVEL if factor is None
                                   else level_label(cond.get(factor) if isinstance(cond, dict) else None))
    return out


def _ordered_levels(level_of: dict, factor: str | None, level_order: dict | None, keep=None) -> list[str]:
    """The levels in level_of (only those in `keep`, when given) in first-appearance order, reordered by
    level_order[factor] when there is one."""
    seen = []
    for level in level_of.values():
        if level not in seen and (keep is None or level in keep):
            seen.append(level)
    return order_levels(seen, (level_order or {}).get(factor) if factor is not None else None)


def survey_shifts(dyads, status, surveys, factor=None, level_order=None, origin="run") -> dict:
    """The mentor's survey shift by level of `factor`. Per item: n (dyads with both answers), mean_pre and
    mean_post over those dyads, mean_delta and se_delta. Per battery: each dyad's mean of
    delta / (scale.max - scale.min) over the battery's items, then the mean and SE across dyads, so a 0-10
    thermometer and a 1-5 agreement item are on one footing (zero-range items are left out). Rows of
    `origin` only ("run": the pass the dialogue run made; a row without an origin is one)."""
    level_of = _levels_of(dyads, status, factor)
    by_dyad: dict[tuple[str, int], list[dict]] = {}
    items: dict[str, dict] = {}
    batteries: list[str] = []
    for r in surveys:
        key = (r.get("dyad_id"), _attempt(r))
        if key not in level_of or r.get("origin", "run") != origin or r.get("phase") not in ("pre", "post"):
            continue
        by_dyad.setdefault(key, []).append(r)
        if r.get("item_id") not in items:
            items[r.get("item_id")] = {"item_id": r.get("item_id"), "battery": r.get("battery"),
                                       "scale": r.get("scale")}
        if r.get("battery") not in batteries:
            batteries.append(r.get("battery"))
    levels = _ordered_levels(level_of, factor, level_order)
    pairs = {key: runs.survey_pairs(rows, origin=origin) for key, rows in by_dyad.items()}

    n_dyads = {level: 0 for level in levels}
    by_item = {level: {i: {"pre": [], "post": [], "delta": []} for i in items} for level in levels}
    norm = {level: {b: [] for b in batteries} for level in levels}
    for key, level in level_of.items():
        n_dyads[level] += 1
        per_battery: dict[str, list[float]] = {}
        for p in pairs.get(key, []):
            if p["delta"] is None:
                continue
            acc = by_item[level][p["item_id"]]
            acc["pre"].append(p["pre"]); acc["post"].append(p["post"]); acc["delta"].append(p["delta"])
            scale = p["scale"] if isinstance(p["scale"], dict) else {}
            lo, hi = scale.get("min"), scale.get("max")
            if _number(lo) and _number(hi) and hi != lo:
                per_battery.setdefault(p["battery"], []).append(p["delta"] / (hi - lo))
        for battery, values in per_battery.items():
            norm[level][battery].append(statistics.fmean(values))

    return {"factor": factor, "levels": levels, "n_dyads": n_dyads, "items": list(items.values()),
            "batteries": batteries,
            "by_item": {level: {i: {"n": len(a["delta"]), "mean_pre": _mean(a["pre"]), "mean_post": _mean(a["post"]),
                                    "mean_delta": _mean(a["delta"]), "se_delta": _se(a["delta"])}
                                for i, a in by_item[level].items()} for level in levels},
            "by_battery": {level: {b: {"n": len(v), "mean_delta_norm": _mean(v), "se_delta_norm": _se(v)}
                                   for b, v in norm[level].items()} for level in levels}}


def score_curves(dyads, status, scores, factor=None, metric="alignment", judge=None, level_order=None) -> dict:
    """One judge's `metric` score over turns by level of `factor`: per level and turn, n, mean and SE of
    the non-null scores. When two judges have scored the metric, `judge` (a sha256 prefix matching exactly
    one of them) is required; without it the block comes back with `error` set and no series, because
    pooling two judges' scores would average two instruments."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {tuple(METRICS)}, got {metric!r}")
    level_of = _levels_of(dyads, status, factor)
    rows = [r for r in scores if r.get("metric") == metric and not r.get("error")
            and (r.get("dyad_id"), _attempt(r)) in level_of]
    judges = []
    for r in rows:
        if r.get("judge_sha256") and r["judge_sha256"] not in judges:
            judges.append(r["judge_sha256"])
    block = {"metric": metric, "judge": None, "judges": judges, "factor": factor, "levels": [], "error": None,
             "series": {}}
    if judge:
        matches = [j for j in judges if j.startswith(judge)]
        if len(matches) != 1:
            block["error"] = f"judge {judge!r} matches {len(matches)} of the {len(judges)} judges that scored {metric}"
            return block
        chosen = matches[0]
    elif len(judges) > 1:
        block["error"] = f"{len(judges)} judges scored {metric}; choose one"
        return block
    else:
        chosen = judges[0] if judges else None
    block["judge"] = chosen

    # One score per (dyad, attempt, turn, agent): a repeated row takes the last, as everywhere else.
    latest: dict[tuple, tuple[str, float]] = {}
    for r in rows:
        if r.get("judge_sha256") == chosen and _number(r.get("score")):
            key = (r["dyad_id"], _attempt(r), r.get("turn"), r.get("agent"))
            latest[key] = (level_of[key[:2]], float(r["score"]))
    cells: dict[str, dict[int, list[float]]] = {}
    for (_, _, turn, _), (level, score) in latest.items():
        cells.setdefault(level, {}).setdefault(turn, []).append(score)
    block["levels"] = _ordered_levels(level_of, factor, level_order, keep=cells)
    block["series"] = {level: [{"turn": t, "n": len(v), "mean": statistics.fmean(v), "se": _se(v)}
                               for t, v in sorted(cells[level].items())] for level in block["levels"]}
    return block


def analyze_run(data_dir, run_id, factor=None, metric="alignment", judge=None, level_order=None, root=None) -> dict:
    """The analysis view of one run: survey shifts and score curves by `factor`, with what the run offers
    to choose from (its condition keys and levels, the metrics and judges in scores.jsonl). `level_order`
    defaults to the run's own (runs.level_order). ValueError for a factor the run's conditions do not have."""
    factor = factor or None
    judge = judge or None
    path = runs.run_dir(data_dir, run_id)
    dyads = runs.read_rows(path / "dyads.jsonl")
    status = runs.read_rows(path / "status.jsonl")
    surveys = runs.read_rows(path / "surveys.jsonl")
    scores = runs.read_rows(path / "scores.jsonl")
    if level_order is None:
        level_order = runs.level_order(data_dir, run_id, root=root)
    factors = runs.condition_levels(dyads, level_order)
    if factor is not None and factors and factor not in factors:
        raise ValueError(f"factor {factor!r} is not a condition key of run {run_id!r} ({', '.join(factors)})")
    return {"factor": factor, "metric": metric, "judge": judge, "factors": factors,
            "status_counts": runs.status_counts(status), "metrics_available": runs.metrics_present(scores),
            "judges": runs.judge_shas(scores),
            "survey": survey_shifts(dyads, status, surveys, factor=factor, level_order=level_order),
            "scores": score_curves(dyads, status, scores, factor=factor, metric=metric, judge=judge,
                                   level_order=level_order)}
