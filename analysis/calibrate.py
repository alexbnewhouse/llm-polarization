"""Calibrating the adherence judge on hand labels (docs/decisions/persona-stability.md §3): sample pilot
turns for labelling, then score the judge against the labels and pick the flag threshold.

    python -m analysis.calibrate sample --run-dir data/<pilot> [--n 120] [--seed 0] [--out labels.csv]
    python -m analysis.calibrate score  --run-dir data/<pilot> --labels labels.csv \\
        [--label-min 1 --label-max 5 --adherent-at 4] [--metric prompt_to_line] [--judge PREFIX]

`sample` draws seeker turns of treated dyads (adherence is undefined for the bare control), stratified
by ideology level x turn-index bin (`--bins`, equal-width over 1..n_turns), as evenly as the strata
allow: a stratum too small to fill its share gives the remainder to the others. The sheet has the
persona and the line, and an empty `label` column; it never shows the judge's score. Labels are on
`--label-min`..`--label-max` (default 1-5, 5 = fully in character). An optional `label2` column from a
second labeller adds inter-labeller agreement.

`score` joins the labels to the judge's scores for the same (dyad, attempt, turn) and reports:
- Krippendorff's alpha (interval) between judge and hand label, labels rescaled to 0-1, with a
  bootstrap 95% CI over labelled turns;
- quadratic-weighted kappa, the judge's score rounded onto the label scale;
- the threshold on the judge score that best separates non-adherent turns (label below
  `--adherent-at`) from adherent ones by balanced accuracy, for the flag rule's `score < threshold`.
  That number goes to `python -m harness.run flags --threshold`."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
from analysis._util import fmt, read_csv, write_csv, write_json
from analysis.load import LEVELS, load_run, pick_judge
from harness.scorer import ADHERENCE_METRICS
from harness.transcript import SEEKER

SHEET_COLUMNS = ["sample_id", "dyad_id", "attempt", "turn", "agent", "topic", "ideology", "openness", "role",
                 "turn_bin", "persona_text", "text", "label", "label2", "labeller", "note"]


def turn_bin(turn: int, n_turns: int, bins: int) -> int:
    """1-based equal-width bin of `turn` over 1..n_turns."""
    return min(bins, 1 + (turn - 1) * bins // max(n_turns, 1))


def allocate(sizes: dict, n: int) -> dict:
    """Split n across strata as evenly as their sizes allow (water-filling), deterministic."""
    alloc = {k: 0 for k in sizes}
    left = min(n, sum(sizes.values()))
    open_ = sorted(k for k in sizes if sizes[k] > 0)
    while left > 0 and open_:
        share = max(1, left // len(open_))
        for k in list(open_):
            take = min(share, sizes[k] - alloc[k], left)
            alloc[k] += take
            left -= take
            if alloc[k] >= sizes[k]:
                open_.remove(k)
            if left == 0:
                break
    return alloc


def sample_turns(run_dir, n: int = 120, seed: int = 0, bins: int = 4, agent: str = SEEKER) -> list[dict]:
    """The label sheet rows (see the module docstring)."""
    run = load_run(run_dir)
    dy = {d["dyad_id"]: d for d in run.dyads if d["in_itt"] and not d["control"]}
    personas = {(r["dyad_id"], int(r.get("attempt", 1))): r.get("persona_text") for r in run.raw["dyads"]}
    strata: dict = {}
    for r in run.turns:
        d = dy.get(r["dyad_id"])
        if d is None or r["agent"] != agent or r.get("error") or r.get("finish_reason") == "error":
            continue
        key = (d["ideology"], turn_bin(r["turn"], int(d["n_turns"] or r["turn"]), bins))
        strata.setdefault(key, []).append(r)
    alloc = allocate({k: len(v) for k, v in strata.items()}, n)
    rng = np.random.default_rng(seed)
    rows = []
    for key in sorted(strata, key=lambda k: (LEVELS.index(k[0]) if k[0] in LEVELS else 99, k[1])):
        pool = sorted(strata[key], key=lambda r: (r["dyad_id"], r["turn"]))
        pick = sorted(rng.choice(len(pool), size=alloc[key], replace=False)) if alloc[key] else []
        for i in pick:
            r = pool[i]
            d = dy[r["dyad_id"]]
            rows.append({"dyad_id": r["dyad_id"], "attempt": r["attempt"], "turn": r["turn"],
                         "agent": r["agent"],
                         "topic": d["topic"], "ideology": d["ideology"], "openness": d["openness"],
                         "role": d["role"], "turn_bin": key[1],
                         "persona_text": personas.get((r["dyad_id"], r["attempt"])), "text": r["text"],
                         "label": "", "label2": "", "labeller": "", "note": ""})
    order = rng.permutation(len(rows))              # shuffled: the labeller does not see the strata in order
    rows = [rows[i] for i in order]
    for i, r in enumerate(rows, 1):
        r["sample_id"] = i
    return rows


def krippendorff_interval(units: list[list[float | None]]) -> float | None:
    """Krippendorff's alpha, interval metric, for units coded by any number of coders (None = missing).
    Units with fewer than two values are not pairable and are dropped."""
    vals = [[v for v in u if v is not None] for u in units]
    vals = [u for u in vals if len(u) >= 2]
    if not vals:
        return None
    n = sum(len(u) for u in vals)
    do = sum(sum((a - b) ** 2 for i, a in enumerate(u) for j, b in enumerate(u) if i != j) / (len(u) - 1)
             for u in vals) / n
    allv = np.array([v for u in vals for v in u])
    de = float(((allv[:, None] - allv[None, :]) ** 2).sum()) / (n * (n - 1))
    return None if de == 0 else 1 - do / de


def weighted_kappa(a: list[int], b: list[int], cats: list[int]) -> float | None:
    """Cohen's kappa with quadratic weights between two ratings on the ordered categories `cats`."""
    k = len(cats)
    if k < 2 or not a:
        return None
    idx = {c: i for i, c in enumerate(cats)}
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    w = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    de = (w * exp).sum()
    return None if de == 0 else float(1 - (w * obs).sum() / de)


def best_threshold(scores: list[float], adherent: list[bool]) -> dict:
    """The threshold t maximising balanced accuracy of `score < t` for non-adherent turns. Candidates are
    midpoints between distinct scores; ties report the whole tied range and pick its midpoint."""
    s = np.array(scores, float); bad = ~np.array(adherent, bool)
    if bad.all() or not bad.any():
        return {"threshold": None, "reason": "labels are all one class"}
    u = np.unique(s)
    cands = np.concatenate([[u[0] - 1e-6], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-6]])
    best, ties = -1.0, []
    for t in cands:
        pred = s < t
        tpr = (pred & bad).sum() / bad.sum()
        tnr = (~pred & ~bad).sum() / (~bad).sum()
        ba = (tpr + tnr) / 2
        if ba > best + 1e-12:
            best, ties = ba, [(t, tpr, tnr)]
        elif abs(ba - best) <= 1e-12:
            ties.append((t, tpr, tnr))
    mid = ties[len(ties) // 2]
    return {"threshold": float(mid[0]), "balanced_accuracy": float(best), "sensitivity": float(mid[1]),
            "specificity": float(mid[2]), "tied_range": [float(ties[0][0]), float(ties[-1][0])],
            "n": int(len(s)), "n_nonadherent": int(bad.sum())}


def score_labels(run_dir, labels_path, label_min: float = 1, label_max: float = 5, adherent_at: float = 4,
                 metric: str = "prompt_to_line", judge: str | None = None, n_boot: int = 1000,
                 seed: int = 0) -> dict:
    """Agreement between the judge and the hand labels, and the threshold (see the module docstring)."""
    run = load_run(run_dir, judge=judge, metric=metric)
    j = pick_judge(run.scores, metric, judge)
    if j is None:
        raise ValueError(f"no {metric} scores in {run_dir}")
    sc = {(s["dyad_id"], int(s["attempt"]), int(s["turn"])): float(s["score"]) for s in run.scores
          if s.get("judge_sha256") == j and s.get("metric") == metric and s.get("agent") == SEEKER
          and s.get("score") is not None and not s.get("error")}
    span = label_max - label_min
    pairs, second, unmatched, unlabelled = [], [], 0, 0
    for r in read_csv(labels_path):
        if not str(r.get("label", "")).strip():
            unlabelled += 1
            continue
        key = (r["dyad_id"], int(r["attempt"]), int(r["turn"]))
        lab = float(r["label"])
        if not label_min <= lab <= label_max:
            raise ValueError(f"label {lab} outside {label_min}-{label_max} (sample {r.get('sample_id')})")
        if str(r.get("label2", "")).strip():
            second.append([(lab - label_min) / span, (float(r["label2"]) - label_min) / span])
        if key not in sc:
            unmatched += 1
            continue
        pairs.append((sc[key], lab))
    if len(pairs) < 2:
        raise ValueError(f"{len(pairs)} labelled turns have a judge score; nothing to compare")
    js = np.array([p[0] for p in pairs]); ls = np.array([p[1] for p in pairs])
    lnorm = (ls - label_min) / span
    alpha = krippendorff_interval([[a, b] for a, b in zip(js, lnorm)])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(js), len(js))
        v = krippendorff_interval([[a, b] for a, b in zip(js[i], lnorm[i])])
        if v is not None:
            boots.append(v)
    cats = list(range(int(label_min), int(label_max) + 1))
    judge_cat = [int(min(label_max, max(label_min, round(label_min + x * span)))) for x in js]
    kappa = weighted_kappa(judge_cat, [int(round(x)) for x in ls], cats) if span == int(span) else None
    thr = best_threshold(list(js), list(ls >= adherent_at))
    r = float(np.corrcoef(js, lnorm)[0, 1]) if js.std() > 0 and lnorm.std() > 0 else None
    return {"run_dir": str(run_dir), "judge": j, "metric": metric, "n_pairs": len(pairs),
            "unmatched": unmatched,
            "unlabelled": unlabelled, "label_scale": [label_min, label_max], "adherent_at": adherent_at,
            "krippendorff_alpha_interval": alpha,
            "alpha_ci95": ([float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
                           if boots else None),
            "weighted_kappa_quadratic": kappa, "pearson_r": r,
            "mean_abs_diff": float(np.abs(js - lnorm).mean()),
            "threshold": thr,
            "inter_labeller_alpha": krippendorff_interval(second) if len(second) >= 2 else None,
            "n_double_labelled": len(second)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m analysis.calibrate", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="write a label sheet")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--n", type=int, default=120)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--bins", type=int, default=4)
    s.add_argument("--out", default=None, help="CSV path (default <run-dir>/analysis/labels.csv)")
    c = sub.add_parser("score", help="judge agreement and the flag threshold from a labelled sheet")
    c.add_argument("--run-dir", required=True)
    c.add_argument("--labels", required=True)
    c.add_argument("--label-min", type=float, default=1)
    c.add_argument("--label-max", type=float, default=5)
    c.add_argument("--adherent-at", type=float, default=4, help="a label at or above this is adherent")
    c.add_argument("--metric", default="prompt_to_line", choices=ADHERENCE_METRICS)
    c.add_argument("--judge", default=None)
    c.add_argument("--out", default=None, help="JSON path (default <run-dir>/analysis/calibration.json)")
    a = p.parse_args(argv)
    try:
        if a.cmd == "sample":
            rows = sample_turns(a.run_dir, a.n, a.seed, a.bins)
            path = Path(a.out) if a.out else Path(a.run_dir) / "analysis" / "labels.csv"
            if path.exists():
                print(f"error: {path} exists; labels are not overwritten", file=sys.stderr)
                return 1
            write_csv(path, rows, SHEET_COLUMNS)
            by: dict = {}
            for r in rows:
                by[(r["ideology"], r["turn_bin"])] = by.get((r["ideology"], r["turn_bin"]), 0) + 1
            print(f"wrote {path}: {len(rows)} turns in {len(by)} strata (ideology x turn bin)")
            return 0
        res = score_labels(a.run_dir, a.labels, a.label_min, a.label_max, a.adherent_at, a.metric, a.judge)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    path = Path(a.out) if a.out else Path(a.run_dir) / "analysis" / "calibration.json"
    write_json(path, res)
    t = res["threshold"]
    ci = res["alpha_ci95"]
    print(f"judge {res['judge'][:12]} vs hand labels, {res['metric']}: {res['n_pairs']} turns "
          f"({res['unmatched']} labelled turns without a judge score, {res['unlabelled']} unlabelled)")
    print(f"  Krippendorff alpha (interval) {fmt(res['krippendorff_alpha_interval'])}"
          + (f" [{fmt(ci[0])}, {fmt(ci[1])}]" if ci else "")
          + f"; weighted kappa (quadratic) {fmt(res['weighted_kappa_quadratic'])}"
          + f"; r {fmt(res['pearson_r'])}")
    if res["inter_labeller_alpha"] is not None:
        print(f"  inter-labeller alpha {fmt(res['inter_labeller_alpha'])} on "
              f"{res['n_double_labelled']} turns")
    if t.get("threshold") is None:
        print(f"  threshold: none ({t['reason']})")
    else:
        print(f"  threshold {t['threshold']:.3f}: balanced accuracy {t['balanced_accuracy']:.3f} "
              "(sensitivity "
              f"{t['sensitivity']:.3f}, specificity {t['specificity']:.3f}; ties "
              f"{t['tied_range'][0]:.3f}-{t['tied_range'][1]:.3f}; "
              f"{t['n_nonadherent']}/{t['n']} non-adherent)")
        print(f"  -> python -m harness.run flags --threshold {t['threshold']:.3f}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
