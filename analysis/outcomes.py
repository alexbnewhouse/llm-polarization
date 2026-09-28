"""Per-dyad pre and post answers, recoded, the four indices and their change scores.

    python -m analysis.outcomes --run-dir data/<run_id> [--batteries PATH] [--json-only] [--available-items]
        [--impute-refusals mid|low|high] [--baseline data/<baseline_run_id> ...] [--out DIR]

The instrument is the file the run recorded in manifest.json (`batteries.path`), checked against the
recorded `batteries.sha256`. When the file on disk no longer hashes to it, the matching version is
looked up in git history (REPRODUCIBILITY.md standard 12); failing that the command refuses, because a
different file can mean different directions. `--batteries` points at an archived copy instead.

Index definitions come from the instrument's `indices` block when it has one (instruments/batteries.json
from 1.0.0), otherwise from the remediation plan's Shared definitions:

- ideological: mean of the ideological items, each recoded so high = right (`min + max - x` for items
  whose `direction` is left), on the 1-5 scale. Positive change = rightward. The primary outcome.
- therm_gap: mean(therm_rep_*) - mean(therm_dem_*), signed, positive = warmer to Republicans.
- affective_abs: |therm_gap|, affective polarization.
- norms: mean of the agreement items.
- topic_item: the dialogue's own topic item alone, recoded high = right (PAP S6, secondary).

affective_abs changes as |post gap| - |pre gap| (PAP 3.4).

An index with any of its items missing (null answer, error, a reply cut off, absent row) is missing for
that dyad, which
is logged as `missing_survey` for that index only; the dyad stays in the other indices. `--json-only`
also treats answers salvaged from free text (`answer_method` other than json) as missing, the
sensitivity check REPRODUCIBILITY.md asks for. The PAP §6 sensitivities: `--available-items` (a mean
index from the items present when 5 of 7 ideological or 2 of 3 norms items are there) and
`--impute-refusals` (a null answer whose reply reads as a refusal, analysis.rates, put at the midpoint or
at either recoded end, for bounds).

`baseline_reference` reads the no-dialogue baseline (`harness.run baseline`, PAP §10 option C): per arm,
the mean and SD of every index over the K administrations, the scale for how large a movement is.
`--baseline` (once per arm) writes it to baseline_reference.json and adds it to outcomes.md."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from analysis._util import fmt, md_table, write_csv, write_json
from analysis.load import (LEVELS, CONTROL, Baseline, Run, add_common_args, answer_truncated, default_out,
                           load_baseline, load_from_args)
from analysis.rates import is_refusal
from harness.log import sha256_file, sha256_text

REPO = Path(__file__).resolve().parents[1]
INDICES = ("ideological", "therm_gap", "affective_abs", "norms")
SIGNED = ("ideological", "therm_gap", "topic_item")     # a direction exists: the shape test applies
# Directions for instruments that predate the `direction` field (0.1.0-placeholder). Same ids, same
# poles as instruments/batteries.json 1.0.0.
FALLBACK_DIRECTIONS = {
    "ideo_gender_racial_equality": "left", "ideo_immigration": "right", "ideo_redistribution": "left",
    "ideo_multiculturalism": "left", "ideo_gun_control": "left",
    "ideo_enforcement_militarization": "right", "ideo_decarbonization": "left",
}
TOPIC_ITEMS = {"immigration_enforcement": "ideo_enforcement_militarization",
               "decarbonization": "ideo_decarbonization"}


@dataclass
class Instrument:
    """The survey instrument a run used: its items, the index definitions and where they came from."""
    path: str
    sha256: str
    items: list[dict]
    indices: dict
    source: str
    warnings: list[str] = field(default_factory=list)

    @property
    def by_id(self) -> dict:
        return {it["id"]: it for it in self.items}


def _candidates(path: str, run_root: Path) -> list[Path]:
    p = Path(path)
    if p.is_absolute():
        return [p]
    return [Path.cwd() / p, run_root.parent.parent / p, REPO / p]


def _from_git(path: str, sha: str) -> tuple[str, str] | None:
    """(commit, text) of the first version of `path` in this repository's history hashing to `sha`."""
    rel = Path(path)
    if rel.is_absolute():
        try:
            rel = rel.resolve().relative_to(REPO)
        except ValueError:
            return None
    try:
        revs = subprocess.run(["git", "-C", str(REPO), "log", "--all", "--format=%H", "--", str(rel)],
                              capture_output=True, text=True, timeout=30).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
    for rev in revs:
        r = subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:{rel}"], capture_output=True, timeout=30)
        if r.returncode == 0 and hashlib.sha256(r.stdout).hexdigest() == sha:
            return rev, r.stdout.decode("utf-8")
    return None


def load_instrument(run: Run | Baseline, override: str | None = None,
                    allow_mismatch: bool = False) -> Instrument:
    """The instrument `run` (a run or a baseline run) recorded, verified by sha256 (see the module
    docstring)."""
    rec = run.manifest.get("batteries") or {}
    want = rec.get("sha256")
    warnings = []
    text = None
    source = ""
    recorded = rec.get("path") or "instruments/batteries.json"
    paths = [Path(override)] if override else _candidates(recorded, run.root)
    found = next((p for p in paths if p.exists()), None)
    if found is not None:
        got = sha256_file(found)
        if not want or got == want or allow_mismatch:
            text, source = found.read_text(encoding="utf-8"), str(found)
            if want and got != want:
                warnings.append(f"{found} hashes to {got[:12]}, the run recorded {want[:12]}; used anyway")
            if not want:
                warnings.append("manifest.json records no batteries sha256; the file was not verified")
    if text is None and want:
        hit = _from_git(rec.get("path") or "instruments/batteries.json", want)
        if hit:
            text, source = hit[1], f"git {hit[0][:12]}:{rec.get('path')}"
    if text is None:
        raise ValueError(f"no instrument file hashing to {str(want)[:12]} (manifest batteries.path "
                         f"{rec.get('path')!r}); pass --batteries with the archived copy, or "
                         "--allow-instrument-mismatch to use the file on disk")
    data = json.loads(text)
    items = data["items"] if isinstance(data, dict) else data
    ids = [it["id"] for it in items]
    if rec.get("item_ids") and rec["item_ids"] != ids:
        warnings.append("item ids differ from manifest.json batteries.item_ids")
    indices = parse_indices(data if isinstance(data, dict) else {}, items, warnings)
    return Instrument(str(found or rec.get("path")), sha256_text(text), items, indices, source, warnings)


def _direction(it: dict) -> str | None:
    return it.get("direction") or FALLBACK_DIRECTIONS.get(it["id"])


def parse_indices(data: dict, items: list[dict], warnings: list[str]) -> dict:
    """Normalise index definitions to {name: {kind: mean|diff|abs|topic, ...}} from the instrument's
    `indices` block, or from the Shared definitions when it has none."""
    by_id = {it["id"]: it for it in items}
    out: dict = {}
    spec = data.get("indices")
    if spec:
        for name, s in spec.items():
            if name.startswith("_"):
                continue
            combine = str(s.get("combine", "mean")).replace(" ", "")
            if "plus" in s or combine.startswith("mean(plus)"):
                out[name] = {"kind": "diff", "plus": list(s["plus"]), "minus": list(s["minus"])}
            elif combine == "abs" or "of" in s:
                out[name] = {"kind": "abs", "of": s["of"]}
            elif combine == "mean":
                rev = s.get("reverse")
                if rev is None:                          # derive from item directions
                    hm = s.get("high_means")
                    rev = [i for i in s["items"]
                           if hm in ("left", "right") and _direction(by_id[i]) not in (None, hm)]
                out[name] = {"kind": "mean", "items": list(s["items"]), "reverse": list(rev)}
            else:
                raise ValueError(f"index {name}: combine {s.get('combine')!r} is not one analysis knows")
    else:
        warnings.append("the instrument has no `indices` block; using the remediation plan's Shared "
                        "definitions")
        ideo = [it["id"] for it in items if it.get("battery") == "ideological"]
        unknown = [i for i in ideo if _direction(by_id[i]) is None]
        if unknown:
            raise ValueError(f"ideological items with no direction: {unknown}")
        out["ideological"] = {"kind": "mean", "items": ideo,
                              "reverse": [i for i in ideo if _direction(by_id[i]) == "left"]}
        out["therm_gap"] = {"kind": "diff", "plus": [i for i in by_id if i.startswith("therm_rep_")],
                            "minus": [i for i in by_id if i.startswith("therm_dem_")]}
        out["affective_abs"] = {"kind": "abs", "of": "therm_gap"}
        agree = [it["id"] for it in items if it.get("battery") == "agreement"]
        out["norms"] = {"kind": "mean", "items": agree, "reverse": []}
    missing_topic = [i for i in TOPIC_ITEMS.values() if i not in by_id]
    if missing_topic:
        warnings.append(f"topic items not in the instrument: {missing_topic}; the ideological index cannot "
                        "include them and topic_item is not computed")
    elif "topic_item" not in out:
        out["topic_item"] = {"kind": "topic", "items": TOPIC_ITEMS}
    for name, s in out.items():
        need = s.get("items", []) if s["kind"] == "mean" else s.get("plus", []) + s.get("minus", [])
        absent = [i for i in need if i not in by_id]
        if absent:
            raise ValueError(f"index {name} names items the instrument does not have: {absent}")
    return out


def recode(it: dict, x: float | None, reverse: bool) -> float | None:
    """`min + max - x` for a reversed item, x otherwise."""
    if x is None:
        return None
    return it["scale"]["min"] + it["scale"]["max"] - x if reverse else float(x)


def answers_by_dyad(run: Run, json_only: bool = False) -> tuple[dict, dict, set]:
    """{(dyad_id, phase): {item_id: answer or None}} from the analysed attempts' survey rows, the counts of
    what was dropped, and the (dyad_id, phase, item_id) keys whose null answer reads as a refusal
    (analysis.rates.is_refusal on raw_text). The last non-error row for a key wins; an error row alone is
    None."""
    out: dict = {}
    refused: set = set()
    counts = {"rows": 0, "error": 0, "null": 0, "truncated": 0, "refused": 0, "salvaged": 0,
              "salvaged_dropped": 0, "duplicates": 0}
    for r in run.surveys:
        counts["rows"] += 1
        slot = out.setdefault((r["dyad_id"], r["phase"]), {})
        if r.get("error"):
            counts["error"] += 1
            slot.setdefault(r["item_id"], None)
            continue
        val = r.get("answer")
        method = r.get("answer_method")
        if answer_truncated(r):                        # cut off: missing, and not read as a refusal
            counts["truncated"] += 1
            val = None
        elif val is None:
            counts["null"] += 1
            if is_refusal(r.get("raw_text")):
                counts["refused"] += 1
                refused.add((r["dyad_id"], r["phase"], r["item_id"]))
        elif method not in (None, "json"):
            counts["salvaged"] += 1
            if json_only:
                counts["salvaged_dropped"] += 1
                val = None
        if slot.get(r["item_id"]) is not None:
            counts["duplicates"] += 1
        slot[r["item_id"]] = val
    return out, counts, refused


def _reversed_items(inst: Instrument) -> set:
    """Items read with high = left: an ideological index's reverse list and every left topic item."""
    rev = {i for s in inst.indices.values() if s["kind"] == "mean" for i in s.get("reverse", [])}
    return rev | {it["id"] for it in inst.items
                  if it.get("battery") == "ideological" and _direction(it) == "left"}


def impute_value(it: dict, where: str, reversed_: bool) -> int:
    """The answer that puts an item at the scale midpoint (`mid`), or at the left/low (`low`) or right/high
    (`high`) end once recoded: the refusal sensitivity of PAP §6."""
    lo, hi = it["scale"]["min"], it["scale"]["max"]
    v = {"mid": (lo + hi) / 2, "low": lo, "high": hi}[where]
    return lo + hi - v if reversed_ else v


def index_value(inst: Instrument, name: str, ans: dict, topic: str | None, cache: dict,
                available: bool = False) -> float | None:
    """One index for one dyad and phase; None when any item it needs is missing. With `available`, a mean
    index uses the items present when at most two ideological items (one norms item) are missing, the
    PAP §6 sensitivity (5 of 7, 2 of 3)."""
    s = inst.indices[name]
    by_id = inst.by_id
    if s["kind"] == "mean":
        vals = [recode(by_id[i], ans.get(i), i in s["reverse"]) for i in s["items"]]
        have = [v for v in vals if v is not None]
        if not vals:
            return None
        if len(have) == len(vals):
            return sum(have) / len(have)
        allowed = 2 if name == "ideological" else 1
        return sum(have) / len(have) if available and have and len(vals) - len(have) <= allowed else None
    if s["kind"] == "diff":
        p = [ans.get(i) for i in s["plus"]]; m = [ans.get(i) for i in s["minus"]]
        if any(v is None for v in p + m) or not p or not m:
            return None
        return sum(p) / len(p) - sum(m) / len(m)
    if s["kind"] == "abs":
        v = cache.get(s["of"])
        return None if v is None else abs(v)
    if s["kind"] == "topic":
        item = s["items"].get(topic)
        if item is None or item not in by_id:
            return None
        return recode(by_id[item], ans.get(item), _direction(by_id[item]) == "left")
    raise ValueError(s["kind"])


def compute_outcomes(run: Run, inst: Instrument, json_only: bool = False, available: bool = False,
                     impute_refusals: str | None = None, log: bool = True) -> tuple[list[dict], dict]:
    """One row per planned dyad with pre, post and change for every index, and the counts. ITT dyads
    with a missing index are logged as `missing_survey` for that index. `available` and
    `impute_refusals` (mid, low or high) are the PAP §6 sensitivities; the primary analysis uses neither.
    `log=False` leaves the exclusion log alone (a sensitivity refit of the same run)."""
    answers, counts, refused = answers_by_dyad(run, json_only)
    if impute_refusals:
        rev = _reversed_items(inst)
        for dyad_id, phase, item in refused:
            if item in inst.by_id:
                answers[(dyad_id, phase)][item] = impute_value(inst.by_id[item], impute_refusals,
                                                               item in rev)
        counts["imputed"] = len(refused)
    order = [n for n in INDICES if n in inst.indices] + [n for n in inst.indices if n not in INDICES]
    rows = []
    for d in run.dyads:
        row = {k: d.get(k) for k in ("run_id", "arm", "dyad_id", "attempt", "topic", "ideology", "openness",
                                      "role", "cell", "dose", "control", "persona_mode", "in_itt", "flagged",
                                      "adherence_mean", "adherence_n")}
        missing = []
        for phase in ("pre", "post"):
            ans = answers.get((d["dyad_id"], phase), {})
            cache: dict = {}
            for name in order:
                cache[name] = (index_value(inst, name, ans, d.get("topic"), cache, available)
                               if d["attempt"] else None)
                row[f"{name}_{phase}"] = cache[name]
        for name in order:
            pre, post = row[f"{name}_pre"], row[f"{name}_post"]
            row[f"{name}_change"] = None if pre is None or post is None else post - pre
            if d["in_itt"] and row[f"{name}_change"] is None and not (name == "topic_item" and d.get("topic")
                                                                      not in TOPIC_ITEMS):
                missing.append(name)
        for name in missing if log else []:
            if name == "affective_abs" and "therm_gap" in missing:
                continue                                 # already logged under therm_gap
            run.exclude(d, "missing_survey", "null, error or absent pre/post answer", index=name)
        rows.append(row)
    counts["indices"] = order
    counts["rules"] = {"json_only": json_only, "available_items": available,
                       "impute_refusals": impute_refusals}
    return rows, counts


def item_changes(run: Run, inst: Instrument, index: str = "ideological",
                 json_only: bool = False) -> list[dict]:
    """Per-item change scores, recoded as the index recodes them: the rows of the item-level stacked model
    (PAP §7.5a). One row per ITT dyad and item with both answers."""
    answers, _, _ = answers_by_dyad(run, json_only)
    s = inst.indices[index]
    out = []
    for d in run.dyads:
        if not d["in_itt"]:
            continue
        pre, post = answers.get((d["dyad_id"], "pre"), {}), answers.get((d["dyad_id"], "post"), {})
        for i in s["items"]:
            a, b = (recode(inst.by_id[i], x.get(i), i in s["reverse"]) for x in (pre, post))
            if a is not None and b is not None:
                keys = ("run_id", "arm", "dyad_id", "topic", "ideology", "openness", "role", "dose",
                        "control", "flagged", "persona_mode")
                out.append({**{k: d.get(k) for k in keys}, "item": i, f"{index}_change": b - a,
                            "in_itt": True})
    return out


def _baseline_values(inst: Instrument, answers: dict) -> dict[str, list[float]]:
    """{index: values over the administrations}; topic_item once per topic as `topic_item:<topic>`."""
    order = [n for n in INDICES if n in inst.indices] + [n for n in inst.indices if n not in INDICES]
    vals: dict[str, list[float]] = {}
    for ans in answers.values():
        cache: dict = {}
        for name in order:
            if inst.indices[name]["kind"] == "topic":
                for topic in inst.indices[name]["items"]:
                    v = index_value(inst, name, ans, topic, cache)
                    vals.setdefault(f"{name}:{topic}", [])
                    if v is not None:
                        vals[f"{name}:{topic}"].append(v)
                continue
            cache[name] = index_value(inst, name, ans, None, cache)
            vals.setdefault(name, [])
            if cache[name] is not None:
                vals[name].append(cache[name])
    return vals


def baseline_reference(baselines, inst: Instrument | None = None) -> dict[str, dict]:
    """PAP §10 option C: {arm: {index: {n, missing, mean, sd}}} from each arm's no-dialogue baseline, the
    pre battery administered K times (analysis.load.load_baseline). `baselines` is a list of Baseline or
    of baseline run directories, one per arm. An administration with an item an index needs missing
    leaves that index out (counted in `missing`). SD is the sample SD (ddof 1). The instrument is each
    baseline's own recorded one unless `inst` is given; `settings` repeats the run's temperature, top_p,
    n_predict and schema, since the reference means little without them."""
    out: dict[str, dict] = {}
    for b in baselines:
        b = b if isinstance(b, Baseline) else load_baseline(b)
        if b.arm in out:
            raise ValueError(f"two baseline runs for arm {b.arm!r}")
        ins = inst or load_instrument(b)
        k = len(b.answers)
        ref = {}
        for name, v in _baseline_values(ins, b.answers).items():
            ref[name] = {"n": len(v), "missing": k - len(v),
                         "mean": float(np.mean(v)) if v else None,
                         "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None}
        out[b.arm] = {"run_id": b.run_id, "administrations": k, "settings": b.settings, "counts": b.counts,
                      "indices": ref}
    return out


def baseline_markdown(ref: dict) -> str:
    """The baseline reference as a table: mean (SD) of each index per arm."""
    arms = list(ref)
    names = list(dict.fromkeys(n for a in arms for n in ref[a]["indices"]))
    rows = []
    for n in names:
        cells = []
        for a in arms:
            e = ref[a]["indices"].get(n) or {}
            cells.append("" if e.get("mean") is None else f"{fmt(e['mean'])} ({fmt(e['sd'])}, n={e['n']})")
        rows.append([n] + cells)
    head = [f"{a} ({ref[a]['run_id']}; K={ref[a]['administrations']}, temperature "
            f"{ref[a]['settings'].get('temperature')})" for a in arms]
    return "\n".join(["## No-dialogue baseline: mean (SD) per arm", "",
                      md_table(["index"] + head, rows)]) + "\n"


def summary_markdown(run: Run, inst: Instrument, rows: list[dict], counts: dict) -> str:
    """Mean change by topic and ideology level for each index, ITT sample."""
    out = [f"# Outcomes: {run.run_id} ({run.arm})", "",
           f"Instrument {inst.source} (sha256 {inst.sha256[:12]}). Survey rows {counts['rows']}: "
           f"{counts['null']} null, {counts['truncated']} cut off, {counts['error']} error, "
           f"{counts['salvaged']} salvaged"
           + (" (dropped: --json-only)" if counts["salvaged_dropped"] else "") + ".", ""]
    out += [f"- {w}" for w in inst.warnings]
    levels = list(LEVELS) + [CONTROL]
    for name in counts["indices"]:
        tab = []
        for topic in sorted({r["topic"] for r in rows if r["topic"]}):
            cells = []
            for lv in levels:
                v = [r[f"{name}_change"] for r in rows if r["in_itt"] and r["topic"] == topic
                     and r["ideology"] == lv and r[f"{name}_change"] is not None]
                cells.append(f"{fmt(sum(v) / len(v))} (n={len(v)})" if v else "")
            tab.append([topic] + cells)
        out += ["", f"## {name}: mean post - pre", "", md_table(["topic"] + levels, tab)]
    return "\n".join(out) + "\n"


def add_instrument_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--batteries", default=None,
                   help="the instrument file, overriding manifest batteries.path")
    p.add_argument("--allow-instrument-mismatch", action="store_true",
                   help="use the file on disk even when it does not hash to what the run recorded")
    p.add_argument("--json-only", action="store_true",
                   help="treat answers salvaged from free text as missing")
    p.add_argument("--available-items", action="store_true",
                   help="sensitivity: a mean index from the items present (5 of 7 ideological, 2 of 3 "
                        "norms)")
    p.add_argument("--impute-refusals", choices=("mid", "low", "high"), default=None,
                   help="sensitivity: refused items at the scale midpoint or at either recoded end")
    p.add_argument("--baseline", action="append", default=[],
                   help="a no-dialogue baseline run directory (harness.run baseline); repeat per arm")


def outcomes_from_args(a, run: Run) -> tuple[Instrument, list[dict], dict]:
    inst = load_instrument(run, a.batteries, a.allow_instrument_mismatch)
    rows, counts = compute_outcomes(run, inst, a.json_only, a.available_items, a.impute_refusals)
    return inst, rows, counts


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m analysis.outcomes", description=__doc__.split("\n\n")[0])
    add_common_args(p)
    add_instrument_args(p)
    a = p.parse_args(argv)
    try:
        run = load_from_args(a)
        inst, rows, counts = outcomes_from_args(a, run)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    out = Path(a.out) if a.out else default_out(a.run_dir)
    write_csv(out / "outcomes.csv", rows)
    md = summary_markdown(run, inst, rows, counts)
    wrote = "outcomes.csv, outcomes.md"
    if a.baseline:
        try:
            ref = baseline_reference(a.baseline)
        except (FileNotFoundError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        write_json(out / "baseline_reference.json", ref)
        md += "\n" + baseline_markdown(ref)
        wrote += ", baseline_reference.json"
    (out / "outcomes.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {out}/{wrote}")
    return 0



if __name__ == "__main__":
    sys.exit(main())
