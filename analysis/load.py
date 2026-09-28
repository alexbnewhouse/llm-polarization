"""Read one run directory into tidy tables, apply the latest-complete-attempt rule and the pre-registered
technical exclusions, and write the exclusion log.

    python -m analysis.load --run-dir data/<run_id> [--judge PREFIX] [--threshold T] [--out DIR]

The attempt rule is the harness's own (`harness.scorer.latest_complete_attempts`, from
`harness.log.resume_index`): a dyad is analysed at the highest attempt whose status is `complete`, and
rows of every other attempt are dropped. Exclusions (docs/pap/pre-analysis-plan.md §6) are technical
only; adherence never excludes a dyad. Each has a scope:

| reason | scope | meaning |
|---|---|---|
| not_run | dyad | in input-dyads.jsonl, no status row at all (a wave stopped before it) |
| incomplete | dyad | status rows, but no attempt reached `complete` |
| error_rows | dyad | the analysed attempt has a turn or in-run survey row with an error |
| short_dialogue | dyad | the analysed attempt has fewer than n_turns turns (2 * n_turns message rows) |
| truncated | dyad | a message row has `truncated: true` (the slot ran out of context) |
| missing_survey | index | a pre or post answer an index needs is null, cut off or absent (outcomes.py) |
| survey_settings | row | survey rows at another temperature or n_predict than the analysed pass: set aside |
| judge_failure | turn | a main-scope target of the chosen judge has only `error` rows: unscored, counted |
| unscored | adherence | a treated dyad with no adherence score at all while the run has been scored |

The dyad-scope reasons are "technical incompleteness" (PAP §6), handled as missing data and tested
against ideology level (`incompleteness_test`). `finish_reason: "length"` (the n_predict cap),
`cache_warning`, `attempt > 1` and refusals are reported (analysis.rates), not excluded. The bare control
is never in the adherence sample: adherence is undefined without a persona (F10).

Survey rows. The analysed pass is `origin` (default `run`) at the instrument's settings: the JSON schema,
temperature 0 and 32 tokens (`survey_key`; `--survey-temperature` and `--survey-n-predict` pick a pass
made with `survey --temperature/--n-predict`). Rows of that origin at other settings are not mixed in:
they are set aside and logged as `survey_settings`. The unconstrained check's rows (`schema: false`,
`survey --no-schema --sample N`) never enter the outcome tables; `unconstrained_surveys` returns them for
the agreement rate in analysis.rates, and `--include-unconstrained` analyses them in place of the
constrained pass. A row with `truncated: true` or answer_method `truncated` is a missing answer.

Score rows. Each row's `scope` says which `score` pass wrote it (pilot, main, stance); rows written before
scope was recorded get one from the turn cadence (`score_scope`). Adherence, `judge_failure` and the flag
rule use main-scope rows only; analysis.estimate's stance analyses use stance-scope rows, and a row's
`subsample` marks the second judge's pass for the agreement statistic.

The no-dialogue baseline (`harness.run baseline`) is its own run directory per arm, read by
`load_baseline` and `baseline_distributions`.
"""
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from analysis import stats
from analysis._util import md_table, write_csv, write_json, write_jsonl
from harness.log import read_jsonl
from harness.scorer import ADHERENCE_METRICS, MAIN_CADENCE, flag_dialogues, latest_complete_attempts
from harness.survey import SURVEY_N_PREDICT, SURVEY_TEMPERATURE, TRUNCATED
from harness.transcript import SEEKER

LEVELS = ("strong_left", "lean_left", "moderate", "lean_right", "strong_right")
CONTROL = "none"
# Signed ideology dose (PAP §3.5): -2 strong_left .. +2 strong_right. The control is dose 0 with the
# treated indicator T = 0; moderate is dose 0 with T = 1, so moderate is not the control.
DOSE = {"strong_left": -2, "lean_left": -1, "moderate": 0, "lean_right": 1, "strong_right": 2, CONTROL: 0}
DYAD_REASONS = ("not_run", "incomplete", "error_rows", "short_dialogue", "truncated")
REASONS = DYAD_REASONS + ("missing_survey", "survey_settings", "judge_failure", "unscored")
SCOPE = {**{r: "dyad" for r in DYAD_REASONS}, "missing_survey": "index", "survey_settings": "row",
         "judge_failure": "turn", "unscored": "adherence"}


def survey_key(row: dict) -> tuple:
    """How a survey row was sampled: (schema, temperature, n_predict), as harness.run reads it. Rows written
    before these fields existed were schema-constrained, greedy and 32 tokens."""
    return (row.get("schema", True) is not False, float(row.get("temperature", SURVEY_TEMPERATURE)),
            int(row.get("n_predict", SURVEY_N_PREDICT)))


def answer_truncated(row: dict) -> bool:
    """A survey row whose answer is missing because the reply was cut off: `truncated` (the context ran
    out) or answer_method `truncated` (the n_predict cap cut it before it parsed)."""
    return row.get("truncated") is True or row.get("answer_method") == TRUNCATED


def score_scope(row: dict, final_turn: int | None = None) -> str:
    """The `score` pass that wrote a score row: its `scope` field, else (a row written before scope was
    recorded) main for a seeker row and stance for a mentor row on the main cadence (turns 4, 8, ... and
    the dyad's final turn), pilot otherwise."""
    if row.get("scope"):
        return row["scope"]
    t = row.get("turn")
    if t is not None and (t % MAIN_CADENCE == 0 or t == final_turn):
        return "main" if row.get("agent") == SEEKER else "stance"
    return "pilot"


@dataclass
class Run:
    """One run, loaded. `dyads` has one row per planned dyad (the analysed attempt's condition and
    status); `turns`, `surveys` and `scores` hold the analysed attempts' rows only; `raw` holds every file
    as read, for the rates that need all attempts."""
    root: Path
    run_id: str
    manifest: dict
    arm: str
    dyads: list[dict]
    turns: list[dict]
    surveys: list[dict]
    scores: list[dict]
    raw: dict
    exclusions: list[dict] = field(default_factory=list)
    judge: str | None = None
    metric: str = "prompt_to_line"
    notes: list[str] = field(default_factory=list)
    survey_key: tuple = (True, SURVEY_TEMPERATURE, SURVEY_N_PREDICT)

    def __post_init__(self):
        self._by_id = {d["dyad_id"]: d for d in self.dyads}
        self._final = {d["dyad_id"]: d.get("n_turns") for d in self.dyads}

    def dyad(self, dyad_id: str) -> dict:
        return self._by_id[dyad_id]

    def scope_of(self, score_row: dict) -> str:
        """score_scope for one of this run's score rows."""
        return score_scope(score_row, self._final.get(score_row["dyad_id"]))

    def scores_in(self, scope: str) -> list[dict]:
        """The analysed attempts' score rows of one scope (pilot, main or stance)."""
        return [s for s in self.scores if self.scope_of(s) == scope]

    def exclude(self, dyad: dict, reason: str, detail: str = "", index: str | None = None) -> None:
        """Record one exclusion. A dyad-scope reason takes the dyad out of the ITT sample; `unscored` takes
        it out of the adherence sample; `judge_failure` only counts (the turn is missing, not the dyad)."""
        if reason not in REASONS:
            raise ValueError(f"unknown exclusion reason {reason!r}")
        row = {"run_id": self.run_id, "arm": self.arm, "dyad_id": dyad["dyad_id"],
               "attempt": dyad.get("attempt"),
               "reason": reason, "scope": SCOPE[reason], "index": index, "detail": detail,
               **{k: dyad.get(k) for k in ("topic", "ideology", "openness", "role", "persona_mode")}}
        self.exclusions.append(row)
        dyad.setdefault("excluded", []).append(reason if index is None else f"{reason}:{index}")
        if SCOPE[reason] == "dyad":
            dyad["in_itt"] = False
        if SCOPE[reason] == "adherence":
            dyad["in_adherence"] = False


def arm_label(manifest: dict) -> str:
    """The mentor arm a run belongs to: its alias, else the GGUF file name, else its family."""
    m = manifest.get("mentor") or {}
    return m.get("alias") or Path(str(m.get("model_path") or "")).name or m.get("family") or "unknown"


def _condition_fields(row: dict) -> dict:
    c = row.get("condition") or {}
    ideology = c.get("ideology")
    return {"topic": c.get("topic"), "ideology": ideology, "openness": c.get("openness"),
            "role": c.get("role"),
            "control": ideology == CONTROL, "dose": DOSE.get(ideology),
            "persona_mode": row.get("persona_mode"), "n_turns": row.get("n_turns"), "seed": row.get("seed")}


def cell_of(d: dict) -> str:
    """The factorial cell: topic/ideology/openness, the control as topic/none."""
    if d.get("control") or d.get("ideology") == CONTROL:
        return f"{d.get('topic')}/none"
    return f"{d.get('topic')}/{d.get('ideology')}/{d.get('openness')}"


def pick_judge(scores: list[dict], metric: str, prefix: str | None) -> str | None:
    """The judge whose scores to use for `metric`: the one matching `prefix`, else the only one. Raises
    when a prefix is ambiguous or when two judges scored and none was named."""
    judges = sorted({s.get("judge_sha256") for s in scores if s.get("metric") == metric} - {None})
    if prefix:
        m = [j for j in judges if str(j).startswith(prefix)]
        if len(m) != 1:
            raise ValueError(f"--judge {prefix!r} matches {len(m)} of the judges that scored {metric}: "
                             f"{judges}")
        return m[0]
    if len(judges) > 1:
        raise ValueError(f"{len(judges)} judges scored {metric}; pass --judge PREFIX ({judges})")
    return judges[0] if judges else None


def load_run(run_dir, judge: str | None = None, metric: str = "prompt_to_line", origin: str = "run",
             threshold: float | None = None, run_length: int = 3, include_unconstrained: bool = False,
             survey_temperature: float = SURVEY_TEMPERATURE, survey_n_predict: int = SURVEY_N_PREDICT) -> Run:
    """Load a run directory and apply the exclusions. `origin` picks the in-run survey pass (`run`) or a
    later re-administration (`readministered`), and `survey_temperature` and `survey_n_predict` the pass's
    settings; `include_unconstrained` analyses the unconstrained check's rows (schema false) instead of
    the constrained ones (see the module docstring). `metric` and `judge` choose the adherence scores;
    flags come from flags.jsonl, or are computed with `threshold` when given (harness.scorer.flag_dialogues),
    on main-scope rows."""
    root = Path(run_dir)
    if not (root / "manifest.json").exists():
        raise FileNotFoundError(f"{root}/manifest.json not found: not a run directory")
    if metric not in ADHERENCE_METRICS:
        raise ValueError(f"metric must be one of {ADHERENCE_METRICS}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    raw = {name: read_jsonl(root / f"{name}.jsonl")
           for name in ("input-dyads", "dyads", "status", "turns", "surveys", "scores", "flags")}
    run_id = manifest.get("run_id") or root.name
    complete = latest_complete_attempts(raw["status"])
    seen_status: dict[str, int] = {}
    for r in raw["status"]:
        seen_status[r["dyad_id"]] = max(seen_status.get(r["dyad_id"], 0), int(r["attempt"]))
    dyad_rows = {(d["dyad_id"], int(d.get("attempt", 1))): d for d in raw["dyads"]}
    any_row: dict[str, dict] = {}
    for d in raw["input-dyads"] + raw["dyads"]:
        any_row.setdefault(d["dyad_id"], d)
    order = [d["dyad_id"] for d in raw["input-dyads"]]
    order += sorted({d for d in list(seen_status) + [k[0] for k in dyad_rows]} - set(order))

    dyads = []
    for dyad_id in order:
        att = complete.get(dyad_id)
        src = dyad_rows.get((dyad_id, att)) if att else None
        rec = {"run_id": run_id, "arm": arm_label(manifest), "dyad_id": dyad_id, "attempt": att,
               "max_attempt": seen_status.get(dyad_id), **_condition_fields(src or any_row.get(dyad_id, {})),
               "in_itt": True, "in_adherence": None, "excluded": []}
        rec["cell"] = cell_of(rec)
        dyads.append(rec)
    want = (not include_unconstrained, float(survey_temperature), int(survey_n_predict))
    run = Run(root, run_id, manifest, arm_label(manifest), dyads, [], [], [], raw, metric=metric,
              survey_key=want)

    keep = {(d["dyad_id"], d["attempt"]) for d in dyads if d["attempt"]}
    key = lambda r: (r["dyad_id"], int(r.get("attempt", 1)))  # noqa: E731
    run.turns = sorted((r for r in raw["turns"] if key(r) in keep),
                       key=lambda r: (r["dyad_id"], r["turn"], 0 if r["agent"] == SEEKER else 1))
    passed = [r for r in raw["surveys"] if key(r) in keep and r.get("origin", "run") == origin]
    run.surveys = [r for r in passed if survey_key(r) == want]
    # Rows of the other schema setting are the unconstrained check (or, with include_unconstrained, the
    # constrained pass): never mixed into the outcomes, noted. Rows of the same schema setting at another
    # temperature or n_predict are another measurement: set aside and logged per dyad (survey_settings).
    other_schema = sum(1 for r in passed if survey_key(r)[0] != want[0])
    if other_schema:
        what = "constrained" if include_unconstrained else "unconstrained (schema false)"
        run.notes.append(f"{other_schema} {what} {origin} survey rows kept out of the outcomes"
                         + ("" if include_unconstrained else "; analysis.rates compares them"))
    off: dict[str, dict[tuple, int]] = {}
    for r in passed:
        k = survey_key(r)
        if k[0] == want[0] and k != want:
            off.setdefault(r["dyad_id"], {})
            off[r["dyad_id"]][k[1:]] = off[r["dyad_id"]].get(k[1:], 0) + 1
    run.scores = [r for r in raw["scores"] if key(r) in keep]
    unscoped = sum(1 for r in run.scores if not r.get("scope"))
    if unscoped:
        run.notes.append(f"{unscoped} score rows carry no scope (written before it was recorded); their "
                         "scope is inferred from the turn cadence")
    turns_by: dict[str, list] = {}
    for r in run.turns:
        turns_by.setdefault(r["dyad_id"], []).append(r)
    survey_err = {r["dyad_id"] for r in run.surveys if r.get("error")}

    for d in dyads:
        if d["attempt"] is None:
            if d["max_attempt"] is None:
                run.exclude(d, "not_run", "no status row")
            else:
                run.exclude(d, "incomplete", f"latest attempt {d['max_attempt']} not complete")
            continue
        rows = turns_by.get(d["dyad_id"], [])
        errs = [r for r in rows if r.get("error") or r.get("finish_reason") == "error"]
        if errs or d["dyad_id"] in survey_err:
            where = f"turn {errs[0]['turn']} {errs[0]['agent']}" if errs else "survey"
            run.exclude(d, "error_rows", f"error row at {where}")
        ok = [r for r in rows if not (r.get("error") or r.get("finish_reason") == "error")]
        n_turns = int(d.get("n_turns") or 0)
        if n_turns and (len(ok) < 2 * n_turns or max((r["turn"] for r in ok), default=0) < n_turns):
            run.exclude(d, "short_dialogue", f"{len(ok)} of {2 * n_turns} messages")
        trunc = [r for r in rows if r.get("truncated") is True]
        if trunc:
            run.exclude(d, "truncated", f"turn {trunc[0]['turn']} {trunc[0]['agent']} ran out of context")
        for (temp, n_pred), n in sorted(off.get(d["dyad_id"], {}).items()):
            run.exclude(d, "survey_settings", f"{n} {origin} survey rows at temperature {temp}, n_predict "
                        f"{n_pred} set aside (analysed: temperature {want[1]}, n_predict {want[2]})")

    _adherence(run, judge, metric)
    _flags(run, threshold, run_length)
    return run


def _adherence(run: Run, judge_prefix: str | None, metric: str) -> None:
    """Attach mean seeker adherence (the chosen judge, `metric`, main-scope rows) to each treated ITT dyad
    and count judge failures. Controls get none: the bare control has no persona to adhere to."""
    main = run.scores_in("main")
    run.judge = pick_judge(main, metric, judge_prefix)
    if run.judge is None:
        if run.scores:
            run.notes.append(f"no main-scope scores for {metric}; adherence analyses skipped")
        return
    mine = [s for s in main if s.get("judge_sha256") == run.judge]
    targets: dict[tuple, list[dict]] = {}
    for s in mine:
        targets.setdefault((s["dyad_id"], s["turn"], s["agent"], s["metric"]), []).append(s)
    failed: dict[str, list] = {}
    for k, rs in targets.items():
        if all(r.get("error") for r in rs):
            failed.setdefault(k[0], []).append(k)
    by_dyad: dict[str, list[dict]] = {}
    for s in mine:
        if (s.get("metric") == metric and s.get("agent") == SEEKER and not s.get("error")
                and s.get("score") is not None):
            by_dyad.setdefault(s["dyad_id"], []).append(s)
    n_control = sum(1 for d in run.dyads if d["control"] and d["dyad_id"] in by_dyad)
    if n_control:
        run.notes.append(f"{n_control} control dyads carry adherence scores; dropped (adherence is "
                         "undefined "
                         "for the bare control)")
    for d in run.dyads:
        if not d["in_itt"] or d["control"]:
            continue
        d["in_adherence"] = True
        rs = sorted(by_dyad.get(d["dyad_id"], []), key=lambda s: s["turn"])
        d["adherence_n"] = len(rs)
        d["adherence_mean"] = sum(s["score"] for s in rs) / len(rs) if rs else None
        for k in failed.get(d["dyad_id"], []):
            run.exclude(d, "judge_failure", f"turn {k[1]} {k[2]} {k[3]}: only error rows")
        if not rs:
            run.exclude(d, "unscored", f"no {metric} score from judge {run.judge[:12]}")


def _flags(run: Run, threshold: float | None, run_length: int) -> None:
    """Attach `flagged` to treated ITT dyads: computed with `threshold` when given, else read from
    flags.jsonl (the chosen judge and metric). Control rows are ignored (F10)."""
    flags: dict[str, dict] = {}
    if threshold is not None:
        if run.judge is None:
            return
        # The registered rule runs on main-scope rows only (PAP §5, red-team M9): a pilot pass scores every
        # turn, and "3 consecutive scored turns" would mean something else for the dyads it covered. A row
        # without a scope counts as main when it is on the main cadence (score_scope).
        main = run.scores_in("main")
        for (dyad_id, attempt), f in flag_dialogues(main, threshold, metric=run.metric,
                                                    run_length=run_length, judge_sha256=run.judge).items():
            flags[dyad_id] = {**f, "attempt": attempt}
        run.notes.append(f"flags computed: {run.metric} < {threshold} on {run_length} consecutive scored "
                         "turns")
    else:
        rows = [f for f in run.raw["flags"] if f.get("metric", run.metric) == run.metric
                and f.get("scope", "main") == "main"
                and (run.judge is None or f.get("judge_sha256") in (None, run.judge))]

        for f in rows:
            flags[f["dyad_id"]] = f
        if rows:
            t = sorted({f.get("threshold") for f in rows}, key=str)
            run.notes.append(f"flags read from flags.jsonl (threshold {t})")
    for d in run.dyads:
        f = flags.get(d["dyad_id"])
        if d["control"] or not d["in_itt"] or f is None or f.get("attempt", d["attempt"]) != d["attempt"]:
            d["flagged"] = None
        else:
            d["flagged"] = bool(f["flagged"])


def incompleteness_test(run: Run, by: str = "ideology") -> dict:
    """Chi-square test of technical incompleteness (any dyad-scope exclusion) against `by` (PAP §6)."""
    cats = sorted({str(d[by]) for d in run.dyads}, key=lambda c: (LEVELS + (CONTROL,)).index(c)
                  if c in LEVELS + (CONTROL,) else 99)
    bad = [sum(1 for d in run.dyads if str(d[by]) == c and not d["in_itt"]) for c in cats]
    good = [sum(1 for d in run.dyads if str(d[by]) == c and d["in_itt"]) for c in cats]
    return {**stats.chi2_independence([bad, good]), "by": by, "categories": cats, "incomplete": bad,
            "complete": good}


def exclusion_summary(run: Run) -> dict:
    """Counts by reason overall, by ideology and by cell, the sample sizes that remain, and the
    incompleteness test."""
    by_reason: dict[str, int] = {}
    by_ideology: dict[str, dict[str, int]] = {}
    by_cell: dict[str, dict[str, int]] = {}
    for e in run.exclusions:
        r = e["reason"] if e["index"] is None else f"{e['reason']}:{e['index']}"
        by_reason[r] = by_reason.get(r, 0) + 1
        by_ideology.setdefault(r, {})
        by_ideology[r][str(e["ideology"])] = by_ideology[r].get(str(e["ideology"]), 0) + 1
        cell = cell_of(e)
        by_cell.setdefault(r, {})
        by_cell[r][cell] = by_cell[r].get(cell, 0) + 1
    planned: dict[str, int] = {}
    itt: dict[str, int] = {}
    for d in run.dyads:
        planned[str(d["ideology"])] = planned.get(str(d["ideology"]), 0) + 1
        if d["in_itt"]:
            itt[str(d["ideology"])] = itt.get(str(d["ideology"]), 0) + 1
    return {"run_id": run.run_id, "arm": run.arm, "planned": len(run.dyads),
            "itt": sum(d["in_itt"] for d in run.dyads), "planned_by_ideology": planned,
            "itt_by_ideology": itt,
            "attempt_gt_1": sum(1 for d in run.dyads if (d["attempt"] or 0) > 1),
            "by_reason": by_reason, "by_ideology": by_ideology, "by_cell": by_cell,
            "incompleteness_test": incompleteness_test(run), "judge": run.judge, "metric": run.metric,
            "notes": run.notes}


def exclusions_markdown(run: Run) -> str:
    """The exclusion log as markdown: counts by reason and ideology level, then by cell, then the test."""
    s = exclusion_summary(run)
    levels = [lv for lv in LEVELS + (CONTROL,) if lv in s["planned_by_ideology"]]
    levels += sorted(set(s["planned_by_ideology"]) - set(levels))
    out = [f"# Exclusions: {run.run_id} ({run.arm})", "",
           f"Planned dyads {s['planned']}; ITT sample {s['itt']}; analysed at attempt > 1: "
           f"{s['attempt_gt_1']}.",
           f"Adherence judge: {str(s['judge'])[:12]} ({s['metric']}).", ""]
    rows = [["planned"] + [s["planned_by_ideology"].get(lv, 0) for lv in levels] + [s["planned"]]]
    for r in sorted(s["by_ideology"], key=lambda r: (REASONS.index(r.split(":")[0]), r)):
        rows.append([f"{r} ({SCOPE[r.split(':')[0]]})"] + [s["by_ideology"][r].get(lv, 0) for lv in levels]
                    + [s["by_reason"][r]])
    rows.append(["ITT sample"] + [s["itt_by_ideology"].get(lv, 0) for lv in levels] + [s["itt"]])
    out += [md_table(["reason"] + levels + ["total"], rows), ""]
    if s["by_cell"]:
        cells = sorted({c for v in s["by_cell"].values() for c in v})
        rows = [[r] + [s["by_cell"][r].get(c, 0) for c in cells] for r in s["by_cell"]]
        out += ["By cell:", "", md_table(["reason"] + cells, rows), ""]
    t = s["incompleteness_test"]
    p = "n/a" if t["p"] is None else f"{t['p']:.3f}"
    chi2 = "n/a" if t["chi2"] is None else f"{t['chi2']:.2f}"
    lee = " (below 0.05: estimate reports Lee bounds)" if t["p"] is not None and t["p"] < 0.05 else ""
    out.append(f"Incompleteness against ideology: chi2 = {chi2}, df = {t['df']}, p = {p}{lee}")
    out += [""] + [f"- {n}" for n in s["notes"]]
    return "\n".join(out) + "\n"


def unconstrained_surveys(run: Run, phase: str | None = None) -> list[dict]:
    """The unconstrained check's survey rows (`schema: false`, from `survey --no-schema --sample N`) of the
    analysed attempts, any origin and settings, error rows included. They never enter `run.surveys`
    unless the run was loaded with include_unconstrained; this is how analysis.rates compares them."""
    keep = {(d["dyad_id"], d["attempt"]) for d in run.dyads if d["attempt"]}
    return [r for r in run.raw["surveys"] if (r["dyad_id"], int(r.get("attempt", 1))) in keep
            and r.get("schema", True) is False and (phase is None or r.get("phase") == phase)]


def constrained_answers(run: Run) -> dict:
    """{(dyad_id, attempt, phase, item_id): answer} of the run's own constrained pass (origin run, the
    schema, the instrument's settings), the reference the unconstrained check is compared with. The last
    non-error row wins."""
    keep = {(d["dyad_id"], d["attempt"]) for d in run.dyads if d["attempt"]}
    out = {}
    for r in run.raw["surveys"]:
        k = (r["dyad_id"], int(r.get("attempt", 1)))
        if (k in keep and r.get("origin", "run") == "run" and not r.get("error")
                and survey_key(r) == (True, SURVEY_TEMPERATURE, SURVEY_N_PREDICT)):
            out[k + (r["phase"], r["item_id"])] = None if answer_truncated(r) else r.get("answer")
    return out


@dataclass
class Baseline:
    """One arm's no-dialogue baseline run (`harness.run baseline`): the pre battery administered K times to
    the mentor in an empty context. `answers` maps each administration to {item_id: answer or None}; an
    item with only error rows, or a reply cut off (answer_truncated), is None. `settings` is manifest.json's
    `baseline` block (phase, k, temperature, top_p, n_predict, schema)."""
    root: Path
    run_id: str
    manifest: dict
    arm: str
    settings: dict
    answers: dict[int, dict]
    counts: dict

    def distribution(self) -> dict[str, dict]:
        """{item_id: {"n": answered, "missing": administrations without an answer, "counts": {answer: n},
        "mode": the most frequent answer}} over the administrations."""
        items: dict[str, dict] = {}
        for ans in self.answers.values():
            for item, v in ans.items():
                e = items.setdefault(item, {"n": 0, "missing": 0, "counts": {}})
                if v is None:
                    e["missing"] += 1
                else:
                    e["n"] += 1
                    e["counts"][v] = e["counts"].get(v, 0) + 1
        for e in items.values():
            e["counts"] = dict(sorted(e["counts"].items()))
            e["mode"] = max(e["counts"], key=lambda v: (e["counts"][v], -v)) if e["counts"] else None
        return items


def load_baseline(run_dir) -> Baseline:
    """Read a baseline run directory (manifest.json of kind `baseline`, baseline.jsonl). Error rows are
    dropped (a re-run fills them in: the last non-error row per administration and item wins); an item
    that has only error rows is a missing answer, as is one whose reply was cut off."""
    root = Path(run_dir)
    if not (root / "manifest.json").exists():
        raise FileNotFoundError(f"{root}/manifest.json not found: not a run directory")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "baseline":
        raise ValueError(f"{root} is not a baseline run (manifest.json kind {manifest.get('kind')!r})")
    rows = read_jsonl(root / "baseline.jsonl")
    answers: dict[int, dict] = {}
    counts = {"rows": len(rows), "error": 0, "truncated": 0, "null": 0}
    for r in rows:
        slot = answers.setdefault(int(r["administration"]), {})
        if r.get("error"):
            counts["error"] += 1
            slot.setdefault(r["item_id"], None)
            continue
        if answer_truncated(r):
            counts["truncated"] += 1
            slot[r["item_id"]] = None
            continue
        if r.get("answer") is None:
            counts["null"] += 1
        slot[r["item_id"]] = r.get("answer")
    counts["administrations"] = len(answers)
    return Baseline(root, manifest.get("run_id") or root.name, manifest, arm_label(manifest),
                    manifest.get("baseline") or {}, dict(sorted(answers.items())), counts)


def baseline_distributions(run_dirs) -> dict[str, dict]:
    """{arm: {item_id: distribution}} over one baseline run directory per arm (Baseline.distribution)."""
    out: dict[str, dict] = {}
    for rd in run_dirs:
        b = load_baseline(rd)
        if b.arm in out:
            raise ValueError(f"two baseline runs for arm {b.arm!r}")
        out[b.arm] = b.distribution()
    return out


def default_out(run_dir) -> Path:
    """Where the analysis writes by default: data/<run_id>/analysis/, git-ignored with the rows."""
    return Path(run_dir) / "analysis"


def add_common_args(p: argparse.ArgumentParser, multi: bool = False) -> None:
    """The arguments every analysis CLI shares."""
    if multi:
        p.add_argument("--run-dir", action="append", required=True, help="a run directory; repeat for arms")
    else:
        p.add_argument("--run-dir", required=True, help="data/<run_id>")
    p.add_argument("--judge", default=None, help="judge_sha256 prefix, when more than one judge scored")
    p.add_argument("--metric", default="prompt_to_line", choices=ADHERENCE_METRICS, help="adherence metric")
    p.add_argument("--threshold", type=float, default=None,
                   help="compute flags at this threshold instead of reading flags.jsonl")
    p.add_argument("--run-length", type=int, default=3)
    p.add_argument("--origin", default="run", choices=("run", "readministered"), help="which survey pass")
    p.add_argument("--survey-temperature", type=float, default=SURVEY_TEMPERATURE,
                   help=f"the survey pass's temperature (default {SURVEY_TEMPERATURE}); rows of the "
                        "origin at other settings are set aside and logged")
    p.add_argument("--survey-n-predict", type=int, default=SURVEY_N_PREDICT,
                   help=f"the survey pass's n_predict (default {SURVEY_N_PREDICT})")
    p.add_argument("--include-unconstrained", action="store_true",
                   help="analyse the unconstrained check's rows (schema false) instead of the constrained "
                        "pass; they cover only the sampled dyads")
    p.add_argument("--out", default=None, help="output directory (default <first run-dir>/analysis)")


def load_from_args(a, run_dir=None) -> Run:
    return load_run(run_dir or a.run_dir, judge=a.judge, metric=a.metric, origin=a.origin,
                    threshold=a.threshold, run_length=a.run_length,
                    include_unconstrained=a.include_unconstrained, survey_temperature=a.survey_temperature,
                    survey_n_predict=a.survey_n_predict)



def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m analysis.load", description=__doc__.split("\n\n")[0])
    add_common_args(p)
    a = p.parse_args(argv)
    try:
        run = load_from_args(a)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    out = Path(a.out) if a.out else default_out(a.run_dir)
    md = exclusions_markdown(run)
    out.mkdir(parents=True, exist_ok=True)
    (out / "exclusions.md").write_text(md, encoding="utf-8")
    write_json(out / "exclusions.json", exclusion_summary(run))
    write_jsonl(out / "exclusions.jsonl", run.exclusions)
    write_csv(out / "dyads.csv", run.dyads)
    print(md)
    print(f"wrote {out}/exclusions.md, exclusions.json, exclusions.jsonl, dyads.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
