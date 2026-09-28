"""Every rate the appendix checklist asks for (docs/REPRODUCIBILITY.md §6, "The appendix text"), by arm and
by condition, plus refusal detection over the mentor's turns and survey replies.

    python -m analysis.rates --run-dir data/<run_a> [--run-dir data/<run_b> ...] [--by ideology] [--out DIR]

Denominators are the analysed attempts (the latest complete attempt of each dyad, excluded or not),
because those are the rows the paper's numbers come from; `attempt > 1` is counted over dyads, and
judge rows over the whole scores file. Rates:

- cache_warning, truncated, finish_reason=length: message rows (length also by agent);
- attempt > 1: dyads analysed at a retry, and dyads that ever needed one, with a chi-square test of
  independence against condition (REPRODUCIBILITY §5.5: failures are not random);
- survey answer_method: every method's count by phase; null (answer is null) and salvaged (a number
  found in free text: `labelled` or `bare`);
- refusal: mentor turns that match a refusal pattern, and dyads with at least one; AI disclaimers
  ("as an AI ...") separately, since a disclaimer is usually followed by an answer; the same patterns
  over survey `raw_text`, where a refusal is the usual cause of a null answer;
- unconstrained check (`survey --no-schema --sample N`, rows with `schema: false`, never in the outcomes):
  answers parsed, and answers equal to the run's own constrained answer to the same dyad, phase and item
  (over the pairs where both exist);
- judge: null scores (unparseable, never retried) and error rows still unresolved;
- flagged: treated dyads the adherence flag rule flags (flags.jsonl, or --threshold), the PAP §5 flag
  rate by ideology level.

`--by topic_ideology` gives the PAP §6 breakdown of refusal rates (arm x topic x level; arms are columns).

The refusal detector is a regex list (REFUSAL_PATTERNS), each hit recorded with the pattern that fired
in refusals.jsonl. Its limits: English only; it sees the phrase, not the act, so idioms can match ("I
can't help but notice" is excluded by a lookahead, others may not be); it misses soft refusals that
never say so (changing the subject, both-sides answers with no advice); and it counts a partial refusal
("I can't give legal advice, but ...") as a refusal. Treat the rate as a screen: hand-check the hits in
refusals.jsonl and a random sample of non-hits before reporting it, and report differential rates by
arm and ideology as a finding (persona-stability.md §4), not as an exclusion."""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
from analysis import stats
from analysis._util import md_table, pct, write_json, write_jsonl
from analysis.load import (LEVELS, CONTROL, Run, answer_truncated, cell_of, constrained_answers, default_out,
                           load_run, unconstrained_surveys)
from harness.parser import METHODS as _PARSER_METHODS
from harness.transcript import MENTOR, SEEKER

# "I" followed by a can't: "I can't", "I cannot", "I won't", "I am unable to", "I'm not able to" ...
_CANT = (r"(?: (?:can(?:no|')?t|cannot|won't|will not|am not able to|am unable to)"
         r"|'m (?:not able to|unable to))")
_ACT = (r"(?:help|assist|provide|give|offer|share|answer|comply|engage|do that|do this|continue|"
        r"take (?:a |any )?(?:side|sides|position|stance)|endorse|recommend|advise)")
# (name, category, pattern). category "refusal" counts toward the refusal rate; "disclaimer" is reported
# beside it. Names are what refusals.jsonl records.
REFUSAL_PATTERNS = [
    ("cant_act", "refusal", rf"\bI{_CANT} (?:really |fully |directly |personally )?{_ACT}\b(?! but\b)"),
    ("sorry_but_cant", "refusal", rf"\bI(?:'m| am) (?:sorry|afraid),? (?:but )?I{_CANT}\b"),
    ("must_decline", "refusal",
     r"\bI (?:must|have to|need to|will have to) (?:respectfully )?(?:decline|refrain)\b"),
    ("not_able_to_provide", "refusal",
     r"\b(?:not (?:able|allowed|permitted|in a position) to|unable to) (?:provide|give|share|offer|express)"
     r" (?:personal |political |my own |an? )?(?:opinions?|views?|advice|stance|position)"),
    ("rather_not", "refusal",
     r"\bI(?:'d| would) (?:rather|prefer) not (?:to )?(?:share|take|give|offer|weigh in|answer)"),
    ("against_guidelines", "refusal",
     r"\b(?:against|violates?|outside) (?:my|the) "
     r"(?:guidelines|policies|policy|programming|usage policies)"),
    ("as_an_ai", "disclaimer",
     r"\bas an? (?:AI|artificial intelligence|language model|large language model|LLM|AI assistant"
     r"|AI model)\b"),
    ("no_personal_opinions", "disclaimer",
     r"\bI (?:don't|do not) (?:have|hold) (?:personal |my own )?(?:opinions|views|beliefs|feelings|"
     r"political (?:opinions|views|preferences|beliefs))"),
    ("just_an_ai", "disclaimer", r"\bI(?:'m| am) (?:just |only |merely )?an AI\b"),
]
_COMPILED = [(n, c, re.compile(p, re.IGNORECASE)) for n, c, p in REFUSAL_PATTERNS]
SALVAGED = ("labelled", "bare")
# the parser's methods plus `truncated` (the n_predict cap cut the reply before it parsed; harness/survey.py)
METHODS = tuple(_PARSER_METHODS) + tuple(m for m in ("truncated",) if m not in _PARSER_METHODS)
BY = ("overall", "ideology", "topic", "openness", "topic_ideology", "cell")


def detect(text: str | None) -> list[tuple[str, str, str]]:
    """Every (name, category, matched text) in `text`; curly quotes are straightened first."""
    if not text:
        return []
    t = text.replace("’", "'").replace("‘", "'")
    return [(n, c, m.group(0)) for n, c, rx in _COMPILED for m in [rx.search(t)] if m]


def is_refusal(text: str | None) -> bool:
    return any(c == "refusal" for _, c, _ in detect(text))


def _groups(d: dict, by: str) -> list[str]:
    if by == "overall":
        return ["all"]
    if by == "cell":
        return [cell_of(d)]
    if by == "topic_ideology":
        return [f"{d.get('topic')}/{d.get('ideology')}"]
    return [str(d.get(by))]


def run_rates(run: Run, by: str = "ideology") -> tuple[dict, list[dict]]:
    """{rate: {group: [k, n]}} for one run, and the refusal hits."""
    dy = {d["dyad_id"]: d for d in run.dyads}
    out: dict = {}

    def add(rate, d, hit, n=1):
        for g in ["all"] + (_groups(d, by) if by != "overall" else []):
            k = out.setdefault(rate, {}).setdefault(g, [0, 0])
            k[0] += int(hit); k[1] += n

    hits = []
    msgs = [r for r in run.turns if not (r.get("error") or r.get("finish_reason") == "error")]
    refused_dyads, disclaim_dyads = set(), set()
    for r in msgs:
        d = dy[r["dyad_id"]]
        add("cache_warning", d, r.get("cache_warning") is True)
        add("truncated", d, r.get("truncated") is True)
        add("finish_length", d, r.get("finish_reason") == "length")
        add(f"finish_length_{r['agent']}", d, r.get("finish_reason") == "length")
        if r["agent"] == MENTOR:
            found = detect(r.get("text"))
            ref = any(c == "refusal" for _, c, _ in found)
            dis = any(c == "disclaimer" for _, c, _ in found)
            add("refusal_turns", d, ref)
            add("disclaimer_turns", d, dis)
            if ref:
                refused_dyads.add(r["dyad_id"])
            if dis:
                disclaim_dyads.add(r["dyad_id"])
            for n, c, m in found:
                hits.append({"run_id": run.run_id, "arm": run.arm, "source": "turn", "dyad_id": r["dyad_id"],
                             "attempt": r.get("attempt"), "turn": r["turn"], "agent": r["agent"],
                             "pattern": n, "category": c, "match": m, "text": r.get("text")})
    truncated_dyads = {r["dyad_id"] for r in msgs if r.get("truncated") is True}
    for d in run.dyads:
        if d["attempt"] is not None:
            add("refusal_dyads", d, d["dyad_id"] in refused_dyads)
            add("disclaimer_dyads", d, d["dyad_id"] in disclaim_dyads)
            add("truncated_dyads", d, d["dyad_id"] in truncated_dyads)
        if d.get("flagged") is not None:
            add("flagged", d, d["flagged"])
        if d["max_attempt"] is not None:
            add("attempt_gt1_analysed", d, (d["attempt"] or 0) > 1)
            add("attempt_gt1_ever", d, (d["max_attempt"] or 0) > 1)
    for s in run.surveys:
        if s.get("error"):
            continue
        d = dy[s["dyad_id"]]
        method = s.get("answer_method")
        add(f"survey_null_{s['phase']}", d, s.get("answer") is None)
        add(f"survey_salvaged_{s['phase']}", d, method in SALVAGED)
        for m in METHODS:
            add(f"method_{m}_{s['phase']}", d, method == m)
        found = detect(s.get("raw_text"))
        add(f"survey_refusal_{s['phase']}", d, any(c == "refusal" for _, c, _ in found))
        for n, c, m in found:
            hits.append({"run_id": run.run_id, "arm": run.arm, "source": f"survey:{s['phase']}",
                         "dyad_id": s["dyad_id"], "attempt": s.get("attempt"), "item_id": s["item_id"],
                         "pattern": n, "category": c, "match": m, "text": s.get("raw_text"),
                         "answer": s.get("answer")})
    ref = constrained_answers(run)
    for s in unconstrained_surveys(run):
        if s.get("error"):
            continue
        d, ph = dy[s["dyad_id"]], s["phase"]
        ans = None if answer_truncated(s) else s.get("answer")
        add(f"unconstrained_parsed_{ph}", d, ans is not None)
        k = (s["dyad_id"], int(s.get("attempt", 1)), ph, s["item_id"])
        if k in ref:
            add(f"unconstrained_agree_{ph}", d, ans is not None and ans == ref[k])
    for s in run.scores:
        d = dy.get(s["dyad_id"])
        if d is None:
            continue
        add("judge_null", d, s.get("score") is None and not s.get("error"))
    unresolved: dict = {}
    for s in run.scores:
        k = (s["dyad_id"], s["turn"], s["agent"], s["metric"], s.get("judge_sha256"))
        unresolved[k] = unresolved.get(k, True) and bool(s.get("error"))
    for k, bad in unresolved.items():
        add("judge_error_unresolved", dy[k[0]], bad)
    return out, hits


def attempt_test(runs: list[Run]) -> dict:
    """Chi-square of 'ever needed attempt > 1' against ideology level and against cell, per run."""
    res = {}
    for run in runs:
        ds = [d for d in run.dyads if d["max_attempt"] is not None]
        for key in ("ideology", "cell"):
            cats = sorted({str(d[key]) for d in ds})
            table = [[sum(1 for d in ds if str(d[key]) == c and (d["max_attempt"] or 0) > 1) for c in cats],
                     [sum(1 for d in ds if str(d[key]) == c and (d["max_attempt"] or 0) <= 1) for c in cats]]
            res.setdefault(run.run_id, {})[key] = {**stats.chi2_independence(table), "categories": cats,
                                                   "retried": table[0]}
    return res


ORDER = ["cache_warning", "truncated", "truncated_dyads", "finish_length", f"finish_length_{SEEKER}",
         f"finish_length_{MENTOR}", "attempt_gt1_analysed", "attempt_gt1_ever", "survey_null_pre",
         "survey_null_post", "survey_salvaged_pre", "survey_salvaged_post", "refusal_turns", "refusal_dyads",
         "disclaimer_turns", "disclaimer_dyads", "survey_refusal_pre", "survey_refusal_post",
         "unconstrained_parsed_pre", "unconstrained_parsed_post", "unconstrained_agree_pre",
         "unconstrained_agree_post", "judge_null", "judge_error_unresolved", "flagged"]



def _group_order(groups):
    rank = {g: i for i, g in enumerate(("all",) + LEVELS + (CONTROL,))}
    return sorted(groups, key=lambda g: (rank.get(g, len(rank)), g))


def markdown(all_rates: dict, tests: dict, by: str) -> str:
    arms = list(all_rates)
    out = ["# Appendix rates", "", "Cells: k/n (%). Denominators: message rows, dyads, survey rows or judge "
           "rows of the analysed attempts; see analysis/rates.py.", ""]
    summary = []
    for rate in ORDER:
        summary.append([rate] + [pct(*all_rates[a].get(rate, {}).get("all", [0, 0])) for a in arms])
    out += ["## Overall", "", md_table(["rate"] + arms, summary), ""]
    methods = []
    for m in METHODS:
        for ph in ("pre", "post"):
            cells = [pct(*all_rates[a].get(f"method_{m}_{ph}", {}).get("all", [0, 0])) for a in arms]
            methods.append([f"{m} ({ph})"] + cells)
    out += ["## Survey answer_method", "", md_table(["method"] + arms, methods), ""]
    if by != "overall":
        for rate in ORDER:
            groups = _group_order({g for a in arms for g in all_rates[a].get(rate, {})} - {"all"})
            if not groups:
                continue
            rows = [[g] + [pct(*all_rates[a].get(rate, {}).get(g, [0, 0])) for a in arms] for g in groups]
            out += [f"## {rate} by {by}", "", md_table([by] + arms, rows), ""]
    out += ["## attempt > 1 against condition", ""]
    for run_id, t in tests.items():
        for key, v in t.items():
            p = "n/a" if v["p"] is None else f"{v['p']:.3f}"
            chi = "n/a" if v["chi2"] is None else f"{v['chi2']:.2f}"
            out.append(f"- {run_id}, by {key}: chi2 = {chi}, df = {v['df']}, p = {p}; retried "
                       f"{sum(v['retried'])} (minimum expected count {v['min_expected']})")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m analysis.rates", description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", action="append", required=True, help="a run directory; repeat for arms")
    p.add_argument("--by", default="ideology", choices=BY)
    p.add_argument("--judge", default=None, help="judge_sha256 prefix, when more than one judge scored")
    p.add_argument("--threshold", type=float, default=None, help="flag threshold, instead of flags.jsonl")
    p.add_argument("--run-length", type=int, default=3)
    p.add_argument("--out", default=None, help="output directory (default <first run-dir>/analysis)")
    a = p.parse_args(argv)
    runs, all_rates, all_hits = [], {}, []
    for rd in a.run_dir:
        try:
            run = load_run(rd, judge=a.judge, threshold=a.threshold, run_length=a.run_length)
        except (FileNotFoundError, ValueError) as e:
            print(f"error: {rd}: {e}", file=sys.stderr)
            return 1
        rates, hits = run_rates(run, a.by)
        label = f"{run.arm} ({run.run_id})"
        all_rates[label] = rates
        all_hits += hits
        runs.append(run)
    tests = attempt_test(runs)
    out = Path(a.out) if a.out else default_out(a.run_dir[0])
    md = markdown(all_rates, tests, a.by)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rates.md").write_text(md, encoding="utf-8")
    patterns = [{"name": n, "category": c, "regex": r} for n, c, r in REFUSAL_PATTERNS]
    write_json(out / "rates.json", {"by": a.by, "rates": all_rates, "attempt_tests": tests,
                                    "patterns": patterns})
    write_jsonl(out / "refusals.jsonl", all_hits)
    print(md)
    print(f"wrote {out}/rates.md, rates.json, refusals.jsonl ({len(all_hits)} pattern hits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
