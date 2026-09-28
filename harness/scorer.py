"""Offline scoring with a judge model: seeker adherence and mentor stance per logged turn, written to
scores.jsonl; plus the adherence flag rule (flags.jsonl) and cross-judge agreement, both computed from it."""
from __future__ import annotations
import dataclasses
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from harness.client import ServerError
from harness.dialogue import AgentHandle, GenSettings
from harness.log import JsonlWriter, RunPaths, derive_seed, now_iso, read_jsonl, resume_index, sha256_text
from harness.templates import render
from harness.transcript import SEEKER, MENTOR, message_order

METRICS = {"prompt_to_line": SEEKER, "line_to_line": SEEKER, "alignment": MENTOR}
# pilot: every turn, both agents. main: seeker adherence on the turn-4 cadence plus the final turn.
# stance: the mentor's alignment metric on the same cadence -- the turn-level DV -- meant to be run by
# two different judges on a subsample so cross-judge agreement is reportable (persona-stability §3).
SCOPES = ("pilot", "main", "stance")
MAIN_CADENCE = 4

# Model families, matched against the GGUF file name, then its directory, then the server alias,
# lower-cased. The judge may never share the mentor's family (docs/decisions/persona-stability.md §3): the
# mentor's stance score is the DV, and a same-family judge is both self-favouring and likely to share its
# political priors. A model none of these match (an ollama blob, sha256-...) needs `family` in its config
# block; `score` refuses an unknown judge or mentor family.
MODEL_FAMILIES = (
    ("gpt-oss", r"gpt[-_]?oss"), ("qwen", r"qwen|qwq"), ("olmo", r"olmo"), ("glm", r"\bglm|(^|[^a-z])glm"),
    ("gemma", r"gemma"), ("llama", r"llama"), ("mistral", r"mistral|mixtral|ministral|magistral"),
    ("phi", r"(^|[^a-z])phi[-_ ]?\d"), ("deepseek", r"deepseek"),
)


def model_family(name: str | None) -> str | None:
    """The family slug of a model, from its GGUF path (the file name, then the directory it is in) or its
    alias; None when no pattern matches. Unknown is unknown: the caller must not treat None as 'different
    family'."""
    if not name:
        return None
    parts = [p for p in str(name).replace("\\", "/").split("/") if p]
    for part in reversed(parts[-2:]):
        for family, pattern in MODEL_FAMILIES:
            if re.search(pattern, part.lower()):
                return family
    return None


def declared_family(value: str | None) -> str | None:
    """A `family` given in the config, as a slug: a known family's name maps to its slug ("Qwen3" -> qwen),
    anything else is kept lower-cased. None when not given."""
    if value is None or not str(value).strip():
        return None
    return model_family(str(value)) or str(value).strip().lower()
# Fixed rather than configurable, and written into judge-*.json with the prompt text below.
JUDGE_N_PREDICT = 160
JUDGE_TEMPERATURE = 0.0

JUDGE_SYSTEM = ("You are a careful annotator for a study of LLM-to-LLM political dialogue. You score one line at a "
           "time on a 0.0 to 1.0 scale and give a one-sentence rationale. Respond only with JSON.")

JUDGE_TASKS = {
    "prompt_to_line": ("Does the LINE fit the PERSONA (backstory, values, stance anchors, openness)? "
                       "1.0 = fully in character, 0.0 = contradicts the persona."),
    "line_to_line": ("Is the LINE consistent with the speaker's own EARLIER LINES (positions, facts about themselves, tone)? "
                     "1.0 = fully consistent, 0.0 = contradicts them."),
    "alignment": ("How far does the LINE agree with the position the PERSONA holds on the TOPIC? "
                  "1.0 = fully agrees, 0.5 = neutral or balanced, 0.0 = opposes."),
}


def score_schema() -> dict:
    """The JSON schema the judge's reply must satisfy: a 0-1 `score` and a `rationale`, both required."""
    return {"type": "object",
            "properties": {"score": {"type": "number", "minimum": 0, "maximum": 1}, "rationale": {"type": "string"}},
            "required": ["score", "rationale"]}


def build_judge_messages(metric: str, persona_text: str, topic: str, line: str,
                         prior_own_lines: list[str], partner_line: str | None) -> list[dict]:
    """Build the system+user chat messages that ask the judge to score one line on the given metric."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}")
    parts = [f"TASK: {JUDGE_TASKS[metric]}", f"TOPIC: {topic}", f"PERSONA:\n{persona_text}"]
    if metric == "line_to_line":
        earlier = "\n".join(f"- {l}" for l in prior_own_lines) or "- (none yet)"
        parts.append(f"EARLIER LINES by the same speaker:\n{earlier}")
    if partner_line is not None:
        parts.append(f"PARTNER'S PRECEDING LINE:\n{partner_line}")
    parts.append(f"LINE to score:\n{line}")
    parts.append('Return JSON: {"score": <0.0-1.0>, "rationale": "<one sentence>"}')
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def subsample_dyads(dyad_ids, fraction: float, run_seed: int) -> set[str]:
    """A deterministic subsample of whole dyads: a dyad is in when the first 32 bits of
    sha256(run_seed|dyad_id) fall under `fraction`. Whole dyads, so the two judges see the same turns;
    deterministic, so a second `score` pass (or a second judge) picks exactly the same set."""
    if not 0 < fraction <= 1:
        raise ValueError(f"subsample must be in (0, 1], got {fraction}")
    return {d for d in dyad_ids if int(sha256_text(f"{run_seed}|{d}")[:8], 16) / 2 ** 32 < fraction}


def select_targets(turn_rows: list[dict], scope: str, subsample: float | None = None,
                   run_seed: int = 0) -> list[tuple[dict, str]]:
    """Pick which (turn row, metric) pairs to score. 'pilot': every non-error row and metric. 'main':
    seeker rows on the turn-4 cadence plus the dyad's final turn, both seeker metrics. 'stance': mentor
    rows on that same cadence, the alignment metric only. `subsample` keeps that fraction of dyads
    (subsample_dyads). Raises ValueError for any other scope."""
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    rows = [r for r in turn_rows if r.get("finish_reason") != "error"]
    if subsample is not None:
        keep = subsample_dyads({r["dyad_id"] for r in rows}, subsample, run_seed)
        rows = [r for r in rows if r["dyad_id"] in keep]
    out = []
    if scope == "pilot":
        for r in rows:
            for metric, agent in METRICS.items():
                if r["agent"] == agent:
                    out.append((r, metric))
        return out
    agent_wanted = SEEKER if scope == "main" else MENTOR
    agent_rows = [r for r in rows if r["agent"] == agent_wanted]
    by_dyad: dict[tuple, int] = {}
    for r in agent_rows:
        key = (r["dyad_id"], r.get("attempt", 1))
        by_dyad[key] = max(by_dyad.get(key, 0), r["turn"])
    for r in agent_rows:
        # turns 4, 8, 12, ... plus the dyad's final turn (spec section 6). pilot scope scores every turn.
        if r["turn"] % MAIN_CADENCE == 0 or r["turn"] == by_dyad[(r["dyad_id"], r.get("attempt", 1))]:
            for metric, agent in METRICS.items():
                if agent == agent_wanted:
                    out.append((r, metric))
    return out


ADHERENCE_METRICS = tuple(m for m, a in METRICS.items() if a == SEEKER)
FLAG_RULE = "consecutive scored seeker turns"


def flag_dialogues(score_rows: list[dict], threshold: float, metric: str = "prompt_to_line",
                   run_length: int = 3, judge_sha256: str | None = None) -> dict[tuple[str, int], dict]:
    """The three-consecutive-turns flag rule (docs/decisions/persona-stability.md §3), in code. A dyad
    attempt is flagged when `metric` (a seeker adherence metric) is strictly under `threshold` on
    `run_length` consecutive SCORED seeker turns -- consecutive in the sequence of turns the judge scored,
    not in dialogue turns, because main scope scores the seeker on a turn-4 cadence. A null score (a
    judge reply that did not parse) is an unscored turn: it is counted in `unscored_turns` and neither
    extends nor breaks a run. Rows must come from one judge: pass `judge_sha256` when two have scored.

    `threshold` has no default on purpose: it is calibrated on the pilot's hand labels, never carried over
    from a paper whose metric is a rate rather than a per-turn score. Returns {(dyad_id, attempt): {...}}.
    Flagged dialogues are kept (ITT); the flag reports an instrument statistic and triggers the
    per-protocol sensitivity analysis."""
    if metric not in ADHERENCE_METRICS:
        raise ValueError(f"metric must be a seeker adherence metric {ADHERENCE_METRICS}, got {metric!r}")
    if run_length < 1:
        raise ValueError("run_length must be at least 1")
    rows = [r for r in score_rows if r.get("metric") == metric and r.get("agent") == SEEKER and not r.get("error")]
    judges = {r.get("judge_sha256") for r in rows}
    if judge_sha256 is not None:
        rows = [r for r in rows if r.get("judge_sha256") == judge_sha256]
    elif len(judges) > 1:
        raise ValueError(f"scores from {len(judges)} judges for {metric}; pass judge_sha256 to pick one")
    by_dyad: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        by_dyad.setdefault((r["dyad_id"], int(r.get("attempt", 1))), []).append(r)
    out = {}
    for key, rs in by_dyad.items():
        rs = sorted(rs, key=lambda r: r["turn"])
        scored = [r for r in rs if r.get("score") is not None]
        run = 0
        first_flag = None
        for r in scored:
            run = run + 1 if r["score"] < threshold else 0
            if run >= run_length and first_flag is None:
                first_flag = r["turn"]
        out[key] = {"flagged": first_flag is not None, "first_flag_turn": first_flag,
                    "scored_turns": len(scored), "unscored_turns": len(rs) - len(scored),
                    "turns_under": sum(1 for r in scored if r["score"] < threshold),
                    "min_score": min((r["score"] for r in scored), default=None),
                    "mean_score": (sum(r["score"] for r in scored) / len(scored)) if scored else None,
                    "final_turn": rs[-1]["turn"]}
    return out


def cross_judge_agreement(score_rows: list[dict], metric: str = "alignment") -> dict:
    """Agreement between every pair of judges that scored `metric`, on the (dyad, attempt, turn, agent)
    targets both scored with a non-null score: n, mean absolute difference, Pearson r, and the share of
    targets within 0.1. One judge on the outcome metric is a single point of failure; this is the number
    the paper reports beside it. Returns {"metric", "judges", "per_judge", "pairs"}."""
    import numpy as np
    by_judge: dict[str, dict[tuple, float]] = {}
    for r in score_rows:
        if r.get("metric") != metric or r.get("score") is None or r.get("error"):
            continue
        key = (r["dyad_id"], int(r.get("attempt", 1)), r["turn"], r["agent"])
        by_judge.setdefault(r["judge_sha256"], {})[key] = float(r["score"])
    judges = sorted(by_judge)
    per_judge = {j: {"n": len(v), "mean": float(np.mean(list(v.values()))) if v else None} for j, v in by_judge.items()}
    pairs = []
    for i, a in enumerate(judges):
        for b in judges[i + 1:]:
            shared = sorted(set(by_judge[a]) & set(by_judge[b]))
            xa = np.array([by_judge[a][k] for k in shared]); xb = np.array([by_judge[b][k] for k in shared])
            n = len(shared)
            r = None
            if n >= 2 and xa.std() > 0 and xb.std() > 0:
                r = float(np.corrcoef(xa, xb)[0, 1])
            pairs.append({"judges": [a, b], "n": n,
                          "mean_abs_diff": float(np.abs(xa - xb).mean()) if n else None,
                          "pearson_r": r,
                          "within_0.1": float((np.abs(xa - xb) <= 0.1 + 1e-9).mean()) if n else None})
    return {"metric": metric, "judges": judges, "per_judge": per_judge, "pairs": pairs}


def latest_complete_attempts(status_rows: list[dict]) -> dict[str, int]:
    """Return {dyad_id: attempt} for dyads whose latest attempt (per harness.log.resume_index) is complete."""
    return {d: v["attempt"] for d, v in resume_index(status_rows).items() if v["status"] == "complete"}


def parse_score(text: str) -> tuple[float | None, str]:
    """Parse a judge reply's JSON, returning (score, rationale) or (None, rationale-or-empty) if it's invalid."""
    try:
        d = json.loads(text)
        score = d.get("score")
        rationale = str(d.get("rationale") or "")   # a JSON null rationale must not land as "None"
    except (ValueError, AttributeError):
        return None, ""
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
        return None, rationale
    return float(score), rationale


class Scorer:
    """Scores a run's logged turns with a third model and appends the results to scores.jsonl."""
    def __init__(self, run_id: str, run_seed: int, judge: AgentHandle, scores_log: JsonlWriter,
                 settings: GenSettings, clock=now_iso, harness_commit: str = "", concurrency: int = 1):
        """Wire up the run identity, judge agent, output log, generation settings, clock, the harness
        commit that is doing the scoring (scoring can happen long after the run, from different code) and
        how many judge slots to use at once."""
        self.run_id, self.run_seed, self.judge = run_id, run_seed, judge
        self.scores_log, self.settings, self.clock = scores_log, settings, clock
        self.harness_commit = harness_commit
        self.errors = 0          # error rows written by the last score_run
        self.concurrency = max(1, int(concurrency))

    def check_independence(self, manifest: dict, mentor_family: str | None = None) -> None:
        """Raise ValueError for a judge that is the seeker or the mentor of this run, or of the mentor's
        model family (persona-stability §3). An unknown family on either side refuses too, since None is
        'unknown', not 'different': `mentor_family` (the score config's mentor.family) supplies the mentor's
        when manifest.json has none."""
        for role in ("seeker", "mentor"):
            if manifest.get(role, {}).get("model_sha256") == self.judge.model_sha256:
                raise ValueError(f"judge model is the same as the {role} model; pick a third model")
        mentor = manifest.get("mentor", {})
        mentor_family = (mentor.get("family") or declared_family(mentor_family)
                         or model_family(mentor.get("model_path")) or model_family(mentor.get("alias")))
        if not mentor_family:
            raise ValueError("the mentor's model family is unknown (manifest.json mentor.family is null); "
                             "set mentor.family in the config, so a same-family judge can be refused")
        if not self.judge.family:
            raise ValueError("the judge's model family is unknown from its GGUF name, directory or alias; "
                             "set judge.family in the config, so a same-family judge can be refused")
        if mentor_family == self.judge.family:
            raise ValueError(f"judge is from the mentor's model family ({mentor_family}); the mentor's stance "
                             "score is the outcome and a same-family judge is not independent of it")

    def score_run(self, paths: RunPaths, scope: str, manifest: dict, subsample: float | None = None,
                  mentor_family: str | None = None) -> int:
        """Score every not-yet-scored target for the run's complete dyads and return how many rows were
        written, after check_independence. Rows are done per judge, so a second judge scores the same
        targets. Targets go in (dyad_id, attempt, metric, turn) order, one dyad at a time per worker on
        its own judge slot, so consecutive prompts on a slot share the dyad's growing prefix and the judge
        prefills only what was added; `concurrency` workers run at once on slots 0..concurrency-1."""
        self.check_independence(manifest, mentor_family)
        self.errors = 0
        complete = latest_complete_attempts(read_jsonl(paths.status))
        dyads = {(d["dyad_id"], d["attempt"]): d for d in read_jsonl(paths.dyads)}
        done = {(s["dyad_id"], s["attempt"], s["turn"], s["agent"], s["metric"])
                for s in read_jsonl(paths.scores) if not s.get("error") and s.get("judge_sha256") == self.judge.model_sha256}
        turns = [r for r in read_jsonl(paths.turns) if complete.get(r["dyad_id"]) == r.get("attempt", 1)]
        by_dyad: dict[tuple, list[dict]] = {}
        for r in turns:
            by_dyad.setdefault((r["dyad_id"], r.get("attempt", 1)), []).append(r)
        histories: dict[tuple, list[dict]] = {k: sorted(v, key=message_order) for k, v in by_dyad.items()}
        # Position of each row in its dyad's history, keyed by object identity. Two rows can be
        # value-identical (a duplicate append after a crash), and a content-derived key would collapse
        # them and feed the first row its own line back as an "earlier" one.
        # See tests/test_scorer.py::test_score_run_duplicate_row_uses_position_not_value_equality.
        positions: dict[tuple, dict[int, int]] = {
            k: {id(r): i for i, r in enumerate(v)} for k, v in histories.items()}
        targets = [(row, metric) for row, metric in select_targets(turns, scope, subsample=subsample,
                                                                   run_seed=self.run_seed)
                   if (row["dyad_id"], row.get("attempt", 1), row["turn"], row["agent"], metric) not in done]
        # A stable sort: value-identical duplicate rows keep their file order.
        targets.sort(key=lambda t: (str(t[0]["dyad_id"]), t[0].get("attempt", 1), t[1], t[0]["turn"]))
        groups: dict[tuple, list] = {}
        for t in targets:
            groups.setdefault((t[0]["dyad_id"], t[0].get("attempt", 1)), []).append(t)

        def score_one(judge: AgentHandle, row: dict, metric: str) -> bool:
            """Score one target on `judge`'s slot and write its row; True when it wrote a non-error row."""
            key = (row["dyad_id"], row.get("attempt", 1), row["turn"], row["agent"], metric)
            # The dyad row carries the per-dyad seed the run used, so a score seed is derived from the same
            # (run_seed, dyad_seed) pair the dialogue was: look the row up before deriving the seed.
            spec = dyads.get(key[:2])
            seed = derive_seed(self.run_seed, int((spec or {}).get("seed", 0)), row["dyad_id"], key[1],
                               row["turn"], f"judge:{row['agent']}:{metric}")
            out = {"run_id": self.run_id, "dyad_id": row["dyad_id"], "attempt": key[1], "turn": row["turn"],
                   "agent": row["agent"], "metric": metric, "judge_sha256": judge.model_sha256,
                   "id_slot": judge.slot, "harness_commit": self.harness_commit, "seed": seed}
            if spec is None:
                out.update({"judge_prompt_sha256": "", "prompt_chars": None, "score": None, "rationale": "",
                            "raw_text": "", "error": "no dyads.jsonl row for this dyad/attempt", "ts": self.clock()})
                self.scores_log.write(out)
                return False
            history = histories[key[:2]]
            idx = positions[key[:2]][id(row)]
            prior_own = [r["text"] for r in history[:idx] if r["agent"] == row["agent"]]
            # The partner's most recent line before this one; None on the seeker's opening turn, when
            # the partner has not spoken.
            partner = next((r["text"] for r in reversed(history[:idx]) if r["agent"] != row["agent"]), None)
            topic = spec.get("condition", {}).get("topic", "")
            messages = build_judge_messages(metric, spec.get("persona_text", ""), topic, row["text"],
                                            prior_own, partner)
            prompt = render(judge.template, messages, now=self.settings.now,
                            enable_thinking=self.settings.enable_thinking)
            out["judge_prompt_sha256"] = sha256_text(prompt)
            out["prompt_chars"] = len(prompt)
            try:
                comp = judge.client.complete(prompt, id_slot=judge.slot, seed=seed, n_predict=JUDGE_N_PREDICT,
                                             temperature=JUDGE_TEMPERATURE, json_schema=score_schema(),
                                             cache_prompt=True)
            except ServerError as e:
                out.update({"score": None, "rationale": "", "raw_text": "", "error": str(e), "ts": self.clock()})
                self.scores_log.write(out)
                return False
            score, rationale = parse_score(comp.text)
            out.update({"score": score, "rationale": rationale, "raw_text": comp.text, "ts": self.clock()})
            self.scores_log.write(out)
            return True

        # One worker per judge slot, as `run` does for dialogues: `free` holds the slots not in use, and a
        # dyad's targets stay on one slot so its cache is reused from one target to the next.
        free = list(range(self.concurrency))
        lock = threading.Lock()
        counts = {"written": 0, "errors": 0}

        def job(group: list) -> None:
            with lock:
                slot = free.pop()
            try:
                judge = dataclasses.replace(self.judge, slot=slot)
                for row, metric in group:
                    ok = score_one(judge, row, metric)
                    with lock:
                        counts["written" if ok else "errors"] += 1
            finally:
                with lock:
                    free.append(slot)

        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            for f in [ex.submit(job, g) for g in groups.values()]:
                f.result()
        self.errors = counts["errors"]
        return counts["written"]
