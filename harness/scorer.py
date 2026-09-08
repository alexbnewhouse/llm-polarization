"""Offline adherence scorer: a judge model scores logged turns; results go to scores.jsonl."""
from __future__ import annotations
import json
from harness.client import ServerError
from harness.dialogue import AgentHandle, GenSettings
from harness.log import JsonlWriter, RunPaths, derive_seed, now_iso, read_jsonl, resume_index, sha256_text
from harness.templates import render
from harness.transcript import SEEKER, MENTOR

METRICS = {"prompt_to_line": SEEKER, "line_to_line": SEEKER, "alignment": MENTOR}
SCOPES = ("pilot", "main")
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
    """Return the JSON schema the judge's response must satisfy: a 0-1 score and a rationale string, both required."""
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


def select_targets(turn_rows: list[dict], scope: str) -> list[tuple[dict, str]]:
    """Pick which (turn row, metric) pairs to score: every non-error row/metric for 'pilot', a turn-4 cadence
    of seeker rows for 'main'; raises ValueError for any other scope."""
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    rows = [r for r in turn_rows if r.get("finish_reason") != "error"]
    out = []
    if scope == "pilot":
        for r in rows:
            for metric, agent in METRICS.items():
                if r["agent"] == agent:
                    out.append((r, metric))
        return out
    seeker_rows = [r for r in rows if r["agent"] == SEEKER]
    by_dyad: dict[tuple, int] = {}
    for r in seeker_rows:
        key = (r["dyad_id"], r.get("attempt", 1))
        by_dyad[key] = max(by_dyad.get(key, 0), r["turn"])
    for r in seeker_rows:
        if r["turn"] % 4 == 0 or r["turn"] == by_dyad[(r["dyad_id"], r.get("attempt", 1))]:
            for metric, agent in METRICS.items():
                if agent == SEEKER:
                    out.append((r, metric))
    return out


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
    """Scores a finished run's logged turns with a third judge model and appends the results to scores.jsonl."""
    def __init__(self, run_id: str, run_seed: int, judge: AgentHandle, scores_log: JsonlWriter,
                 settings: GenSettings, clock=now_iso, harness_commit: str = ""):
        """Wire up the run identity, judge agent, output log, generation settings, clock and the harness
        commit that is doing the scoring (scoring can happen long after the run, from different code)."""
        self.run_id, self.run_seed, self.judge = run_id, run_seed, judge
        self.scores_log, self.settings, self.clock = scores_log, settings, clock
        self.harness_commit = harness_commit

    def score_run(self, paths: RunPaths, scope: str, manifest: dict) -> int:
        """Score every not-yet-scored target for the run's complete dyads and return how many rows were written."""
        for role in ("seeker", "mentor"):
            if manifest.get(role, {}).get("model_sha256") == self.judge.model_sha256:
                raise ValueError(f"judge model is the same as the {role} model; pick a third model")
        complete = latest_complete_attempts(read_jsonl(paths.status))
        dyads = {(d["dyad_id"], d["attempt"]): d for d in read_jsonl(paths.dyads)}
        done = {(s["dyad_id"], s["attempt"], s["turn"], s["agent"], s["metric"])
                for s in read_jsonl(paths.scores) if not s.get("error")}
        turns = [r for r in read_jsonl(paths.turns) if complete.get(r["dyad_id"]) == r.get("attempt", 1)]
        by_dyad: dict[tuple, list[dict]] = {}
        for r in turns:
            by_dyad.setdefault((r["dyad_id"], r.get("attempt", 1)), []).append(r)
        histories: dict[tuple, list[dict]] = {
            k: sorted(v, key=lambda r: (r["turn"], 0 if r["agent"] == SEEKER else 1)) for k, v in by_dyad.items()}
        positions: dict[tuple, dict[int, int]] = {
            k: {id(r): i for i, r in enumerate(v)} for k, v in histories.items()}
        written = 0
        for row, metric in select_targets(turns, scope):
            key = (row["dyad_id"], row.get("attempt", 1), row["turn"], row["agent"], metric)
            if key in done:
                continue
            # The dyad row carries the per-dyad seed the run used, so a score seed is derived from the same
            # (run_seed, dyad_seed) pair the dialogue was: look the row up before deriving the seed.
            spec = dyads.get(key[:2])
            seed = derive_seed(self.run_seed, int((spec or {}).get("seed", 0)), row["dyad_id"], key[1],
                               row["turn"], f"judge:{row['agent']}:{metric}")
            out = {"run_id": self.run_id, "dyad_id": row["dyad_id"], "attempt": key[1], "turn": row["turn"],
                   "agent": row["agent"], "metric": metric, "judge_sha256": self.judge.model_sha256,
                   "id_slot": self.judge.slot, "harness_commit": self.harness_commit, "seed": seed}
            if spec is None:
                out.update({"judge_prompt_sha256": "", "prompt_chars": None, "score": None, "rationale": "",
                            "raw_text": "", "error": "no dyads.jsonl row for this dyad/attempt", "ts": self.clock()})
                self.scores_log.write(out)
                continue
            history = histories[key[:2]]
            idx = positions[key[:2]][id(row)]
            prior_own = [r["text"] for r in history[:idx] if r["agent"] == row["agent"]]
            partner = next((r["text"] for r in reversed(history[:idx]) if r["agent"] != row["agent"]), None)
            messages = build_judge_messages(metric, spec.get("persona_text", ""), spec.get("condition", {}).get("topic", ""),
                                            row["text"], prior_own, partner)
            prompt = render(self.judge.template, messages, now=self.settings.now, enable_thinking=self.settings.enable_thinking)
            out["judge_prompt_sha256"] = sha256_text(prompt)
            out["prompt_chars"] = len(prompt)
            try:
                comp = self.judge.client.complete(prompt, id_slot=self.judge.slot, seed=seed, n_predict=JUDGE_N_PREDICT,
                                                  temperature=JUDGE_TEMPERATURE, json_schema=score_schema(),
                                                  cache_prompt=True)
            except ServerError as e:
                out.update({"score": None, "rationale": "", "raw_text": "", "error": str(e), "ts": self.clock()})
                self.scores_log.write(out)
                continue
            score, rationale = parse_score(comp.text)
            out.update({"score": score, "rationale": rationale, "raw_text": comp.text, "ts": self.clock()})
            self.scores_log.write(out)
            written += 1
        return written
