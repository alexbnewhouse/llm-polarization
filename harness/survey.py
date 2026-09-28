"""Pre/post survey batteries for the mentor: one branch per item, numeric answer forced by JSON schema and
parsed by harness.parser (the schema path first, free-text salvage behind it, the method on every row)."""
from __future__ import annotations
import json
from pathlib import Path
from harness.client import ServerError
from harness.dialogue import AgentHandle, DyadSpec, GenSettings
from harness.log import JsonlWriter, derive_seed, now_iso, sha256_text
from harness.parser import Parsed, parse_scale_answer
from harness.templates import render
from harness.transcript import Transcript, MENTOR

PHASES = ("pre", "post")
# run: the dialogue run's own pass; readministered: a later `survey` pass; baseline: `baseline`, no dialogue.
ORIGINS = ("run", "readministered", "baseline")
SURVEY_N_PREDICT = 32
SURVEY_TEMPERATURE = 0.0
SURVEY_TOP_P = 0.95
# answer_method for a reply the n_predict cap cut off before it parsed as JSON: `{"answer": 1` on a 0-10
# item may have been heading for 10, so no number is salvaged from it.
TRUNCATED = "truncated"


class SurveyError(Exception):
    """A survey item's request failed; carries the dyad, phase and item for the status row."""
    def __init__(self, dyad_id: str, phase: str, item_id: str, cause: Exception):
        super().__init__(f"{dyad_id} {phase} {item_id}: {cause}")
        self.dyad_id, self.phase, self.item_id, self.cause = dyad_id, phase, item_id, cause


def load_batteries(path: str | Path) -> list[dict]:
    """Load the survey items (a list, or an object with `items`) and check each has id, battery, non-empty
    text and integer scale.min < scale.max, with ids unique. File order is administration order. A bad
    scale would otherwise surface later, as every dyad failing its pre-survey (red-team L6)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    seen = set()
    for it in items:
        for k in ("id", "battery", "text", "scale"):
            if k not in it:
                raise ValueError(f"survey item missing {k!r}: {it}")
        if not isinstance(it["text"], str) or not it["text"].strip():
            raise ValueError(f"survey item {it['id']} has no text")
        if not isinstance(it["scale"], dict) or "min" not in it["scale"] or "max" not in it["scale"]:
            raise ValueError(f"survey item {it['id']} needs scale.min and scale.max")
        lo, hi = it["scale"]["min"], it["scale"]["max"]
        # bool is an int in Python; 1.0 and "5" are not integer bounds either.
        if type(lo) is not int or type(hi) is not int:
            raise ValueError(f"survey item {it['id']}: scale.min and scale.max must be integers, got {lo!r} "
                             f"and {hi!r}")
        if lo >= hi:
            raise ValueError(f"survey item {it['id']}: scale.min {lo} must be below scale.max {hi}")
        if it["id"] in seen:
            raise ValueError(f"duplicate survey item id {it['id']}")
        seen.add(it["id"])
    return items


def answer_schema(item: dict) -> dict:
    """The JSON schema that constrains the reply to `{"answer": <int on the item's scale>}`."""
    return {"type": "object",
            "properties": {"answer": {"type": "integer", "minimum": item["scale"]["min"], "maximum": item["scale"]["max"]}},
            "required": ["answer"]}


def parse_answer(text: str, item: dict) -> Parsed:
    """Parse a reply to `item` into a Parsed(value, method): `json` when the schema-constrained reply
    parsed as an integer on the item's scale, a salvage method otherwise (harness/parser.py)."""
    return parse_scale_answer(text, item["scale"]["min"], item["scale"]["max"])


class SurveyRunner:
    """Administers the survey items to the mentor, one prompt per item, and logs one row per item."""
    def __init__(self, run_id: str, run_seed: int, mentor: AgentHandle, surveys_log: JsonlWriter,
                 settings: GenSettings, clock=now_iso, batteries_sha256: str = "", *,
                 temperature: float = SURVEY_TEMPERATURE, top_p: float = SURVEY_TOP_P,
                 n_predict: int = SURVEY_N_PREDICT, schema: bool = True):
        """Initialize with run metadata, mentor client, logging writer and the sha256 of the batteries file
        whose items are being administered (written onto every row so an item-wording change is visible).
        The sampling settings default to the instrument's (greedy, 32 tokens, schema-constrained);
        `survey --temperature/--n-predict/--no-schema` and `baseline` override them, and every row says
        which it got. schema=False sends no json_schema: the reply is free text and only the parser's
        salvage path can read it."""
        self.run_id, self.run_seed, self.mentor = run_id, run_seed, mentor
        self.surveys_log, self.settings, self.clock = surveys_log, settings, clock
        self.batteries_sha256 = batteries_sha256
        self.temperature, self.top_p, self.n_predict, self.schema = temperature, top_p, n_predict, schema

    def administer(self, spec: DyadSpec, attempt: int, phase: str, transcript: Transcript | None,
                   items: list[dict], origin: str = "run", extra: dict | None = None) -> list[dict]:
        """Administer every item for one phase and return the rows; raise SurveyError on a server failure,
        after logging the failed row. Pre items get a fresh context; post items branch off the mentor's view
        of the dialogue on the same slot, so the cached dialogue prefix is reused and each item costs about
        its own prefill. `origin` is "run" for the pass the dialogue run itself makes,
        "readministered" for a later `harness survey` pass, whose rows otherwise share the same key, and
        "baseline" for `harness baseline`. `extra` fields are added to every row."""
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}")
        if origin not in ORIGINS:
            raise ValueError(f"origin must be one of {ORIGINS}")
        if phase == "post" and transcript is None:
            raise ValueError("post survey needs the dialogue transcript")
        base = transcript.view_for(MENTOR) if phase == "post" else []
        # Not a real turn: a sentinel so the pre and post seeds differ. 0 = before turn 1,
        # n_turns+1 = after the last turn.
        turn = 0 if phase == "pre" else spec.n_turns + 1
        rows = []
        for it in items:
            messages = base + [{"role": "user", "content": it["text"]}]
            prompt = render(self.mentor.template, messages, now=self.settings.now,
                            enable_thinking=self.settings.enable_thinking)
            seed = derive_seed(self.run_seed, spec.seed, spec.dyad_id, attempt, turn, f"survey:{phase}:{it['id']}")
            row = {"run_id": self.run_id, "dyad_id": spec.dyad_id, "attempt": attempt, "phase": phase,
                   "origin": origin, "item_id": it["id"], "battery": it["battery"], "scale": it["scale"],
                   "batteries_sha256": self.batteries_sha256, "model_sha256": self.mentor.model_sha256,
                   "template_sha256": self.mentor.template.sha256, "id_slot": self.mentor.slot, "turn": turn,
                   "temperature": self.temperature, "top_p": self.top_p, "n_predict": self.n_predict,
                   "schema": self.schema, **(extra or {}),
                   "prompt_sha256": sha256_text(prompt), "prompt_chars": len(prompt), "seed": seed}
            try:
                comp = self.mentor.client.complete(prompt, id_slot=self.mentor.slot, seed=seed,
                                                   n_predict=self.n_predict, temperature=self.temperature,
                                                   top_p=self.top_p,
                                                   json_schema=answer_schema(it) if self.schema else None,
                                                   cache_prompt=True, samplers=self.settings.samplers)
            except ServerError as e:
                row.update({"answer": None, "answer_method": None, "raw_text": "", "finish_reason": "error",
                            "predicted_n": None, "truncated": None, "prompt_n": None, "error": str(e),
                            "ts": self.clock()})
                self.surveys_log.write(row)
                raise SurveyError(spec.dyad_id, phase, it["id"], e) from e
            parsed = parse_answer(comp.text, it)
            answer, method = parsed.value, parsed.method
            if comp.finish_reason == "length" and method != "json":
                answer, method = None, TRUNCATED
            row.update({"answer": answer, "answer_method": method, "raw_text": comp.text,
                        "finish_reason": comp.finish_reason, "predicted_n": comp.predicted_n,
                        "truncated": comp.truncated, "prompt_n": comp.prompt_n, "ts": self.clock()})
            self.surveys_log.write(row)
            rows.append(row)
        return rows
