"""Pre/post survey batteries for the mentor: one branch per item, numeric answer forced by JSON schema."""
from __future__ import annotations
import json
from pathlib import Path
from harness.client import ServerError
from harness.dialogue import AgentHandle, DyadSpec, GenSettings
from harness.log import JsonlWriter, derive_seed, now_iso, sha256_text
from harness.templates import render
from harness.transcript import Transcript, MENTOR

PHASES = ("pre", "post")
SURVEY_N_PREDICT = 32


class SurveyError(Exception):
    """Exception raised when a survey item fails during administration, with cause and context."""
    def __init__(self, dyad_id: str, phase: str, item_id: str, cause: Exception):
        super().__init__(f"{dyad_id} {phase} {item_id}: {cause}")
        self.dyad_id, self.phase, self.item_id, self.cause = dyad_id, phase, item_id, cause


def load_batteries(path: str | Path) -> list[dict]:
    """Load and validate survey battery items from a JSON file, checking for required fields and uniqueness."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    seen = set()
    for it in items:
        for k in ("id", "battery", "text", "scale"):
            if k not in it:
                raise ValueError(f"survey item missing {k!r}: {it}")
        if not isinstance(it["scale"], dict) or "min" not in it["scale"] or "max" not in it["scale"]:
            raise ValueError(f"survey item {it['id']} needs scale.min and scale.max")
        if it["id"] in seen:
            raise ValueError(f"duplicate survey item id {it['id']}")
        seen.add(it["id"])
    return items


def answer_schema(item: dict) -> dict:
    """Generate a JSON schema enforcing an integer answer within the item's scale range."""
    return {"type": "object",
            "properties": {"answer": {"type": "integer", "minimum": item["scale"]["min"], "maximum": item["scale"]["max"]}},
            "required": ["answer"]}


def parse_answer(text: str, item: dict) -> int | None:
    """Parse a JSON response for an integer answer field, returning None if absent or out of scale."""
    try:
        v = json.loads(text).get("answer")
    except (ValueError, AttributeError):
        return None
    if isinstance(v, bool) or not isinstance(v, int):
        return None
    if item["scale"]["min"] <= v <= item["scale"]["max"]:
        return v
    return None


class SurveyRunner:
    """Administers survey batteries to a mentor model, branching each item off the dialogue context and logging results."""
    def __init__(self, run_id: str, run_seed: int, mentor: AgentHandle, surveys_log: JsonlWriter,
                 settings: GenSettings, clock=now_iso):
        """Initialize with run metadata, mentor client, and logging writer."""
        self.run_id, self.run_seed, self.mentor = run_id, run_seed, mentor
        self.surveys_log, self.settings, self.clock = surveys_log, settings, clock

    def administer(self, spec: DyadSpec, attempt: int, phase: str, transcript: Transcript | None,
                   items: list[dict]) -> list[dict]:
        """Administer survey items to the mentor, one per prompt, logging and returning answer rows; raise SurveyError on server failure."""
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}")
        if phase == "post" and transcript is None:
            raise ValueError("post survey needs the dialogue transcript")
        base = transcript.view_for(MENTOR) if phase == "post" else []
        turn = 0 if phase == "pre" else spec.n_turns + 1
        rows = []
        for it in items:
            messages = base + [{"role": "user", "content": it["text"]}]
            prompt = render(self.mentor.template, messages, now=self.settings.now,
                            enable_thinking=self.settings.enable_thinking)
            seed = derive_seed(self.run_seed, spec.dyad_id, attempt, turn, f"survey:{phase}:{it['id']}")
            row = {"run_id": self.run_id, "dyad_id": spec.dyad_id, "attempt": attempt, "phase": phase,
                   "item_id": it["id"], "battery": it["battery"], "scale": it["scale"],
                   "prompt_sha256": sha256_text(prompt), "seed": seed}
            try:
                comp = self.mentor.client.complete(prompt, id_slot=self.mentor.slot, seed=seed,
                                                   n_predict=SURVEY_N_PREDICT, temperature=0.0,
                                                   json_schema=answer_schema(it), cache_prompt=True)
            except ServerError as e:
                row.update({"answer": None, "raw_text": "", "prompt_n": None, "error": str(e), "ts": self.clock()})
                self.surveys_log.write(row)
                raise SurveyError(spec.dyad_id, phase, it["id"], e) from e
            row.update({"answer": parse_answer(comp.text, it), "raw_text": comp.text,
                        "prompt_n": comp.prompt_n, "ts": self.clock()})
            self.surveys_log.write(row)
            rows.append(row)
        return rows
