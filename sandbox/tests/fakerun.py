"""A real-shaped run for the sandbox's readers, written by the harness's own code.

`make_fake_run` drives `harness.run.run_dyad` (with `RunContext`, `JsonlWriter`, `write_manifest`) and
`harness.scorer.Scorer` over fake llama-servers, so `sandbox.runs` and `sandbox.analysis` are tested on
exactly the files the harness writes -- dyads, status, turns, surveys, scores, the manifest and the judge
manifest -- rather than on hand-made rows that could drift from them. No network, no GGUF; a run of six
dyads at three turns takes well under a second."""
from __future__ import annotations
import copy
import datetime as _dt
import json
from collections import Counter
from pathlib import Path
from harness import run as R
from harness.dialogue import AgentHandle, DyadSpec, GenSettings
from harness.log import JsonlWriter, read_jsonl, run_paths, sha256_file, sha256_text, write_manifest
from harness.scorer import Scorer
from harness.survey import load_batteries
from harness.templates import ChatTemplate
from harness.tests.conftest import CHATML
from harness.tests.fakes import FakeClient

REPO = Path(__file__).resolve().parents[2]
BATTERIES = REPO / "instruments" / "batteries.json"
GRID = REPO / "prompts" / "grid.json"
RUN_SEED = 20260908
TOPIC = "immigration_enforcement"

# Dyad ids are d01, d02, ... in this order (see dyad_ids). Two dyads at strong_left and strong_right so a
# level has a standard error; the last row is the grid's no-persona control.
DEFAULT_CONDITIONS = (
    {"topic": TOPIC, "ideology": "strong_left", "openness": "open", "role": "union_organizer_placeholder"},
    {"topic": TOPIC, "ideology": "strong_left", "openness": "closed", "role": "union_organizer_placeholder"},
    {"topic": TOPIC, "ideology": "moderate", "openness": "open", "role": "small_business_owner_placeholder"},
    {"topic": TOPIC, "ideology": "strong_right", "openness": "open", "role": "retired_officer_placeholder"},
    {"topic": TOPIC, "ideology": "strong_right", "openness": "closed", "role": "retired_officer_placeholder"},
    {"topic": TOPIC, "ideology": "none", "openness": None, "role": None},
)


def dyad_ids(conditions=None) -> list[str]:
    """The dyad ids make_fake_run gives `conditions`, in order: d01, d02, ..."""
    return [f"d{i:02d}" for i in range(1, len(conditions or DEFAULT_CONDITIONS) + 1)]


class ScriptedClient(FakeClient):
    """A FakeClient whose reply is computed from the request instead of read off a list: a survey request
    (a JSON schema with `answer`) gets an integer on the item's scale, a judge request (`score`) gets a
    score and a rationale, anything else a short line. Deterministic in the prompt, so a run is
    reproducible; the post answer always differs from the pre answer, so every item has a non-zero shift."""

    def __init__(self, role: str, fail_on: int | None = None):
        super().__init__(["unused"], fail_on=fail_on)
        self.role = role

    def complete(self, prompt, **kw):
        # FakeClient keeps the per-slot cache accounting and the scripted failure; only the text is ours.
        self.replies = [self._reply(prompt, kw.get("json_schema"))]
        return super().complete(prompt, **kw)

    def _reply(self, prompt: str, schema: dict | None) -> str:
        h = int(sha256_text(prompt)[:8], 16)
        props = (schema or {}).get("properties", {})
        if "answer" in props:
            lo, hi = props["answer"]["minimum"], props["answer"]["maximum"]
            span = hi - lo + 1
            item = prompt.rsplit("<|im_start|>user\n", 1)[-1]
            value = int(sha256_text(item)[:8], 16) % span            # pre: depends on the item only
            if span > 1 and prompt.count("<|im_start|>assistant") > 1:
                # post: the mentor's own lines are in the prompt. Shift by 1..span-1 (mod span) so post != pre.
                value = (value + 1 + h % (span - 1)) % span
            return json.dumps({"answer": lo + value})
        if "score" in props:
            return json.dumps({"score": round((h % 101) / 100, 2), "rationale": "r"})
        return f"{self.role} line {h % 10000}."


class _Clock:
    """Strictly increasing timestamps in the harness's format, one second apart, so `ts` is deterministic
    and orders rows the way a real run's clock would."""

    def __init__(self):
        self.n = 0
        self.base = _dt.datetime(2026, 10, 7, 9, 0, 0, tzinfo=_dt.timezone.utc)

    def __call__(self) -> str:
        self.n += 1
        return (self.base + _dt.timedelta(seconds=self.n)).strftime("%Y-%m-%dT%H:%M:%S%z")


def _role_entry(role: str, tpl: ChatTemplate) -> dict:
    """A manifest role block with every field harness.run.build_agent records, including the large ones
    (template_source, default_generation_settings) that the readers must strip."""
    return {"url": f"http://fake-{role}", "alias": f"fake-{role}", "model_path": f"/fake/{role}.gguf",
            "model_sha256": sha256_text(f"fake {role} model"), "family": None, "template_sha256": tpl.sha256,
            "template_source": tpl.source, "server_chat_template": None, "build_info": "fake",
            "model_ftype": None, "total_slots": 4, "default_generation_settings": {"n_ctx": 8192}}


def _manifest_row(dyad_id: str, condition: dict, seed: int, n_turns: int) -> dict:
    ideology = condition.get("ideology")
    if condition.get("role") is None:
        persona = f"You came to ask for guidance about {condition.get('topic')}."
        reminder = "Note to self: keep asking for guidance."
    else:
        persona = (f"You are a {ideology} {condition.get('role')}, {condition.get('openness')} to persuasion, "
                   f"talking about {condition.get('topic')}.")
        reminder = f"Note to self: I am {ideology}."
    return {"dyad_id": dyad_id, "condition": dict(condition), "persona_text": persona, "persona_reminder": reminder,
            "persona_mode": "reinforced", "seed": seed, "n_turns": n_turns}


def make_fake_run(data_dir, run_id="fake-run", *, n_turns=3, conditions=None, fail=(), scores=True) -> Path:
    """Write data/<run_id>/ the way `harness run` and `harness score --scope pilot` would, and return it.

    Two passes, like an operator running `run` and then resuming it once. `fail` names dyad ids whose
    attempts fail mid-dialogue (the seeker's server errors at turn 2): an id given once (or mapped to 1)
    fails attempt 1 and completes as attempt 2 on the resume; an id given twice (or mapped to 2 or more)
    fails both attempts, so its latest attempt is `failed`. The input manifest is written beside the run
    directory as `<run_id>-dyads.jsonl` (that is `planned`); config.grid is prompts/grid.json. With
    `scores`, a third fake model scores every turn in pilot scope and its judge manifest is written."""
    data_dir = Path(data_dir)
    conditions = list(conditions or DEFAULT_CONDITIONS)
    ids = dyad_ids(conditions)
    failures = Counter(fail) if not isinstance(fail, dict) else Counter({k: int(v) for k, v in fail.items()})
    unknown = set(failures) - set(ids)
    if unknown:
        raise ValueError(f"fail names dyads that are not in the run: {sorted(unknown)}")
    clock = _Clock()
    paths = run_paths(data_dir, run_id)
    rows = [_manifest_row(d, c, 1000 + i, n_turns) for i, (d, c) in enumerate(zip(ids, conditions))]
    input_manifest = data_dir / f"{run_id}-dyads.jsonl"
    input_manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    tpl = ChatTemplate.from_source(CHATML)
    entries = {role: _role_entry(role, tpl) for role in ("seeker", "mentor", "judge")}
    cfg = copy.deepcopy(R.DEFAULT_CONFIG)
    cfg.update({"data_dir": str(data_dir), "batteries": str(BATTERIES), "grid": str(GRID), "run_seed": RUN_SEED,
                **{role: {"url": entries[role]["url"], "gguf_path": None} for role in ("seeker", "mentor", "judge")}})
    items = load_batteries(BATTERIES)
    manifest = {"run_id": run_id, "started_at": clock(), "harness_commit": "fake", "harness_dirty": False,
                "config": cfg, "input_manifest": {"path": str(input_manifest), "sha256": sha256_file(input_manifest)},
                "batteries": {"path": str(BATTERIES), "sha256": sha256_file(BATTERIES), "n_items": len(items),
                              "item_ids": [it["id"] for it in items]},
                "environment": {}, "seeker": entries["seeker"], "mentor": entries["mentor"]}
    write_manifest(paths, manifest)
    g = cfg["generation"]
    settings = GenSettings(g["temperature"], g["top_p"], g["n_predict"], cfg["now"], g["enable_thinking"],
                           cache_reuse_limit=cfg["cache_reuse_limit"])
    logs = {k: JsonlWriter(getattr(paths, k)) for k in ("dyads", "status", "turns", "surveys")}

    def run_one(i: int, spec: DyadSpec, attempt: int) -> str:
        # Fresh clients per dyad attempt: a failing seeker is scripted per attempt, and a new client is a
        # cold slot, as on a server that has just served another dyad.
        fails = failures.get(spec.dyad_id, 0) >= attempt
        seeker = AgentHandle("seeker", ScriptedClient("seeker", fail_on=min(2, spec.n_turns) if fails else None),
                             tpl, entries["seeker"]["model_sha256"], 0, alias="fake-seeker")
        mentor = AgentHandle("mentor", ScriptedClient("mentor"), tpl, entries["mentor"]["model_sha256"], 0,
                             alias="fake-mentor")
        ctx = R.RunContext(run_id, RUN_SEED, seeker, mentor, settings, items, logs, clock=clock,
                           batteries_sha256=manifest["batteries"]["sha256"])
        return R.run_dyad(i % 4, spec, attempt, ctx)

    for _ in range(2):                    # the run, then one resume of whatever did not complete
        for i, (spec, attempt) in enumerate(R.plan_work(rows, read_jsonl(paths.status))):
            run_one(i, spec, attempt)

    if scores:
        judge = AgentHandle("judge", ScriptedClient("judge"), tpl, entries["judge"]["model_sha256"], 0,
                            alias="fake-judge", family=None)
        R.write_judge_manifest(paths, entries["judge"], "pilot")
        Scorer(run_id, RUN_SEED, judge, JsonlWriter(paths.scores), settings, clock=clock,
               harness_commit="fake").score_run(paths, "pilot", manifest)
    return paths.root
