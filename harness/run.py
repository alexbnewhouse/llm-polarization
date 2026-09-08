"""CLI: check servers, run a dialogue manifest, re-administer surveys, score a run."""
from __future__ import annotations
import argparse, copy, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from harness import log
from harness.client import LlamaClient, ServerError
from harness.dialogue import AgentHandle, DialogueError, DialogueRunner, DyadSpec, GenSettings
from harness.log import JsonlWriter, ManifestMismatch, RunPaths, now_iso, read_jsonl, run_paths
from harness.scorer import Scorer
from harness.survey import SurveyError, SurveyRunner, load_batteries
from harness.templates import FIXTURE_MESSAGES, TemplateError, parity_check, read_template_from_gguf, render
from harness.transcript import MENTOR, SEEKER, Transcript

DEFAULT_CONFIG = {
    "data_dir": "data", "gguf_py_path": None, "batteries": "instruments/batteries.json", "run_seed": 0,
    "concurrency": None, "now": "2026-09-08",
    "generation": {"temperature": 0.7, "top_p": 0.95, "n_predict": 300, "timeout": 600, "enable_thinking": False},
    "seeker": {"url": None, "gguf_path": None}, "mentor": {"url": None, "gguf_path": None},
    "judge": {"url": None, "gguf_path": None},
}
HASH_CACHE = Path.home() / ".cache" / "llm-polarization" / "gguf-hashes.json"


def _merge(base: dict, over: dict) -> dict:
    """Deep-merge override dict `over` onto a deep copy of `base`, recursing into nested dicts."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path) -> dict:
    """Load a JSON config file, deep-merge it onto DEFAULT_CONFIG, and require seeker/mentor URLs."""
    cfg = _merge(DEFAULT_CONFIG, json.loads(Path(path).read_text(encoding="utf-8")))
    for role in ("seeker", "mentor"):
        if not cfg[role].get("url"):
            raise ValueError(f"config needs {role}.url")
    return cfg


def model_sha256_cached(path: str, cache_file: Path | None = None) -> str:
    """Return a GGUF file's sha256, cached on disk keyed by path|size|mtime so re-hashing is skipped."""
    cache_file = cache_file or HASH_CACHE
    st = os.stat(path)
    key = f"{path}|{st.st_size}|{int(st.st_mtime)}"
    cache = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
        except ValueError:
            cache = {}
    if key not in cache:
        cache[key] = log.sha256_file(Path(path))
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache, indent=1))
    return cache[key]


def build_agent(name: str, entry: dict, slot: int, cfg: dict, client_factory=None) -> tuple[AgentHandle, dict]:
    """Connect to a server, read its /props and GGUF chat template, and build the AgentHandle plus the
    manifest entry that records this run's model identity for provenance."""
    factory = client_factory or (lambda url: LlamaClient(url, timeout=cfg["generation"]["timeout"]))
    client = factory(entry["url"])
    props = client.props()
    model_path = entry.get("gguf_path") or props.get("model_path")
    if not model_path:
        raise ValueError(f"{name}: no gguf_path in config and server reports no model_path")
    template = read_template_from_gguf(model_path, gguf_py_path=cfg.get("gguf_py_path"))
    sha = model_sha256_cached(model_path)
    handle = AgentHandle(name, client, template, sha, slot, alias=props.get("model_alias", ""))
    manifest_entry = {"url": entry["url"], "alias": props.get("model_alias", ""), "model_path": model_path,
                      "model_sha256": sha, "template_sha256": template.sha256, "build_info": props.get("build_info", ""),
                      "total_slots": props.get("total_slots"), "default_generation_settings": props.get("default_generation_settings", {})}
    return handle, manifest_entry


def check_agent(handle: AgentHandle, cfg: dict) -> list[tuple[str, bool, str]]:
    """Run this agent's pre-flight checks: server health, template-rendering parity with the server, and,
    for the seeker only, whether its template accepts a trailing system message (the persona reminder)."""
    now = cfg.get("now", "2026-09-08")
    results = [("health", handle.client.health(), handle.client.url if hasattr(handle.client, "url") else "")]
    try:
        ok, ours, theirs = parity_check(handle.template, handle.client, FIXTURE_MESSAGES[:-1], now=now)
        results.append(("template_parity", ok, "" if ok else f"ours={ours[-120:]!r} theirs={theirs[-120:]!r}"))
    except TemplateError as e:
        results.append(("template_parity", False, str(e)))
    if handle.name == SEEKER:
        try:
            render(handle.template, FIXTURE_MESSAGES, now=now)
            results.append(("trailing_system", True, ""))
        except TemplateError as e:
            results.append(("trailing_system", False, str(e)))
    return results


def plan_work(manifest_rows: list[dict], status_rows: list[dict]) -> list[tuple[DyadSpec, int]]:
    """Diff the dyad manifest against the run's status log and return the (spec, attempt) pairs still to run."""
    idx = log.resume_index(status_rows)
    work = []
    for row in manifest_rows:
        spec = DyadSpec.from_row(row)
        attempt = log.next_attempt(idx, spec.dyad_id)
        if attempt is not None:
            work.append((spec, attempt))
    return work


@dataclass
class RunContext:
    """Everything one dyad run needs beyond its own spec/attempt: the run identity, agents, settings,
    survey items, log writers and clock, bundled so worker threads can share it read-only."""
    run_id: str
    run_seed: int
    seeker: AgentHandle
    mentor: AgentHandle
    settings: GenSettings
    items: list[dict]
    logs: dict
    clock: object = now_iso


def _with_slot(h: AgentHandle, slot: int) -> AgentHandle:
    """Return a copy of an AgentHandle pinned to a different server slot, for reuse across worker threads."""
    return AgentHandle(h.name, h.client, h.template, h.model_sha256, slot, h.alias)


def run_dyad(worker_slot: int, spec: DyadSpec, attempt: int, ctx: RunContext) -> str:
    """Run one dyad end to end on the given worker slot: log the dyad row and 'started' status, administer
    the pre-survey, run the dialogue, administer the post-survey, then log 'complete' or, on a
    DialogueError/SurveyError, 'failed' with the reason; returns 'complete' or 'failed'."""
    seeker, mentor = _with_slot(ctx.seeker, worker_slot), _with_slot(ctx.mentor, worker_slot)
    ctx.logs["dyads"].write({"run_id": ctx.run_id, "dyad_id": spec.dyad_id, "attempt": attempt, "condition": spec.condition,
                             "persona_text": spec.persona_text, "persona_reminder": spec.persona_reminder,
                             "persona_mode": spec.persona_mode, "seed": spec.seed, "n_turns": spec.n_turns, "ts": ctx.clock()})
    ctx.logs["status"].write({"run_id": ctx.run_id, "dyad_id": spec.dyad_id, "attempt": attempt, "status": "started", "ts": ctx.clock()})
    dialogue = DialogueRunner(ctx.run_id, ctx.run_seed, seeker, mentor, ctx.settings, ctx.logs["turns"], clock=ctx.clock)
    surveys = SurveyRunner(ctx.run_id, ctx.run_seed, mentor, ctx.logs["surveys"], ctx.settings, clock=ctx.clock)
    try:
        surveys.administer(spec, attempt, "pre", None, ctx.items)
        transcript = dialogue.run(spec, attempt)
        surveys.administer(spec, attempt, "post", transcript, ctx.items)
    except (DialogueError, SurveyError) as e:
        ctx.logs["status"].write({"run_id": ctx.run_id, "dyad_id": spec.dyad_id, "attempt": attempt, "status": "failed",
                                  "reason": str(e), "ts": ctx.clock()})
        return "failed"
    ctx.logs["status"].write({"run_id": ctx.run_id, "dyad_id": spec.dyad_id, "attempt": attempt, "status": "complete", "ts": ctx.clock()})
    return "complete"


def _settings(cfg: dict) -> GenSettings:
    """Build the GenSettings the run will use for dialogue turns from the config's generation block."""
    g = cfg["generation"]
    return GenSettings(g["temperature"], g["top_p"], g["n_predict"], cfg["now"], g["enable_thinking"])


def _agents(cfg: dict, roles=(SEEKER, MENTOR)) -> dict:
    """Build an AgentHandle (and its manifest entry) for each requested role, keyed by role name."""
    out = {}
    for role in roles:
        out[role] = build_agent(role, cfg[role], 0, cfg)
    return out


def cmd_check(cfg: dict) -> int:
    """Build the seeker and mentor agents, print each one's check_agent results, and return 0 iff all pass;
    a role whose server is unreachable prints a FAIL health line instead of letting build_agent's
    ServerError traceback out, since a down server is exactly the failure `check` exists to report."""
    ok_all = True
    for role in (SEEKER, MENTOR):
        try:
            handle, entry = build_agent(role, cfg[role], 0, cfg)
        except ServerError as e:
            print(f"FAIL health {role} {cfg[role].get('url')}: {e}")
            ok_all = False
            continue
        print(f"[{role}] {entry['alias']} {entry['model_path']} sha256={entry['model_sha256'][:12]} slots={entry['total_slots']}")
        for name, ok, detail in check_agent(handle, cfg):
            ok_all &= ok
            print(f"   {'ok ' if ok else 'FAIL'} {name} {detail}")
    return 0 if ok_all else 1


def _rebuild_transcript(dyad_row: dict, turn_rows: list[dict]) -> Transcript:
    """Reconstruct a dyad's Transcript from its dyads.jsonl row and turns.jsonl rows, in turn/agent order,
    so the post survey can be re-administered without re-running the dialogue."""
    t = Transcript(dyad_row["dyad_id"], dyad_row["persona_text"], dyad_row.get("persona_reminder") or None, dyad_row["persona_mode"])
    for r in sorted(turn_rows, key=lambda r: (r["turn"], 0 if r["agent"] == SEEKER else 1)):
        if r.get("finish_reason") != "error":
            t.append(r["turn"], r["agent"], r["text"])
    return t


def cmd_run(cfg: dict, manifest_path: str, run_id: str) -> int:
    """Run the `run` subcommand: check the servers, write/verify the run manifest, plan the outstanding
    (dyad, attempt) work, and execute it across a thread pool with one slot per worker."""
    if cmd_check(cfg) != 0:
        print("check failed; not running", file=sys.stderr)
        return 1
    agents = _agents(cfg)
    (seeker, s_entry), (mentor, m_entry) = agents[SEEKER], agents[MENTOR]
    paths = run_paths(cfg["data_dir"], run_id)
    log.write_manifest(paths, {"run_id": run_id, "started_at": now_iso(), "harness_commit": _git_commit(),
                               "config": cfg, "seeker": s_entry, "mentor": m_entry})
    concurrency = cfg["concurrency"] or min(int(s_entry["total_slots"] or 1), int(m_entry["total_slots"] or 1))
    rows = read_jsonl(Path(manifest_path))
    work = plan_work(rows, read_jsonl(paths.status))
    print(f"run {run_id}: {len(work)} of {len(rows)} dyads to run, concurrency {concurrency}")
    if not work:
        return 0
    logs = {k: JsonlWriter(getattr(paths, k)) for k in ("dyads", "status", "turns", "surveys")}
    ctx = RunContext(run_id, int(cfg["run_seed"]), seeker, mentor, _settings(cfg),
                     load_batteries(cfg["batteries"]), logs)
    free = list(range(concurrency))
    lock = threading.Lock()

    def job(spec, attempt):
        """Run one (spec, attempt) job on a free slot, returning it to the pool when done."""
        with lock:
            slot = free.pop()
        try:
            r = run_dyad(slot, spec, attempt, ctx)
            print(f"  {spec.dyad_id} attempt {attempt}: {r}", flush=True)
            return r
        finally:
            with lock:
                free.append(slot)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(lambda w: job(*w), work))
    failed = results.count("failed")
    print(f"done: {results.count('complete')} complete, {failed} failed")
    return 0


def cmd_survey(cfg: dict, run_id: str, phase: str) -> int:
    """Run the `survey` subcommand: re-administer one survey phase for every complete dyad in a run,
    rebuilding the transcript from turns.jsonl for the post phase."""
    agents = _agents(cfg)
    mentor = agents[MENTOR][0]
    paths = run_paths(cfg["data_dir"], run_id)
    items = load_batteries(cfg["batteries"])
    complete = {d: v["attempt"] for d, v in log.resume_index(read_jsonl(paths.status)).items() if v["status"] == "complete"}
    dyads = {(d["dyad_id"], d["attempt"]): d for d in read_jsonl(paths.dyads)}
    turns = read_jsonl(paths.turns)
    runner = SurveyRunner(run_id, int(cfg["run_seed"]), _with_slot(mentor, 0), JsonlWriter(paths.surveys), _settings(cfg))
    n = 0
    for dyad_id, attempt in complete.items():
        spec = DyadSpec.from_row(dyads[(dyad_id, attempt)])
        transcript = None
        if phase == "post":
            transcript = _rebuild_transcript(dyads[(dyad_id, attempt)],
                                             [r for r in turns if r["dyad_id"] == dyad_id and r.get("attempt", 1) == attempt])
        try:
            n += len(runner.administer(spec, attempt, phase, transcript, items))
        except SurveyError as e:
            print(f"  {dyad_id}: {e}", file=sys.stderr)
    print(f"{phase} survey: {n} rows for {len(complete)} dyads")
    return 0


def cmd_score(cfg: dict, run_id: str, scope: str) -> int:
    """Run the `score` subcommand: build the judge agent and score the run's not-yet-scored targets."""
    if not cfg["judge"].get("url"):
        print("config needs judge.url", file=sys.stderr)
        return 1
    judge, _ = build_agent("judge", cfg["judge"], 0, cfg)
    paths = run_paths(cfg["data_dir"], run_id)
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    n = Scorer(run_id, int(cfg["run_seed"]), judge, JsonlWriter(paths.scores), _settings(cfg)).score_run(paths, scope, manifest)
    print(f"scored {n} new rows ({scope})")
    return 0


def _git_commit() -> str:
    """Return the current git commit hash for provenance, or '' if git is unavailable or this isn't a repo."""
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return ""


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse the check/run/survey/score subcommands, dispatch to the matching cmd_* function,
    and turn a ServerError or ManifestMismatch into a clean error line and exit code 1 instead of a traceback."""
    ap = argparse.ArgumentParser(prog="harness", description="Dyad harness for the LLM polarization study")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "run", "survey", "score"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name != "check":
            p.add_argument("--run-id", required=True)
    sub.choices["run"].add_argument("--manifest", required=True)
    sub.choices["survey"].add_argument("--phase", choices=("pre", "post"), default="post")
    sub.choices["score"].add_argument("--scope", choices=("pilot", "main"), default="pilot")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    try:
        if a.cmd == "check":
            return cmd_check(cfg)
        if a.cmd == "run":
            return cmd_run(cfg, a.manifest, a.run_id)
        if a.cmd == "survey":
            return cmd_survey(cfg, a.run_id, a.phase)
        return cmd_score(cfg, a.run_id, a.scope)
    except (ServerError, ManifestMismatch) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
