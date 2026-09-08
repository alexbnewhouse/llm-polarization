"""CLI: check servers, run a dialogue manifest, re-administer surveys, score a run."""
from __future__ import annotations
import argparse, copy, json, os, platform, sys, threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import jinja2
from harness import __version__, log
from harness.client import LlamaClient, ServerError
from harness.dialogue import AgentHandle, DialogueError, DialogueRunner, DyadSpec, GenSettings
from harness.log import JsonlWriter, ManifestMismatch, RunPaths, now_iso, read_jsonl, run_paths
from harness.scorer import (JUDGE_N_PREDICT, JUDGE_SYSTEM, JUDGE_TASKS, JUDGE_TEMPERATURE, Scorer,
                            latest_complete_attempts)
from harness.survey import SurveyError, SurveyRunner, load_batteries
from harness.templates import (FIXTURE_MESSAGES, FIXTURE_MESSAGES_USER_FIRST, TemplateError, parity_check,
                               read_template_from_gguf, render)
from harness.transcript import MENTOR, PERSONA_MODES, SEEKER, Transcript

DEFAULT_CONFIG = {
    "data_dir": "data", "gguf_py_path": None, "batteries": "instruments/batteries.json", "run_seed": 0,
    "concurrency": None, "now": "2026-09-08",
    "generation": {"temperature": 0.7, "top_p": 0.95, "n_predict": 300, "timeout": 600, "enable_thinking": False},
    "seeker": {"url": None, "gguf_path": None}, "mentor": {"url": None, "gguf_path": None},
    "judge": {"url": None, "gguf_path": None},
}
HASH_CACHE = Path.home() / ".cache" / "llm-polarization" / "gguf-hashes.json"
HARNESS_DIR = Path(__file__).resolve().parent


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
    """Return a GGUF file's sha256, cached on disk keyed by path|size|mtime_ns so re-hashing a 20 GB file
    is skipped. Nanoseconds rather than whole seconds: a file rewritten within the same second, to the same
    size, would otherwise return the previous file's hash."""
    cache_file = cache_file or HASH_CACHE
    st = os.stat(path)
    key = f"{path}|{st.st_size}|{st.st_mtime_ns}"
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
                      "model_sha256": sha, "template_sha256": template.sha256,
                      # The template source, not only its hash: the template lives inside a 5-20 GB GGUF
                      # that git cannot hold, and a hash you cannot check anything against is not provenance.
                      "template_source": template.source,
                      # What the server says it is rendering with, which is not always what the GGUF holds
                      # (an arm started with --chat-template overrides it). None if /props does not say.
                      "server_chat_template": props.get("chat_template"),
                      "build_info": props.get("build_info", ""), "model_ftype": props.get("model_ftype"),
                      "total_slots": props.get("total_slots"), "default_generation_settings": props.get("default_generation_settings", {})}
    return handle, manifest_entry


CONTEXT_HEADROOM = 2048


def _context_budget(handle: AgentHandle, cfg: dict, props: dict, max_n_turns: int) -> tuple[str, bool | None, str]:
    """Does the longest dialogue in the manifest fit in one slot's context? A turn is two messages and each
    can run to n_predict tokens, so a dyad needs max_n_turns * 2 * n_predict plus headroom for the persona
    and the template's own wrapping. llama.cpp does not shift context by default: a slot that runs out stops
    with truncated=true and the next turn fails outright -- at turn 35 of 40, after a day of compute."""
    n_predict = int((cfg.get("generation") or {}).get("n_predict", 300))
    need = max_n_turns * 2 * n_predict + CONTEXT_HEADROOM
    n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
    detail = (f"about {need} tokens needed per dyad ({max_n_turns} turns x 2 messages x n_predict "
              f"{n_predict} + {CONTEXT_HEADROOM} headroom)")
    if not n_ctx:
        return ("context_budget", None, f"{detail}; the server does not report n_ctx")
    return ("context_budget", int(n_ctx) >= need, f"n_ctx {int(n_ctx)} per slot vs {detail}")


def check_agent(handle: AgentHandle, cfg: dict, max_n_turns: int | None = None) -> list[tuple[str, bool | None, str]]:
    """This agent's pre-flight, as (name, ok, detail) rows. `ok` is True, False (blocks a run) or None
    (a warning worth printing that does not block one):

    - `health`: the server answers.
    - `template_parity` / `template_parity_user_first`: our jinja2 rendering is byte-identical to the
      server's own, checked twice -- once with a leading system message and once with none at all, which
      is the mentor's only shape and exactly where a template's default system block would appear.
    - `server_chat_template`: whether the template the server reports at /props is the one we read out of
      the GGUF. A warning rather than a failure: the Olmo arm is deliberately served
      `--no-jinja --chat-template chatml`, and the parity rows above are the gate that matters.
    - `context_budget` (only when a dyad manifest is given): does the longest dialogue fit in a slot?
    - `trailing_system` (seeker only): does the template accept the persona reminder as a trailing
      system message?"""
    now = cfg.get("now", "2026-09-08")
    results: list[tuple[str, bool | None, str]] = [
        ("health", handle.client.health(), handle.client.url if hasattr(handle.client, "url") else "")]
    for name, messages in (("template_parity", FIXTURE_MESSAGES[:-1]),
                           ("template_parity_user_first", FIXTURE_MESSAGES_USER_FIRST)):
        try:
            ok, ours, theirs = parity_check(handle.template, handle.client, messages, now=now)
            results.append((name, ok, "" if ok else f"ours={ours[-120:]!r} theirs={theirs[-120:]!r}"))
        except TemplateError as e:
            results.append((name, False, str(e)))
    try:
        props = handle.client.props()
    except ServerError as e:
        props = {}
        results.append(("props", False, str(e)))
    served = props.get("chat_template")
    if served is None:
        results.append(("server_chat_template", None, "the server does not report one at /props"))
    elif served == handle.template.source:
        results.append(("server_chat_template", True, "matches the GGUF template"))
    else:
        results.append(("server_chat_template", None,
                        f"differs from the GGUF template (server {log.sha256_text(served)[:12]}, "
                        f"gguf {handle.template.sha256[:12]}) -- expected for an arm served with "
                        "--chat-template; template parity is the gate"))
    if max_n_turns:
        results.append(_context_budget(handle, cfg, props, max_n_turns))
    if handle.name == SEEKER:
        try:
            render(handle.template, FIXTURE_MESSAGES, now=now)
            results.append(("trailing_system", True, ""))
        except TemplateError as e:
            results.append(("trailing_system", False, str(e)))
    return results


def validate_manifest_rows(manifest_rows: list[dict]) -> None:
    """Reject a malformed dyad manifest before the run writes anything. A bad row discovered halfway
    through a wave has already spent the pre-survey of every dyad before it, and an empty reminder under
    persona_mode 'reinforced' is worse than a crash: it labels the dyad reinforced without reinforcing it."""
    seen: set[str] = set()
    for i, row in enumerate(manifest_rows, 1):
        for key in ("dyad_id", "persona_text", "n_turns"):
            if key not in row:
                raise ValueError(f"dyad manifest line {i}: missing {key!r}")
        dyad_id = row["dyad_id"]
        if dyad_id in seen:
            raise ValueError(f"dyad manifest line {i}: duplicate dyad_id {dyad_id!r}, which is the resume key")
        seen.add(dyad_id)
        mode = row.get("persona_mode", "reinforced")
        if mode not in PERSONA_MODES:
            raise ValueError(f"dyad manifest line {i} ({dyad_id}): persona_mode must be one of {PERSONA_MODES}, got {mode!r}")
        if int(row["n_turns"]) <= 0:
            raise ValueError(f"dyad manifest line {i} ({dyad_id}): n_turns must be positive, got {row['n_turns']!r}")
        if mode == "reinforced" and not str(row.get("persona_reminder") or "").strip():
            raise ValueError(f"dyad manifest line {i} ({dyad_id}): persona_mode 'reinforced' needs a non-empty "
                             "persona_reminder, or the dyad is logged as reinforced but never reinforced")


def plan_work(manifest_rows: list[dict], status_rows: list[dict]) -> list[tuple[DyadSpec, int]]:
    """Diff the dyad manifest against the run's status log and return the (spec, attempt) pairs still to run."""
    validate_manifest_rows(manifest_rows)
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
    batteries_sha256: str = ""


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
    surveys = SurveyRunner(ctx.run_id, ctx.run_seed, mentor, ctx.logs["surveys"], ctx.settings, clock=ctx.clock,
                           batteries_sha256=ctx.batteries_sha256)
    try:
        surveys.administer(spec, attempt, "pre", None, ctx.items)
        transcript = dialogue.run(spec, attempt)
        surveys.administer(spec, attempt, "post", transcript, ctx.items)
    except (DialogueError, SurveyError) as e:
        ctx.logs["status"].write({"run_id": ctx.run_id, "dyad_id": spec.dyad_id, "attempt": attempt, "status": "failed",
                                  "reason": str(e), "ts": ctx.clock()})
        return "failed"
    except Exception as e:  # noqa: BLE001 -- one malformed dyad must not take the whole pool down
        ctx.logs["status"].write({"run_id": ctx.run_id, "dyad_id": spec.dyad_id, "attempt": attempt, "status": "failed",
                                  "reason": f"{type(e).__name__}: {e}", "ts": ctx.clock()})
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


def cmd_check(cfg: dict, manifest_path: str | None = None) -> int:
    """Build every configured agent (the judge too, when `judge.url` is set: its output is grammar-forced,
    so a mis-rendered judge prompt still yields well-formed but meaningless scores) and print each one's
    check_agent rows. Returns 0 iff nothing FAILed; `warn` rows are printed and do not block. A role whose
    server is unreachable prints a FAIL health line instead of letting build_agent's ServerError traceback
    out, since a down server is exactly the failure `check` exists to report. With --manifest, also checks
    that the manifest's longest dialogue fits in a slot's context."""
    ok_all = True
    max_n_turns = None
    if manifest_path:
        max_n_turns = max((int(r.get("n_turns") or 0) for r in read_jsonl(Path(manifest_path))), default=0) or None
    if cfg[SEEKER].get("url") == cfg[MENTOR].get("url"):
        print(f"WARN seeker and mentor share {cfg[SEEKER].get('url')}: both pin the same slot, so every "
              "turn would evict the other agent's KV cache. `run` refuses this.")
    roles = [SEEKER, MENTOR] + (["judge"] if (cfg.get("judge") or {}).get("url") else [])
    for role in roles:
        try:
            handle, entry = build_agent(role, cfg[role], 0, cfg)
        except ServerError as e:
            print(f"FAIL health {role} {cfg[role].get('url')}: {e}")
            ok_all = False
            continue
        print(f"[{role}] {entry['alias']} {entry['model_path']} sha256={entry['model_sha256'][:12]} slots={entry['total_slots']}")
        for name, ok, detail in check_agent(handle, cfg, max_n_turns=None if role == "judge" else max_n_turns):
            ok_all = ok_all and ok is not False
            print(f"   {'ok  ' if ok else ('FAIL' if ok is False else 'warn')} {name} {detail}")
    return 0 if ok_all else 1


def _load_manifest(paths: RunPaths) -> dict:
    """Read a run's manifest.json, or refuse with the message a missing run directory deserves."""
    if not paths.manifest.exists():
        raise ManifestMismatch(f"{paths.manifest} does not exist; this run has never been started")
    return json.loads(paths.manifest.read_text(encoding="utf-8"))


def _verify_identity(existing: dict, entries: dict, roles=(SEEKER, MENTOR)) -> None:
    """Refuse to add rows to a run whose models are no longer the ones its manifest records. A GGUF
    re-quantized at the same path, a server restarted on another model, or a llama.cpp upgrade that changed
    the served template would otherwise be accepted in silence, and only the rows -- never the manifest --
    would carry the evidence."""
    for role in roles:
        was, now = existing.get(role) or {}, entries[role]
        for key in ("model_sha256", "template_sha256"):
            if was.get(key) and was[key] != now[key]:
                raise ManifestMismatch(f"{role}.{key} is now {now[key][:12]} but manifest.json records "
                                       f"{was[key][:12]}; use a new run_id")


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
    (dyad, attempt) work, and execute it across a thread pool with one slot per worker. Returns 0 when
    every dyad completed, 2 when any failed, 130 when Ctrl-C stopped it, 1 when it refused to start."""
    if cfg[SEEKER]["url"] == cfg[MENTOR]["url"]:
        print(f"error: seeker.url and mentor.url are both {cfg[SEEKER]['url']}; both agents pin the same "
              "slot, so they would evict each other's KV cache every turn. Serve them separately.",
              file=sys.stderr)
        return 1
    if cmd_check(cfg, manifest_path) != 0:
        print("check failed; not running", file=sys.stderr)
        return 1
    agents = _agents(cfg)
    (seeker, s_entry), (mentor, m_entry) = agents[SEEKER], agents[MENTOR]
    paths = run_paths(cfg["data_dir"], run_id)
    commit = _git_commit()
    if not commit:
        print("error: cannot read this repository's git commit; a run with unknown provenance is refused",
              file=sys.stderr)
        return 1
    manifest = {"run_id": run_id, "started_at": now_iso(), "harness_commit": commit,
                "harness_dirty": _git_dirty(), "config": cfg,
                "input_manifest": _file_provenance(manifest_path), "batteries": _batteries_provenance(cfg),
                "environment": _environment(cfg), "seeker": s_entry, "mentor": m_entry}
    if manifest["harness_dirty"]:
        print("WARN the working tree has uncommitted changes; manifest.harness_dirty is true")
    if paths.manifest.exists():
        # Resuming: the config matching is not enough. Re-verify that the servers are still serving the
        # models and templates this run started with.
        _verify_identity(_load_manifest(paths), {SEEKER: s_entry, MENTOR: m_entry})
    log.write_manifest(paths, manifest)
    concurrency = cfg["concurrency"] or min(int(s_entry["total_slots"] or 1), int(m_entry["total_slots"] or 1))
    rows = read_jsonl(Path(manifest_path))
    work = plan_work(rows, read_jsonl(paths.status))
    print(f"run {run_id}: {len(work)} of {len(rows)} dyads to run, concurrency {concurrency}")
    if not work:
        return 0
    logs = {k: JsonlWriter(getattr(paths, k)) for k in ("dyads", "status", "turns", "surveys")}
    ctx = RunContext(run_id, int(cfg["run_seed"]), seeker, mentor, _settings(cfg),
                     load_batteries(cfg["batteries"]), logs,
                     batteries_sha256=manifest["batteries"]["sha256"])
    free = list(range(concurrency))
    lock = threading.Lock()
    stop = threading.Event()

    def job(spec, attempt):
        """Run one (spec, attempt) job on a free slot, returning it to the pool when done. A job that has
        not started yet when Ctrl-C arrives returns immediately: that is what makes the interrupt stop the
        run rather than merely stop the operator watching it."""
        if stop.is_set():
            return "skipped"
        with lock:
            slot = free.pop()
        try:
            r = run_dyad(slot, spec, attempt, ctx)
            print(f"  {spec.dyad_id} attempt {attempt}: {r}", flush=True)
            return r
        except KeyboardInterrupt:
            # Ctrl-C reached this worker: raise the flag here, or the pool picks up the next dyad before
            # the main thread has noticed the interrupt at all.
            stop.set()
            raise
        finally:
            with lock:
                free.append(slot)

    interrupted = False
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(job, spec, attempt) for spec, attempt in work]
        try:
            results = [f.result() for f in futures]
        except KeyboardInterrupt:
            # Spec section 7: in-flight dyads finish, queued ones never start. `stop` covers the jobs the
            # pool has already handed to a thread; cancel_futures covers the ones it has not.
            interrupted = True
            stop.set()
            ex.shutdown(wait=True, cancel_futures=True)
            results = [f.result() for f in futures if f.done() and not f.cancelled() and not f.exception()]
    complete, failed = results.count("complete"), results.count("failed")
    print(f"done: {complete} complete, {failed} failed" +
          (f", {len(work) - complete - failed} not run" if interrupted else ""))
    if interrupted:
        print(f"stopped by Ctrl-C; re-run with --run-id {run_id} to resume", file=sys.stderr)
        return 130
    return 2 if failed else 0


def cmd_survey(cfg: dict, run_id: str, phase: str) -> int:
    """Run the `survey` subcommand: re-administer one survey phase for every complete dyad in a run,
    rebuilding the transcript from turns.jsonl for the post phase. Only the mentor is built: the seeker's
    server has nothing to do with a re-administration and need not even be running."""
    mentor, entry = _agents(cfg, roles=(MENTOR,))[MENTOR]
    paths = run_paths(cfg["data_dir"], run_id)
    _verify_identity(_load_manifest(paths), {MENTOR: entry}, roles=(MENTOR,))
    items = load_batteries(cfg["batteries"])
    complete = latest_complete_attempts(read_jsonl(paths.status))
    dyads = {(d["dyad_id"], d["attempt"]): d for d in read_jsonl(paths.dyads)}
    turns = read_jsonl(paths.turns)
    runner = SurveyRunner(run_id, int(cfg["run_seed"]), _with_slot(mentor, 0), JsonlWriter(paths.surveys),
                          _settings(cfg), batteries_sha256=log.sha256_file(cfg["batteries"]))
    n = 0
    for dyad_id, attempt in complete.items():
        spec = DyadSpec.from_row(dyads[(dyad_id, attempt)])
        transcript = None
        if phase == "post":
            transcript = _rebuild_transcript(dyads[(dyad_id, attempt)],
                                             [r for r in turns if r["dyad_id"] == dyad_id and r.get("attempt", 1) == attempt])
        try:
            n += len(runner.administer(spec, attempt, phase, transcript, items, origin="readministered"))
        except SurveyError as e:
            print(f"  {dyad_id}: {e}", file=sys.stderr)
    print(f"{phase} survey: {n} rows for {len(complete)} dyads")
    return 0


def cmd_score(cfg: dict, run_id: str, scope: str) -> int:
    """Run the `score` subcommand: build the judge agent and score the run's not-yet-scored targets."""
    if not cfg["judge"].get("url"):
        print("config needs judge.url", file=sys.stderr)
        return 1
    judge, entry = build_agent("judge", cfg["judge"], 0, cfg)
    ok_all = True
    for name, ok, detail in check_agent(judge, cfg):
        ok_all = ok_all and ok is not False
        print(f"   {'ok  ' if ok else ('FAIL' if ok is False else 'warn')} judge {name} {detail}")
    if not ok_all:
        print("judge check failed; not scoring", file=sys.stderr)
        return 1
    paths = run_paths(cfg["data_dir"], run_id)
    manifest = _load_manifest(paths)
    write_judge_manifest(paths, entry, scope, judge.template.source)
    n = Scorer(run_id, int(cfg["run_seed"]), judge, JsonlWriter(paths.scores), _settings(cfg),
               harness_commit=_git_commit()).score_run(paths, scope, manifest)
    print(f"scored {n} new rows ({scope})")
    return 0


def write_judge_manifest(paths: RunPaths, entry: dict, scope: str, template_source: str) -> Path:
    """Record the judge's provenance beside the run, in its own small file. It cannot go into
    manifest.json: that file is written once when the run starts and is deliberately never rewritten, and
    scoring happens later -- often from a different harness commit and against a model the run never saw."""
    judge = dict(entry)
    judge.update({"template_source": template_source, "scope": scope,
                  "temperature": JUDGE_TEMPERATURE, "n_predict": JUDGE_N_PREDICT,
                  "judge_system": JUDGE_SYSTEM, "judge_tasks": JUDGE_TASKS,
                  "harness_commit": _git_commit(), "ts": now_iso()})
    path = paths.root / f"judge-{entry['model_sha256'][:12]}.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if {k: v for k, v in old.items() if k != "ts"} == {k: v for k, v in judge.items() if k != "ts"}:
            return path
        # Same judge, different scoring pass (another scope, or another harness commit): keep both records.
        path = paths.root / f"judge-{entry['model_sha256'][:12]}-{scope}.json"
    path.write_text(json.dumps(judge, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _run_text(cmd: list[str], cwd: Path | None = None) -> str | None:
    """Run a read-only command and return its stripped output, or None if it is absent, fails or hangs.
    Every caller is provenance capture: a missing tool must leave a null in the record, not stop a run."""
    try:
        import subprocess
        return subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, stderr=subprocess.DEVNULL,
                                       text=True, timeout=15).strip()
    except Exception:  # noqa: BLE001
        return None


def _git(args: list[str], cwd: Path) -> str | None:
    """Run a read-only git command in `cwd`; return its stripped output, or None if git could not answer."""
    return _run_text(["git", *args], cwd)


def _gpu() -> str | None:
    """The GPU and driver, best effort: whatever nvidia-smi, rocm-smi or /sys/class/drm will say. The
    backend and driver change the numerics, so a reviewer wants them even though nothing depends on them."""
    for cmd in (["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                ["rocm-smi", "--showproductname"]):
        out = _run_text(cmd)
        if out:
            return " | ".join(line.strip() for line in out.splitlines() if line.strip())[:500]
    cards = []
    for uevent in sorted(Path("/sys/class/drm").glob("card*/device/uevent")):
        try:
            fields = dict(l.split("=", 1) for l in uevent.read_text().splitlines() if "=" in l)
        except OSError:
            continue
        if fields.get("DRIVER"):
            cards.append(f"{uevent.parts[-3]}: {fields['DRIVER']} {fields.get('PCI_ID', '')}".strip())
    return " | ".join(cards) or None


def _environment(cfg: dict) -> dict:
    """The software the prompts were built by. Every prompt in the study is rendered by jinja2, so a jinja2
    upgrade that changed whitespace handling would change every prompt; the version has to be in the record
    even though `check`'s parity test would catch such a change before a run."""
    gguf_py = cfg.get("gguf_py_path")
    return {"python": sys.version, "platform": platform.platform(), "jinja2": jinja2.__version__,
            "harness_version": __version__, "gguf_py_path": gguf_py,
            "gguf_py_commit": _git(["rev-parse", "HEAD"], Path(gguf_py)) if gguf_py else None,
            "gpu": _gpu()}


def _git_commit() -> str:
    """The commit of THIS harness, read in the harness's own directory rather than the process's working
    directory -- running the CLI from elsewhere used to record whatever repository happened to be there."""
    return _git(["rev-parse", "HEAD"], HARNESS_DIR) or ""


def _git_dirty() -> bool:
    """True when the harness's repository has uncommitted changes, so harness_commit does not fully
    describe the code that ran. Recorded, not refused: the pilot may legitimately run from a dirty tree."""
    out = _git(["status", "--porcelain"], HARNESS_DIR)
    return bool(out)


def _file_provenance(path: str | Path) -> dict:
    """{path, sha256} for an input file, so 'the dyads we meant to run' is separable from what started."""
    p = Path(path)
    return {"path": str(p), "sha256": log.sha256_file(p) if p.exists() else ""}


def _batteries_provenance(cfg: dict) -> dict:
    """Pin the survey instrument to the run: its path, its sha256, and the item ids in file order (which is
    the administration order, pre and post). The wording is a placeholder until the US-adaptation task
    lands, so a run that cannot name the exact file it used cannot be interpreted afterwards."""
    items = load_batteries(cfg["batteries"])
    prov = _file_provenance(cfg["batteries"])
    prov.update({"n_items": len(items), "item_ids": [it["id"] for it in items]})
    return prov


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse the check/run/survey/score subcommands, dispatch to the matching cmd_* function,
    and turn a ServerError or ManifestMismatch into a clean error line and exit code 1 instead of a traceback."""
    ap = argparse.ArgumentParser(prog="harness", description="Dyad harness for the LLM polarization study")
    sub = ap.add_subparsers(dest="cmd", required=True)
    parsers = {}
    for name in ("check", "run", "survey", "score"):
        parsers[name] = p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name != "check":
            p.add_argument("--run-id", required=True)
    parsers["check"].add_argument("--manifest", help="dyad manifest; adds the context-budget check")
    parsers["run"].add_argument("--manifest", required=True)
    parsers["survey"].add_argument("--phase", choices=("pre", "post"), default="post")
    parsers["score"].add_argument("--scope", choices=("pilot", "main"), default="pilot")
    a = ap.parse_args(argv)
    try:
        cfg = load_config(a.config)
        if a.cmd == "check":
            return cmd_check(cfg, a.manifest)
        if a.cmd == "run":
            return cmd_run(cfg, a.manifest, a.run_id)
        if a.cmd == "survey":
            return cmd_survey(cfg, a.run_id, a.phase)
        return cmd_score(cfg, a.run_id, a.scope)
    except (ServerError, ManifestMismatch, ValueError) as e:
        # Everything the operator can get wrong -- a dead server, a changed model, a malformed manifest or
        # config -- becomes one error line and exit 1, never a traceback.
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
