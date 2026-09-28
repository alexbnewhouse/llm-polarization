"""docs/REPRODUCIBILITY.md §4a as code: verify one dialogue's record without running anything.

    python -m analysis.verify_dialogue --run-dir data/<run_id> --dyad-id <id> [--attempt N] [--no-surveys]
    python -m analysis.verify_dialogue --run-dir data/<run_id> --all

From manifest.json, dyads.jsonl and turns.jsonl alone it checks, for every message row of the attempt
(the latest complete one unless --attempt is given): the role's archived template source hashes to its
`template_sha256`; the row's `model_sha256` is the role's; the prompt rebuilt through
harness.transcript (the egocentric view) and harness.templates (the archived template, the run's `now`
and `enable_thinking`) hashes to the row's `prompt_sha256`; and the row's seed is
derive_seed(run_seed, dyad seed, dyad_id, attempt, turn, agent). With the instrument (verified by
sha256, analysis.outcomes.load_instrument) it does the same for the pre and post survey rows. It
reports the first mismatch and exits 1, or exits 0 when every row checks.

The manifest archives the template source but not the BOS/EOS token strings the harness read from the
GGUF. A template that prints `bos_token` or `eos_token` therefore cannot be rebuilt from the manifest
alone: pass them with --bos/--eos (the check warns when the template uses them)."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from harness.log import derive_seed, read_jsonl, sha256_text
from harness.scorer import latest_complete_attempts
from harness.templates import ChatTemplate, TemplateError, render
from harness.transcript import MENTOR, SEEKER, Transcript, message_order


class _Ctx:
    def __init__(self, root: Path, bos: str = "", eos: str = ""):
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        cfg = self.manifest.get("config") or {}
        self.run_seed = int(cfg.get("run_seed", 0))
        self.now = cfg.get("now", "2026-09-08")
        self.enable_thinking = bool((cfg.get("generation") or {}).get("enable_thinking", False))
        self.templates, self.warnings = {}, []
        for role in (SEEKER, MENTOR):
            m = self.manifest.get(role) or {}
            src = m.get("template_source")
            if src is None:
                raise ValueError(f"manifest.json {role} has no template_source")
            self.templates[role] = ChatTemplate.from_source(src, bos, eos)
            if ("bos_token" in src or "eos_token" in src) and not (bos or eos):
                self.warnings.append(f"{role} template uses bos_token/eos_token, which the manifest does "
                                     "not "
                                     "record; pass --bos/--eos if the prompts do not match")
        self.dyads = {(d["dyad_id"], int(d.get("attempt", 1))): d for d in read_jsonl(root / "dyads.jsonl")}
        self.turns = read_jsonl(root / "turns.jsonl")
        self.surveys = read_jsonl(root / "surveys.jsonl")
        self.status = read_jsonl(root / "status.jsonl")


def _mismatch(kind, row, expected, got, **extra) -> dict:
    return {"kind": kind, "dyad_id": row.get("dyad_id"), "attempt": row.get("attempt"),
            "turn": row.get("turn"),
            "agent": row.get("agent"), "phase": row.get("phase"), "item_id": row.get("item_id"),
            "expected": expected, "got": got, **extra}


def verify(ctx: _Ctx, dyad_id: str, attempt: int | None = None, items: list[dict] | None = None) -> dict:
    """Check one dyad attempt. Returns {ok, checked, first_mismatch, mismatches, attempt, notes}."""
    notes = list(ctx.warnings)
    if attempt is None:
        attempt = latest_complete_attempts(ctx.status).get(dyad_id)
        if attempt is None:
            seen = [int(r["attempt"]) for r in ctx.status if r["dyad_id"] == dyad_id]
            if not seen:
                return {"ok": False, "dyad_id": dyad_id, "checked": 0, "first_mismatch":
                        {"kind": "no_status", "dyad_id": dyad_id}, "mismatches": [], "notes": notes}
            attempt = max(seen)
            notes.append(f"no complete attempt; checking attempt {attempt}")
    dyad = ctx.dyads.get((dyad_id, attempt))
    if dyad is None:
        return {"ok": False, "dyad_id": dyad_id, "attempt": attempt, "checked": 0, "mismatches": [],
                "first_mismatch": {"kind": "no_dyad_row", "dyad_id": dyad_id, "attempt": attempt},
                "notes": notes}
    bad = []
    for role in (SEEKER, MENTOR):
        want = (ctx.manifest.get(role) or {}).get("template_sha256")
        if ctx.templates[role].sha256 != want:
            bad.append({"kind": "template_sha256", "agent": role, "expected": want,
                        "got": ctx.templates[role].sha256})
    rows = sorted((r for r in ctx.turns if r["dyad_id"] == dyad_id and int(r.get("attempt", 1)) == attempt),
                  key=message_order)
    t = Transcript(dyad_id, dyad["persona_text"], dyad.get("persona_reminder") or None, dyad["persona_mode"])
    checked = 0
    for r in rows:
        role_sha = (ctx.manifest.get(r["agent"]) or {}).get("model_sha256")
        if r.get("model_sha256") != role_sha:
            bad.append(_mismatch("model_sha256", r, role_sha, r.get("model_sha256")))
        if r.get("persona_mode") not in (None, dyad["persona_mode"]):
            bad.append(_mismatch("persona_mode", r, dyad["persona_mode"], r.get("persona_mode")))
        try:
            prompt = render(ctx.templates[r["agent"]], t.view_for(r["agent"]), now=ctx.now,
                            enable_thinking=ctx.enable_thinking)
            got = sha256_text(prompt)
        except TemplateError as e:
            got, prompt = f"render failed: {e}", ""
        if got != r.get("prompt_sha256"):
            bad.append(_mismatch("prompt_sha256", r, r.get("prompt_sha256"), got, prompt_chars=len(prompt),
                                 row_prompt_chars=r.get("prompt_chars")))
        seed = derive_seed(ctx.run_seed, int(dyad.get("seed", 0)), dyad_id, attempt, r["turn"], r["agent"])
        if r.get("seed") != seed:
            bad.append(_mismatch("seed", r, r.get("seed"), seed))
        checked += 1
        if not (r.get("error") or r.get("finish_reason") == "error"):
            t.append(r["turn"], r["agent"], r["text"])
    n_turns = int(dyad.get("n_turns", 0))
    if items is not None:
        by_id = {it["id"]: it for it in items}
        for s in (s for s in ctx.surveys if s["dyad_id"] == dyad_id and int(s.get("attempt", 1)) == attempt):
            it = by_id.get(s["item_id"])
            if it is None:
                bad.append(_mismatch("survey_item", s, "an item in the instrument", s["item_id"]))
                continue
            base = t.view_for(MENTOR) if s["phase"] == "post" else []
            messages = base + [{"role": "user", "content": it["text"]}]
            prompt = render(ctx.templates[MENTOR], messages, now=ctx.now,
                            enable_thinking=ctx.enable_thinking)
            if sha256_text(prompt) != s.get("prompt_sha256"):
                bad.append(_mismatch("survey_prompt_sha256", s, s.get("prompt_sha256"), sha256_text(prompt)))
            turn = 0 if s["phase"] == "pre" else n_turns + 1
            seed = derive_seed(ctx.run_seed, int(dyad.get("seed", 0)), dyad_id, attempt, turn,
                               f"survey:{s['phase']}:{s['item_id']}")
            if s.get("seed") != seed or s.get("turn") != turn:
                bad.append(_mismatch("survey_seed", s, s.get("seed"), seed))
            checked += 1
    elif any(s["dyad_id"] == dyad_id for s in ctx.surveys):
        notes.append("survey rows not checked (no instrument)")
    if not rows:
        bad.append({"kind": "no_turn_rows", "dyad_id": dyad_id, "attempt": attempt})
    return {"ok": not bad, "dyad_id": dyad_id, "attempt": attempt, "checked": checked,
            "first_mismatch": bad[0] if bad else None, "mismatches": bad, "notes": notes}


def _instrument_items(root: Path, override: str | None):
    """The instrument's items when a file matching the run's recorded sha256 can be found, else None."""
    from analysis.load import Run
    from analysis.outcomes import load_instrument
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    stub = Run(root, manifest.get("run_id", root.name), manifest, "", [], [], [], [], {})
    try:
        return load_instrument(stub, override).items, None
    except ValueError as e:
        return None, str(e)


def _describe(m: dict) -> str:
    where = " ".join(f"{k}={m[k]}" for k in ("dyad_id", "attempt", "turn", "agent", "phase", "item_id")
                     if m.get(k) is not None)
    exp, got = str(m.get("expected")), str(m.get("got"))
    return f"{m['kind']} at {where}: row has {exp[:16]}, rebuilt {got[:64]}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m analysis.verify_dialogue",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dyad-id")
    g.add_argument("--all", action="store_true", help="every dyad with a status row")
    p.add_argument("--attempt", type=int, default=None)
    p.add_argument("--no-surveys", action="store_true", help="check the dialogue rows only")
    p.add_argument("--batteries", default=None,
                   help="the instrument file, overriding manifest batteries.path")
    p.add_argument("--bos", default="")
    p.add_argument("--eos", default="")
    a = p.parse_args(argv)
    root = Path(a.run_dir)
    try:
        ctx = _Ctx(root, a.bos, a.eos)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    items = None
    if not a.no_surveys:
        items, why = _instrument_items(root, a.batteries)
        if items is None:
            print(f"warn: survey rows not checked: {why}", file=sys.stderr)
    ids = [a.dyad_id] if a.dyad_id else sorted({r["dyad_id"] for r in ctx.status})
    n_bad = 0
    for dyad_id in ids:
        res = verify(ctx, dyad_id, a.attempt, items)
        for n in res["notes"] if a.dyad_id else []:
            print(f"  note: {n}")
        if res["ok"]:
            if a.dyad_id:
                print(f"ok {dyad_id} attempt {res['attempt']}: {res['checked']} rows rebuilt and matched")
        else:
            n_bad += 1
            print(f"MISMATCH {dyad_id} attempt {res.get('attempt')}: {_describe(res['first_mismatch'])} "
                  f"({len(res['mismatches'])} mismatches in {res['checked']} rows)")
    if a.all:
        for w in ctx.warnings:
            print(f"  note: {w}")
        print(f"{len(ids) - n_bad}/{len(ids)} dyads verified")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
