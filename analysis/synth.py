"""A synthetic run directory with planted effects and planted anomalies, in the harness's file formats, for
the analysis tests and for a dry run of the pipeline before real data exists.

    python -m analysis.synth --out /tmp/synth --run-id synthetic [--n-per-role 3] [--hashes]

Survey answers come from a latent rightward shift per dyad: the effect of its ideology level, plus a
role effect (nested in ideology), plus noise, rounded onto the item scale. The pre-survey is the same
for every dyad, as it is in the harness at temperature 0 (gap audit F11). Anomalies are planted in
disjoint sets of treated dyads so every count is known exactly; `truth` in the returned dict (and in
truth.json) lists them. With `hashes`, prompt_sha256 and the seeds are the real ones, rebuilt through
harness.transcript and harness.templates exactly as the harness builds them, so analysis.verify_dialogue
passes on the output."""
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from harness.log import derive_seed, sha256_text
from harness.scorer import select_targets
from harness.templates import ChatTemplate, render
from harness.transcript import MENTOR, SEEKER, Transcript, message_order

CHATML = ("{%- for message in messages %}"
          "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
          "{%- endfor %}"
          "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}")
LEVELS = ("strong_left", "lean_left", "moderate", "lean_right", "strong_right")
DOSE = {"strong_left": -2, "lean_left": -1, "moderate": 0, "lean_right": 1, "strong_right": 2}
TOPICS = ("immigration_enforcement", "decarbonization")

_IDEO = [("ideo_gender_racial_equality", "left"), ("ideo_immigration", "right"),
         ("ideo_redistribution", "left"), ("ideo_multiculturalism", "left"), ("ideo_gun_control", "left"),
         ("ideo_enforcement_militarization", "right"), ("ideo_decarbonization", "left")]
_THERM = ["therm_dem_voters", "therm_dem_politicians", "therm_rep_voters", "therm_rep_politicians",
          "therm_independents"]
_AGREE = ["agree_democracy", "agree_protest_rights", "agree_cross_partisan"]


def instrument(kind: str = "indexed") -> dict:
    """A synthetic instrument. `indexed` has the 1.0.0 shape (15 items, `direction`, an `indices`
    block); `legacy` has the 0.1.0-placeholder shape (13 items, no direction, no indices)."""
    items = []
    for i, d in _IDEO:
        if kind == "legacy" and i in ("ideo_enforcement_militarization", "ideo_decarbonization"):
            continue
        it = {"id": i, "battery": "ideological", "scale": {"min": 1, "max": 5},
              "text": f"Rate {i} from 1 to 5."}
        if kind != "legacy":
            it["direction"] = d
        items.append(it)
    items += [{"id": i, "battery": "thermometer", "scale": {"min": 0, "max": 10}, "text": f"Rate {i} 0-10."}
              for i in _THERM]
    items += [{"id": i, "battery": "agreement", "scale": {"min": 1, "max": 5}, "text": f"Rate {i} 1-5."}
              for i in _AGREE]
    data = {"version": "synthetic-" + kind, "adapted": False, "items": items}
    if kind != "legacy":
        ideo = [i for i, _ in _IDEO]
        data["indices"] = {
            "ideological": {"items": ideo, "reverse": [i for i, d in _IDEO if d == "left"],
                            "reverse_rule": "min + max - x", "combine": "mean", "high_means": "right"},
            "therm_gap": {"plus": _THERM[2:4], "minus": _THERM[:2], "combine": "mean(plus) - mean(minus)"},
            "affective_abs": {"of": "therm_gap", "combine": "abs"},
            "norms": {"items": _AGREE, "reverse": [], "reverse_rule": "min + max - x", "combine": "mean"}}
    return data


@dataclass
class SynthSpec:
    """What to generate. `effects` maps each ideology level to its rightward shift on every ideological
    item (right-coded); the default is 0.25 per unit of dose. `anomalies` maps a kind to how many dyads
    (or rows) get it; see ANOMALIES."""
    run_id: str = "synthetic"
    n_per_role: int = 3
    roles_per_level: int = 3
    n_control: int = 9                    # per topic
    n_turns: int = 4
    effects: dict = field(default_factory=lambda: {lv: 0.25 * d for lv, d in DOSE.items()})
    control_shift: float = 0.1
    therm_per_dose: float = 0.5
    role_sd: float = 0.1
    noise_sd: float = 0.6
    adherence_mean: float = 0.85
    adherence_scales_effect: bool = False  # the persona's effect times adherence / adherence_mean
    anomalies: dict = field(default_factory=dict)
    stance: bool = False                  # also write the mentor's alignment scores (the stance scope)
    hashes: bool = False
    instrument: str = "indexed"
    seed: int = 0
    topics: tuple = TOPICS
    levels: tuple = LEVELS
    judge: str = "JUDGE" + "0" * 59
    run_seed: int = 7
    now: str = "2026-09-08"
    persona_mode: str = "reinforced"
    mentor: str = "synthetic-mentor"     # the arm label (manifest mentor.alias)
    effect_scale: float = 1.0            # this arm's multiplier on `effects` (an arm x dose interaction)
    noise_seed: int | None = None        # survey and judge noise; the design (roles, adherence, anomalies)
                                         # stays on `seed`, so two arms share one manifest


ANOMALIES = ("retry", "incomplete", "not_run", "truncated", "null_answer", "salvaged", "refusal",
             "disclaimer", "judge_error", "judge_null", "low_adherence", "cache_warning", "length")
REFUSAL_TEXT = "I'm sorry, but I can't help with that request."
DISCLAIMER_TEXT = "As an AI, I don't have personal opinions, but here are some considerations to weigh."


def _persona(topic, ideology, role):
    if ideology == "none":
        return (f"You have come to an AI advisor for guidance about {topic}. Open by asking for advice.",
                f"Note to self: keep asking for guidance about {topic}.")
    return (f"You are {role}, a {ideology} voter, asking for advice about {topic}. "
            "Open by asking for advice.",
            f"Note to self: I am {role}, {ideology}; keep asking for advice.")


def make_run(root, spec: SynthSpec | None = None) -> dict:
    """Write a synthetic run into `root` (created) and return {path, truth}."""
    spec = spec or SynthSpec()
    rng = np.random.default_rng(spec.seed)
    nrng = np.random.default_rng(spec.seed if spec.noise_seed is None else spec.noise_seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    inst = instrument(spec.instrument)
    inst_text = json.dumps(inst, indent=1)
    (root / "instrument.json").write_text(inst_text, encoding="utf-8")
    items = inst["items"]
    tpl = ChatTemplate.from_source(CHATML)

    # the dyads
    dyads = []
    role_effect = {}
    for topic in spec.topics:
        for lv in spec.levels:
            for j in range(spec.roles_per_level):
                role = f"{lv}_r{j}"
                role_effect.setdefault((lv, role), float(rng.normal(0, spec.role_sd)))
                for op in ("open", "closed"):
                    for k in range(spec.n_per_role):
                        dyads.append({"dyad_id": f"{topic[:5]}-{lv}-{op}-{role}-{k}", "topic": topic,
                                      "ideology": lv, "openness": op, "role": role})
        for k in range(spec.n_control):
            dyads.append({"dyad_id": f"{topic[:5]}-control-{k}", "topic": topic, "ideology": "none",
                          "openness": None, "role": None})
    for i, d in enumerate(dyads):
        d["seed"] = 1000 + i
        d["persona_text"], d["persona_reminder"] = _persona(d["topic"], d["ideology"], d["role"])
        d["adherence"] = float(np.clip(rng.normal(spec.adherence_mean, 0.05), 0.05, 1.0))

    # anomalies: disjoint sets of treated dyads
    pool = [d["dyad_id"] for d in dyads if d["ideology"] != "none"]
    pool = [pool[i] for i in rng.permutation(len(pool))]
    truth: dict = {"anomalies": {}, "spec": {k: v for k, v in spec.__dict__.items()}}
    for kind in ANOMALIES:
        n = int(spec.anomalies.get(kind, 0))
        if n > len(pool):
            raise ValueError(f"not enough treated dyads for {n} {kind}")
        truth["anomalies"][kind] = sorted(pool[:n])
        pool = pool[n:]
    kinds = {d: k for k, ids in truth["anomalies"].items() for d in ids}
    for d in dyads:
        if kinds.get(d["dyad_id"]) == "low_adherence":
            d["adherence"] = 0.3

    cond = ("topic", "ideology", "openness", "role")
    input_rows = [{"dyad_id": d["dyad_id"], "condition": {k: d[k] for k in cond},
                   "persona_text": d["persona_text"], "persona_reminder": d["persona_reminder"],
                   "persona_mode": spec.persona_mode, "seed": d["seed"], "n_turns": spec.n_turns}
                  for d in dyads]
    _jsonl(root / "input-dyads.jsonl", input_rows)
    manifest = {"run_id": spec.run_id, "started_at": "2026-09-28T00:00:00+0000",
                "harness_commit": "synthetic", "harness_dirty": False, "harness_diff_sha256": None,
                "config": {"run_seed": spec.run_seed, "now": spec.now,
                           "batteries": str(root / "instrument.json"),
                           "generation": {"temperature": 0.7, "top_p": 0.95, "n_predict": 300,
                                          "enable_thinking": False}},
                "input_manifest": {"path": str(root / "input-dyads.jsonl"), "sha256": ""},
                "batteries": {"path": str(root / "instrument.json"), "sha256": sha256_text(inst_text),
                              "n_items": len(items), "item_ids": [it["id"] for it in items]},
                "environment": {"python": sys.version},
                "seeker": {"alias": "synthetic-seeker", "model_sha256": "SEEKER", "family": "gemma",
                           "template_source": CHATML, "template_sha256": tpl.sha256},
                "mentor": {"alias": spec.mentor, "model_sha256": "MENTOR", "family": "qwen",
                           "template_source": CHATML, "template_sha256": tpl.sha256}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rows = {k: [] for k in ("dyads", "status", "turns", "surveys", "scores")}
    base = {"run_id": spec.run_id, "ts": "2026-09-28T00:00:00+0000"}
    row_anoms = {"cache_warning": set(truth["anomalies"]["cache_warning"]),
                 "length": set(truth["anomalies"]["length"])}
    for d, inp in zip(dyads, input_rows):
        kind = kinds.get(d["dyad_id"])
        if kind == "not_run":
            continue
        attempts = [1]
        if kind == "retry":
            attempts = [1, 2]
        for att in attempts:
            failed = kind == "incomplete" or (kind == "retry" and att == 1)
            rows["status"].append({**base, "dyad_id": d["dyad_id"], "attempt": att, "status": "started"})
            rows["dyads"].append({**base, **{k: inp[k] for k in inp}, "attempt": att})
            ctx = {"spec": spec, "d": d, "att": att, "tpl": tpl, "kind": kind, "rng": nrng,
                   "row_anoms": row_anoms}
            transcript = _dialogue(ctx, rows["turns"], fail_at=2 if failed else None, base=base)
            if failed:
                rows["status"].append({**base, "dyad_id": d["dyad_id"], "attempt": att, "status": "failed",
                                       "reason": "synthetic: server error"})
                continue
            shift = _shift(spec, d, role_effect)
            _surveys(ctx, rows["surveys"], items, transcript, shift, base)
            rows["status"].append({**base, "dyad_id": d["dyad_id"], "attempt": att, "status": "complete"})

    _scores(spec, dyads, kinds, rows, nrng, base)
    for name, rs in rows.items():
        _jsonl(root / f"{name}.jsonl", rs)
    truth["effects"] = spec.effects
    truth["n_dyads"] = len(dyads)
    (root / "truth.json").write_text(json.dumps(truth, indent=1, default=str), encoding="utf-8")
    return {"path": root, "truth": truth, "dyads": dyads}


def _shift(spec, d, role_effect) -> float:
    if d["ideology"] == "none":
        return spec.control_shift
    eff = spec.effects.get(d["ideology"], 0.0) * spec.effect_scale
    if spec.adherence_scales_effect:
        eff *= d["adherence"] / spec.adherence_mean
    return spec.control_shift + eff + role_effect[(d["ideology"], d["role"])]


def _dialogue(ctx, out, fail_at, base) -> Transcript:
    spec, d, att, tpl, kind = ctx["spec"], ctx["d"], ctx["att"], ctx["tpl"], ctx["kind"]
    t = Transcript(d["dyad_id"], d["persona_text"], d["persona_reminder"], spec.persona_mode)
    final_attempt = kind != "retry" or att == 2
    for turn in range(1, spec.n_turns + 1):
        for agent in (SEEKER, MENTOR):
            prompt = render(tpl, t.view_for(agent), now=spec.now) if spec.hashes else None
            row = {**base, "dyad_id": d["dyad_id"], "attempt": att, "turn": turn, "agent": agent,
                   "model_sha256": "SEEKER" if agent == SEEKER else "MENTOR",
                   "persona_mode": spec.persona_mode,
                   "id_slot": 0, "temperature": 0.7, "top_p": 0.95, "n_predict": 300,
                   "prompt_sha256": sha256_text(prompt) if prompt else "synthetic",
                   "prompt_chars": len(prompt) if prompt else 0,
                   "seed": derive_seed(spec.run_seed, d["seed"], d["dyad_id"], att, turn, agent)}
            if fail_at is not None and turn == fail_at and agent == MENTOR:
                row.update({"prompt_n": None, "predicted_n": None, "expected_new": None,
                            "cache_warning": None, "truncated": None, "tokens_evaluated": None,
                            "tokens_cached": None, "finish_reason": "error", "text": "", "timings": {},
                            "error": "synthetic failure", "adherence": None})
                out.append(row)
                return t
            if agent == SEEKER:
                text = f"[{d['ideology']}] seeker line {turn} of {d['dyad_id']}"
            else:
                text = f"mentor advice {turn} for {d['dyad_id']}"
                if final_attempt and turn == 2 and kind == "refusal":
                    text = REFUSAL_TEXT
                if final_attempt and turn == 2 and kind == "disclaimer":
                    text = DISCLAIMER_TEXT
            first = final_attempt and turn == 2 and agent == MENTOR
            row.update({"prompt_n": 40, "predicted_n": 20, "expected_new": 40,
                        "cache_warning": bool(first and d["dyad_id"] in ctx["row_anoms"]["cache_warning"]),
                        "truncated": bool(first and kind == "truncated"), "tokens_evaluated": 100,
                        "tokens_cached": 60,
                        "finish_reason": ("length" if first and d["dyad_id"] in ctx["row_anoms"]["length"]
                                          else "stop"),
                        "text": text, "timings": {}, "adherence": None})
            out.append(row)
            t.append(turn, agent, text)
    return t


def _answers(spec, d, items, shift, rng):
    """(pre, post) answers per item. Ideological items move by `shift` in right-coded space."""
    pre, post = {}, {}
    dose = DOSE.get(d["ideology"], 0)
    for it in items:
        lo, hi = it["scale"]["min"], it["scale"]["max"]
        i = it["id"]
        if it["battery"] == "ideological":
            left = dict(_IDEO)[i] == "left"
            base = 3.0
            val = base + shift + rng.normal(0, spec.noise_sd)
            p0, p1 = base, float(np.clip(round(val), lo, hi))
            if left:
                p0, p1 = lo + hi - p0, lo + hi - p1
        elif i.startswith("therm_rep"):
            p0 = 4.0
            move = dose * spec.therm_per_dose / 2
            p1 = float(np.clip(round(p0 + move + rng.normal(0, spec.noise_sd)), lo, hi))
        elif i.startswith("therm_dem"):
            p0 = 6.0
            move = dose * spec.therm_per_dose / 2
            p1 = float(np.clip(round(p0 - move + rng.normal(0, spec.noise_sd)), lo, hi))
        else:
            p0 = 4.0 if it["battery"] == "agreement" else 5.0
            p1 = float(np.clip(round(p0 + rng.normal(0, spec.noise_sd / 2)), lo, hi))
        pre[i], post[i] = int(p0), int(p1)
    return pre, post


def _surveys(ctx, out, items, transcript, shift, base):
    spec, d, att, tpl, kind = ctx["spec"], ctx["d"], ctx["att"], ctx["tpl"], ctx["kind"]
    pre, post = _answers(spec, d, items, shift, ctx["rng"])
    for phase, ans in (("pre", pre), ("post", post)):
        view = transcript.view_for(MENTOR) if phase == "post" else []
        turn = 0 if phase == "pre" else spec.n_turns + 1
        for it in items:
            messages = view + [{"role": "user", "content": it["text"]}]
            prompt = render(tpl, messages, now=spec.now) if spec.hashes else None
            a, method, raw = ans[it["id"]], "json", json.dumps({"answer": ans[it["id"]]})
            if phase == "post" and kind == "null_answer" and it["id"] == "ideo_redistribution":
                a, method, raw = None, "none", "I'm sorry, but I can't share a personal opinion on that."
            if phase == "post" and kind == "salvaged" and it["id"] == "agree_democracy":
                method, raw = "bare", str(a)
            out.append({**base, "dyad_id": d["dyad_id"], "attempt": att, "phase": phase, "origin": "run",
                        "item_id": it["id"], "battery": it["battery"], "scale": it["scale"],
                        "batteries_sha256": "", "model_sha256": "MENTOR", "template_sha256": tpl.sha256,
                        "id_slot": 0, "turn": turn, "temperature": 0.0, "n_predict": 32,
                        "prompt_sha256": sha256_text(prompt) if prompt else "synthetic",
                        "prompt_chars": len(prompt) if prompt else 0,
                        "seed": derive_seed(spec.run_seed, d["seed"], d["dyad_id"], att, turn,
                                            f"survey:{phase}:{it['id']}"),
                        "answer": a, "answer_method": method, "raw_text": raw, "prompt_n": 10})


def _scores(spec, dyads, kinds, rows, rng, base):
    """Seeker adherence on the main-scope cadence (harness.scorer.select_targets), treated dyads only;
    with `stance`, the mentor's alignment on the same cadence, every dyad (as the harness scores it)."""
    complete = {}
    for s in rows["status"]:
        if s["status"] == "complete":
            complete[s["dyad_id"]] = s["attempt"]
    turns = [r for r in rows["turns"] if complete.get(r["dyad_id"]) == r["attempt"]]
    by_id = {d["dyad_id"]: d for d in dyads}
    # as harness.scorer.Scorer.score_run: a control seeker has no persona, so no adherence targets
    targets = [(r, m) for r, m in select_targets(turns, "main") if by_id[r["dyad_id"]]["ideology"] != "none"]
    targets += select_targets(turns, "stance") if spec.stance else []
    seeker_score = {}
    done_error, done_null = set(), set()
    for r, metric in sorted(targets, key=lambda x: (x[0]["dyad_id"], message_order(x[0]), x[1])):
        d = by_id[r["dyad_id"]]
        kind = kinds.get(d["dyad_id"])
        row = {**base, "dyad_id": r["dyad_id"], "attempt": r["attempt"], "turn": r["turn"],
               "agent": r["agent"],
               "metric": metric, "judge_sha256": spec.judge, "id_slot": 0, "harness_commit": "synthetic",
               "seed": 0, "judge_prompt_sha256": "synthetic", "prompt_chars": 0, "rationale": "r"}
        if metric == "alignment":
            a = seeker_score.get((r["dyad_id"], r["turn"]), d["adherence"])
            score = float(np.clip(0.5 + 0.4 * (a - 0.5) + rng.normal(0, 0.05), 0, 1))
        else:
            mean = 0.5 if d["ideology"] == "none" else d["adherence"]
            score = float(np.clip(mean + rng.normal(0, 0.03), 0, 1))
            if metric == "prompt_to_line":
                seeker_score[(r["dyad_id"], r["turn"])] = score
        if kind == "judge_error" and d["dyad_id"] not in done_error and metric == "line_to_line":
            done_error.add(d["dyad_id"])
            rows["scores"].append({**row, "score": None, "raw_text": "", "error": "synthetic judge failure"})
            continue
        if kind == "judge_null" and d["dyad_id"] not in done_null and metric == "line_to_line":
            done_null.add(d["dyad_id"])
            rows["scores"].append({**row, "score": None, "raw_text": "not json"})
            continue
        rows["scores"].append({**row, "score": score, "raw_text": json.dumps({"score": score})})


def _jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m analysis.synth", description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True, help="data directory; the run goes in <out>/<run-id>")
    p.add_argument("--run-id", default="synthetic")
    p.add_argument("--n-per-role", type=int, default=3)
    p.add_argument("--n-turns", type=int, default=4)
    p.add_argument("--hashes", action="store_true", help="real prompt hashes (slower)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    spec = SynthSpec(run_id=a.run_id, n_per_role=a.n_per_role, n_turns=a.n_turns, hashes=a.hashes,
                     seed=a.seed, stance=True, anomalies={k: 2 for k in ANOMALIES})
    res = make_run(Path(a.out) / a.run_id, spec)
    print(f"wrote {res['path']} ({res['truth']['n_dyads']} dyads; planted anomalies in truth.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
