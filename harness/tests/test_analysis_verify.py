"""analysis.verify_dialogue (REPRODUCIBILITY §4a as code) on a real harness run and on synthetic runs, and
the analysis pipeline end to end on rows the harness itself wrote (FakeClient servers, as in test_run)."""
import json
import re
from pathlib import Path
import pytest
from analysis import verify_dialogue as V
from analysis.estimate import estimate
from analysis.load import DOSE, load_run
from analysis.outcomes import FALLBACK_DIRECTIONS, compute_outcomes, load_instrument
from analysis.synth import SynthSpec, make_run
from harness import run as R
from harness.templates import ChatTemplate
from harness.tests.conftest import CHATML
from harness.tests.fakes import FakeClient

REPO = Path(__file__).resolve().parents[2]
ITEMS = json.loads((REPO / "instruments" / "batteries.json").read_text())["items"]
TAG = re.compile(r"IDEO=(\w+)")


class Seeker(FakeClient):
    """Says which ideology its persona has, so the mentor can react to it."""
    def complete(self, prompt, **kw):
        m = TAG.search(prompt)
        tag = m.group(1) if m else "none"
        self.replies = [f"As a {tag} voter, IDEO={tag} here."]
        return super().complete(prompt, **kw)


class Mentor(FakeClient):
    """Answers survey items as a mentor that moves one scale point per two dose steps toward the seeker:
    pre (no context) at the midpoint, post shifted by the persona the seeker announced."""
    def complete(self, prompt, **kw):
        if kw.get("json_schema"):
            item = next(it for it in ITEMS if it["text"] in prompt[-600:])
            lo, hi = item["scale"]["min"], item["scale"]["max"]
            mid = (lo + hi) // 2
            tags = TAG.findall(prompt)
            shift = round(DOSE.get(tags[-1], 0) / 2) if tags else 0
            if item["battery"] == "ideological":
                right = FALLBACK_DIRECTIONS.get(item["id"], item.get("direction")) == "right"
                ans = mid + shift if right else mid - shift
            else:
                ans = mid
            self.replies = [json.dumps({"answer": ans})]
        else:
            self.replies = ["mentor advice"]
        return super().complete(prompt, **kw)


@pytest.fixture
def harness_run(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "read_template_from_gguf",
                        lambda path, gguf_py_path=None: ChatTemplate.from_source(CHATML))
    monkeypatch.setattr(R, "model_sha256_cached",
                        lambda path, cache_file=None: "HASH-" + path.split("/")[-1])

    def factory(url, timeout=None):
        cls = {"http://s": Seeker, "http://m": Mentor}.get(url, FakeClient)
        c = cls(['{"score": 0.8, "rationale": "r"}'])
        c.props = lambda: {"model_path": f"/{url[7:]}.gguf", "total_slots": 2, "build_info": "b",
                           "model_alias": url[7:], "default_generation_settings": {}}
        return c
    monkeypatch.setattr(R, "LlamaClient", factory)
    cfg = {"data_dir": str(tmp_path / "data"), "gguf_py_path": None, "run_seed": 5, "now": "2026-09-08",
           "batteries": str(REPO / "instruments" / "batteries.json"), "grid": None, "concurrency": 2,
           "seeker": {"url": "http://s"}, "mentor": {"url": "http://m", "family": "qwen"},
           "judge": {"url": "http://j", "family": "gemma"}}
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    rows = []
    for i, lv in enumerate(("strong_left", "moderate", "strong_right", "none", "strong_right", "none")):
        control = lv == "none"
        rows.append({"dyad_id": f"d{i}", "condition": {"topic": "immigration_enforcement", "ideology": lv,
                                                       "openness": None if control else "open",
                                                       "role": None if control else f"r{i % 2}"},
                     "persona_text": f"You are a voter. IDEO={lv}.",
                     "persona_reminder": f"Remember IDEO={lv}.",
                     "persona_mode": "reinforced", "seed": 100 + i, "n_turns": 3})
    man = tmp_path / "dyads.jsonl"
    man.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert R.main(["run", "--config", str(cfg_path), "--manifest", str(man), "--run-id", "r1"]) == 0
    assert R.main(["score", "--config", str(cfg_path), "--run-id", "r1", "--scope", "main"]) == 0
    return tmp_path / "data" / "r1"


def test_every_harness_dialogue_and_survey_rebuilds(harness_run, capsys):
    assert V.main(["--run-dir", str(harness_run), "--all"]) == 0
    assert "6/6 dyads verified" in capsys.readouterr().out
    assert V.main(["--run-dir", str(harness_run), "--dyad-id", "d2"]) == 0
    out = capsys.readouterr().out
    assert "ok d2 attempt 1" in out
    n_items = len(json.loads((harness_run / "manifest.json").read_text())["batteries"]["item_ids"])
    assert f"{3 * 2 + 2 * n_items} rows rebuilt" in out              # 6 messages, pre and post items


def test_the_analysis_reads_what_the_harness_wrote(harness_run):
    run = load_run(harness_run)
    assert len([d for d in run.dyads if d["in_itt"]]) == 6
    assert run.judge is not None and all(d.get("adherence_mean") == pytest.approx(0.8)
                                         for d in run.dyads if not d["control"])
    inst = load_instrument(run)
    rows, counts = compute_outcomes(run, inst)
    by = {r["dyad_id"]: r for r in rows}
    assert by["d2"]["ideological_change"] == pytest.approx(1.0)         # strong_right: one point rightward
    assert by["d0"]["ideological_change"] == pytest.approx(-1.0)
    assert by["d1"]["ideological_change"] == pytest.approx(0.0) and by["d3"]["ideological_change"] == 0.0
    assert counts["null"] == 0 and counts["salvaged"] == 0
    res = estimate([run], rows, [], ["ideological"])
    assert res["outcomes"]["ideological"]["level"]["immigration_enforcement"]["n_control"] == 2


def test_a_tampered_row_is_the_first_mismatch(tmp_path, capsys):
    res = make_run(tmp_path / "s", SynthSpec(run_id="s", n_per_role=1, n_control=1, n_turns=3, hashes=True,
                                                   levels=("lean_left", "strong_right"),
                                                   topics=("immigration_enforcement",)))
    assert V.main(["--run-dir", str(res["path"]), "--all"]) == 0
    p = res["path"] / "turns.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    dyad = rows[0]["dyad_id"]
    i = next(k for k, r in enumerate(rows)
             if r["dyad_id"] == dyad and r["turn"] == 2 and r["agent"] == "seeker")
    rows[i]["text"] += " (edited)"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    capsys.readouterr()
    assert V.main(["--run-dir", str(res["path"]), "--dyad-id", dyad]) == 1
    out = capsys.readouterr().out
    # the edited line is in every later prompt: the first to break is the mentor's reply to it
    assert "MISMATCH" in out and "prompt_sha256 at" in out and "turn=2 agent=mentor" in out


def test_seed_template_and_instrument_mismatches(tmp_path, capsys):
    res = make_run(tmp_path / "s", SynthSpec(run_id="s", n_per_role=1, n_control=1, n_turns=2, hashes=True,
                                                   levels=("moderate",), topics=("decarbonization",)))
    ctx = V._Ctx(res["path"])
    dyad = next(iter(ctx.dyads))[0]
    items, why = V._instrument_items(res["path"], None)
    assert why is None and V.verify(ctx, dyad, None, items)["ok"]
    ctx.turns[0]["seed"] += 1
    r = V.verify(ctx, ctx.turns[0]["dyad_id"], None, items)
    assert r["first_mismatch"]["kind"] == "seed"
    mf = json.loads((res["path"] / "manifest.json").read_text())
    mf["mentor"]["template_source"] += " "
    (res["path"] / "manifest.json").write_text(json.dumps(mf))
    r = V.verify(V._Ctx(res["path"]), dyad, None, items)
    assert r["first_mismatch"]["kind"] == "template_sha256"
    # an instrument that no longer hashes to the record: dialogue rows are still checked, surveys are not
    (res["path"] / "instrument.json").write_text("{}")
    items, why = V._instrument_items(res["path"], None)
    assert items is None and "hashing to" in why
    assert V.main(["--run-dir", str(res["path"]), "--dyad-id", "nope"]) == 1
    assert "no_status" in capsys.readouterr().out


def test_a_template_that_prints_bos_is_warned_about(tmp_path):
    res = make_run(tmp_path / "s", SynthSpec(run_id="s", n_per_role=1, n_control=1, n_turns=1))
    mf = json.loads((res["path"] / "manifest.json").read_text())
    mf["seeker"]["template_source"] = "{{ bos_token }}" + mf["seeker"]["template_source"]
    (res["path"] / "manifest.json").write_text(json.dumps(mf))
    assert any("bos_token" in w for w in V._Ctx(res["path"]).warnings)
