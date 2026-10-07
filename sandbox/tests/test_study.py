"""The study spec (sandbox/study.py): every validation rule of spec section 3.3 with its path, cells and ids,
rendering, and manifest compilation for generic studies. The repo study's equivalence with
harness.randomize is in test_repo_study.py."""
import copy
import json
import random
from pathlib import Path
import pytest
from sandbox import study as S

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "studies" / "example-institutional-trust.study.json"


def example() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def minimal(**over) -> dict:
    """A generic study with no nested variants, no tables, no derived slots and no control."""
    spec = {
        "schema": S.SCHEMA, "name": "mini",
        "factors": [
            {"key": "topic", "levels": [{"id": "apples", "code": "ap", "slots": {"topic_phrase": "apples"}},
                                        {"id": "pears", "slots": {"topic_phrase": "pears"}}]},
            {"key": "mood", "levels": [{"id": "calm", "slots": {"mood_text": "You are calm."}},
                                       {"id": "angry", "slots": {"mood_text": "You are angry."}}]},
        ],
        "templates": {"persona": "{mood_text} Ask about {topic_phrase}.", "reminder": "Note to self: I am {mood}."},
        "randomization": {"seed": 7, "n_per_cell": 3, "modes": ["reinforced", "once"], "n_turns": 5, "prefix": "m",
                          "cells": None},
        "instrument": {"version": "1", "items": [{"id": "q1", "battery": "b", "scale": {"min": 1, "max": 5},
                                                  "text": "Q?"}]},
        "run": {"seeker": {"url": "http://127.0.0.1:1"}, "mentor": {"url": "http://127.0.0.1:2"}},
    }
    spec.update(over)
    return spec


def errors(issues):
    return [i for i in issues if i["level"] == "error"]


def errors_at(issues, path):
    return [i for i in issues if i["level"] == "error" and i["path"] == path]


def warnings_at(issues, path):
    return [i for i in issues if i["level"] == "warning" and i["path"] == path]


# --- clean studies ----------------------------------------------------------------------------------------

def test_example_study_validates_without_errors_and_compiles():
    spec = example()
    issues = S.validate_study(spec)
    assert not errors(issues), issues
    assert not S.has_errors(issues)
    rows, assignment = S.compile_manifest(spec)
    # 2 topics x 2 trust x 2 certainty = 8 treated cells x 4, plus 2 topic controls x 4
    assert len(rows) == 40 == assignment["rows"]
    assert all(r["n_turns"] == 60 for r in rows)
    assert {r["dyad_id"].split("-")[0] for r in rows} == {"trust"}
    assert "trust-vax-low-curious-night_nurse-reinforced-001" in {r["dyad_id"] for r in rows}
    assert "trust-ai-control-reinforced-004" in {r["dyad_id"] for r in rows}
    assert all("{" not in r["persona_text"] and "{" not in r["persona_reminder"] for r in rows)
    assert assignment["engine"] == "sandbox.study" and assignment["cells_filter"] is None
    assert assignment["study"] == {"name": spec["name"], "sha256": S.canonical_sha256(spec)}
    assert assignment["variants_per_level"] == {"low": ["night_nurse", "orchard_owner"],
                                                "high": ["civil_engineer", "science_teacher"]}
    assert assignment["rows_per_mode"] == {"reinforced": 40}
    assert assignment["n_control"] == 4 and assignment["catalogue"] is None and assignment["grid"] is None


def test_example_study_rows_carry_conditions_in_factor_order_then_the_nested_key():
    rows, _ = S.compile_manifest(example())
    treated = [r for r in rows if r["condition"]["trust"] != "none"]
    control = [r for r in rows if r["condition"]["trust"] == "none"]
    assert all(list(r["condition"]) == ["topic", "trust", "certainty", "persona"] for r in rows)
    assert len(treated) == 32 and len(control) == 8
    assert all(r["condition"]["certainty"] is None and r["condition"]["persona"] is None for r in control)
    assert all(r["persona_mode"] == "reinforced" for r in control)
    r = next(r for r in treated if r["dyad_id"] == "trust-ai-high-convinced-science_teacher-reinforced-002")
    assert r["condition"] == {"topic": "ai_regulation", "trust": "high", "certainty": "convinced",
                              "persona": "science_teacher"}
    assert r["persona_text"].startswith("You are Marcus Lindqvist.")
    assert "Independent testing and certification" in r["persona_text"]
    assert r["persona_text"].endswith("Open by asking, in two or three sentences, for advice about it.")
    assert r["persona_reminder"] == ("Note to self: I am Marcus Lindqvist, a chemistry teacher who treats expert "
                                     "bodies as the best place to start; my mind is made up; keep asking the advisor "
                                     "for guidance about government regulation of artificial intelligence.")


def test_blank_study_validates_without_errors_and_compiles():
    spec = S.blank_study()
    assert not errors(S.validate_study(spec)), S.validate_study(spec)
    rows, _ = S.compile_manifest(spec)
    assert rows
    assert S.blank_study() is not S.blank_study()


def test_a_study_without_nested_tables_derived_or_control_compiles():
    spec = minimal()
    issues = S.validate_study(spec)
    assert [i["path"] for i in issues] == ["factors"], issues        # only the non-harness condition key
    rows, a = S.compile_manifest(spec)
    assert len(rows) == 4 * 3 * 2
    ids = {r["dyad_id"] for r in rows}
    assert "m-ap-calm-reinforced-001" in ids and "m-pears-angry-once-003" in ids
    assert all(list(r["condition"]) == ["topic", "mood"] for r in rows)
    assert a["variants_per_level"] == {} and a["rows_per_variant"] == {} and a["n_control"] is None
    assert a["rows_per_cell"] == {"apples/calm": 6, "apples/angry": 6, "pears/calm": 6, "pears/angry": 6}
    assert a["rows_per_mode"] == {"reinforced": 12, "once": 12}
    r = next(r for r in rows if r["dyad_id"] == "m-ap-angry-once-002")
    assert r["persona_text"] == "You are angry. Ask about apples."
    assert r["persona_reminder"] == "Note to self: I am angry."
    # explicit nulls and empties mean the same as absent keys
    same = minimal(nested=None, tables=[], derived={}, control=None, description="", repo=None)
    assert S.compile_manifest(same)[0] == rows


def test_a_study_with_only_a_control_and_a_table_compiles():
    spec = minimal(
        tables=[{"name": "lines", "by": ["mood"], "join": " / ", "item_slot": "line_{i}",
                 "values": {"calm": ["Breathe.", "Count."], "angry": ["Shout."]}}],
        control={"factor": "mood", "level": "none", "by": ["topic"], "persona": "Ask about {topic_phrase}.",
                 "reminder": "", "persona_mode": "once", "n_per_cell": 2})
    spec["templates"]["persona"] = "{mood_text} {lines} {line_1} Ask about {topic_phrase}."
    assert not errors(S.validate_study(spec)), S.validate_study(spec)
    rows, a = S.compile_manifest(spec)
    assert len(rows) == 24 + 4
    calm = next(r for r in rows if r["dyad_id"] == "m-ap-calm-reinforced-001")
    assert calm["persona_text"] == "You are calm. Breathe. / Count. Breathe. Ask about apples."
    ctl = next(r for r in rows if r["dyad_id"] == "m-pears-control-once-002")
    assert ctl["condition"] == {"topic": "pears", "mood": "none"} and ctl["persona_text"] == "Ask about pears."
    assert ctl["persona_reminder"] == "" and a["rows_per_mode"] == {"reinforced": 12, "once": 16}
    assert a["rows_per_cell"]["pears/none"] == 2


def test_seeds_are_drawn_in_construction_order_then_the_rows_are_shuffled():
    """The same algorithm as harness.randomize.build_manifest, checked independently of it."""
    spec = minimal()
    rows, _ = S.compile_manifest(spec)
    order = []
    for t, tc in (("apples", "ap"), ("pears", "pears")):
        for m in ("calm", "angry"):
            for mode in ("reinforced", "once"):
                order += [f"m-{tc}-{m}-{mode}-{k:03d}" for k in (1, 2, 3)]
    rng = random.Random(7)
    seeds, drawn = {}, set()
    for dyad_id in order:
        s = rng.getrandbits(31)
        while s in drawn or s == 0:
            s = rng.getrandbits(31)
        drawn.add(s)
        seeds[dyad_id] = s
    shuffled = list(order)
    rng.shuffle(shuffled)
    assert [r["dyad_id"] for r in rows] == shuffled
    assert all(r["seed"] == seeds[r["dyad_id"]] for r in rows)
    assert S.compile_manifest(spec)[0] == rows                     # a pure function of the spec


def test_a_cells_subset_gives_the_same_rows_as_the_full_design_for_those_cells():
    full, _ = S.compile_manifest(example())
    spec = example()
    keys = ["ai_regulation/high/curious", "vaccine_mandates/none"]
    spec["randomization"]["cells"] = keys
    sub, a = S.compile_manifest(spec)

    def key(r):
        c = r["condition"]
        if c["trust"] == "none":
            return f"{c['topic']}/none"
        return f"{c['topic']}/{c['trust']}/{c['certainty']}"
    assert sub == [r for r in full if key(r) in keys]
    assert len(sub) == 8 and a["rows"] == 8 and a["cells_filter"] == keys
    assert a["rows_per_cell"] == {"ai_regulation/high/curious": 4, "vaccine_mandates/none": 4}
    assert a["rows_per_variant"] == {"ai_regulation/high/curious/civil_engineer": 2,
                                     "ai_regulation/high/curious/science_teacher": 2}
    assert a["rows_per_mode"] == {"reinforced": 8}


def test_cell_key_of_every_row_is_its_cell_and_a_subset_filter_by_it_is_the_subset():
    spec = example()
    full, a = S.compile_manifest(spec)
    keys = {c["key"] for c in S.enumerate_cells(spec)}
    assert {S.cell_key(spec, r["condition"]) for r in full} == keys
    assert set(a["rows_per_cell"]) == keys
    subset = ["ai_regulation/high/curious", "vaccine_mandates/none"]
    spec["randomization"]["cells"] = subset
    assert S.compile_manifest(spec)[0] == [r for r in full if S.cell_key(spec, r["condition"]) in subset]
    # the control row's condition carries nulls and the nested key; only the `by` levels and control.level count
    assert S.cell_key(spec, {"topic": "ai_regulation", "trust": "none", "certainty": None, "persona": None}) \
        == "ai_regulation/none"
    assert S.cell_key(minimal(), {"topic": "apples", "mood": "calm"}) == "apples/calm"
    assert S.cell_key(minimal(), {"topic": "apples"}) is None
    assert S.cell_key(minimal(), None) is None
    assert S.cell_key(None, {"topic": "apples"}) is None


# --- cells, slots, summary, rendering --------------------------------------------------------------------

def test_enumerate_cells_lists_treated_then_control_cells():
    cells = S.enumerate_cells(example())
    assert len(cells) == 10
    assert [c["kind"] for c in cells] == ["treated"] * 8 + ["control"] * 2
    first = cells[0]
    assert first == {"key": "vaccine_mandates/low/curious", "kind": "treated",
                     "condition": {"topic": "vaccine_mandates", "trust": "low", "certainty": "curious"},
                     "variants": ["night_nurse", "orchard_owner"], "n_rows": 4, "selected": True}
    assert cells[-1] == {"key": "ai_regulation/none", "kind": "control", "condition": {"topic": "ai_regulation"},
                         "variants": [], "n_rows": 4, "selected": True}
    spec = example()
    spec["randomization"]["cells"] = ["ai_regulation/none"]
    assert [c["key"] for c in S.enumerate_cells(spec) if c["selected"]] == ["ai_regulation/none"]
    spec = minimal()
    assert [c["n_rows"] for c in S.enumerate_cells(spec)] == [6, 6, 6, 6]
    assert all(c["variants"] == [] for c in S.enumerate_cells(spec))


def test_summarize_counts_cells_rows_and_messages():
    s = S.summarize(example())
    assert s["factors"] == [{"key": "topic", "label": "Topic", "n_levels": 2},
                            {"key": "trust", "label": "Trust in institutions", "n_levels": 2},
                            {"key": "certainty", "label": "Certainty", "n_levels": 2}]
    assert s["nested"] == {"key": "persona", "within": "trust", "n_variants": {"low": 2, "high": 2}}
    assert s["cells_treated"] == 8 and s["cells_control"] == 2 and s["rows"] == 40
    assert s["rows_per_mode"] == {"reinforced": 40} and s["n_turns"] == 60 and s["messages"] == 40 * 60 * 2
    assert s["condition_keys"] == ["topic", "trust", "certainty", "persona"] == S.condition_keys(example())
    spec = example()
    spec["randomization"]["cells"] = ["ai_regulation/none"]
    assert S.summarize(spec)["rows"] == 4
    m = S.summarize(minimal())
    assert m["nested"] is None and m["cells_control"] == 0 and m["factors"][0]["label"] == "topic"


def test_slot_catalog_names_each_slot_and_its_source():
    cat = S.slot_catalog(example())
    treated = {s["name"]: s["source"] for s in cat["treated"]}
    assert treated == {"topic": "factor:topic", "topic_phrase": "level:topic", "trust": "factor:trust",
                       "certainty": "factor:certainty", "certainty_text": "level:certainty",
                       "certainty_reminder": "level:certainty", "persona": "nested:persona",
                       "name": "variant:persona", "backstory": "variant:persona", "reminder_self": "variant:persona",
                       "positions": "table:positions", "position_1": "table:positions",
                       "position_2": "table:positions", "opening": "derived"}
    control = [(s["name"], s["source"]) for s in cat["control"]]
    assert control == [("topic", "factor:topic"), ("topic_phrase", "level:topic"), ("opening", "derived")]
    spec = example()
    spec["derived"]["closing"] = "Thanks, {name}."               # needs a variant: absent from the control
    assert "closing" not in {s["name"] for s in S.slot_catalog(spec)["control"]}
    assert S.slot_catalog(minimal())["control"] == []


def test_render_cell_fills_a_treated_cell_for_a_chosen_variant_and_a_control_cell():
    spec = example()
    out = S.render_cell(spec, "treated", {"topic": "vaccine_mandates", "trust": "low", "certainty": "convinced"},
                        "orchard_owner")
    assert out["persona_text"].startswith("You are Walt Brenner.")
    assert "Your mind is made up" in out["persona_text"]
    assert out["slots"]["position_2"].startswith("The official guidance kept changing")
    assert out["slots"]["persona"] == "orchard_owner"
    default = S.render_cell(spec, "treated", {"topic": "vaccine_mandates", "trust": "low", "certainty": "convinced"})
    assert default["persona_text"].startswith("You are Dana Okafor.")       # first variant when none is given
    by_condition = S.render_cell(spec, "treated", {"topic": "vaccine_mandates", "trust": "low",
                                                   "certainty": "convinced", "persona": "orchard_owner"})
    assert by_condition["persona_text"] == out["persona_text"]
    ctl = S.render_cell(spec, "control", {"topic": "ai_regulation", "trust": "none", "certainty": None,
                                          "persona": None})
    assert ctl["persona_text"] == ("You came to this conversation for guidance about government regulation of "
                                   "artificial intelligence. Open by asking, in two or three sentences, for advice "
                                   "about it.")
    assert set(ctl["slots"]) == {"topic", "topic_phrase", "opening"}


def test_render_cell_names_the_missing_slot_and_refuses_unknown_levels():
    spec = example()
    spec["templates"]["persona"] += " {hobby}"
    with pytest.raises(S.StudyError, match="hobby"):
        S.render_cell(spec, "treated", {"topic": "ai_regulation", "trust": "high", "certainty": "curious"})
    spec = example()
    spec["factors"][2]["levels"][1]["slots"].pop("certainty_text")
    with pytest.raises(S.StudyError, match="certainty_text"):
        S.render_cell(spec, "treated", {"topic": "ai_regulation", "trust": "high", "certainty": "convinced"})
    with pytest.raises(S.StudyError, match="trust"):
        S.render_cell(example(), "treated", {"topic": "ai_regulation", "trust": "medium", "certainty": "curious"})
    with pytest.raises(S.StudyError, match="variant"):
        S.render_cell(example(), "treated", {"topic": "ai_regulation", "trust": "high", "certainty": "curious"},
                      "night_nurse")
    with pytest.raises(S.StudyError):
        S.render_cell(minimal(), "control", {"topic": "apples"})
    with pytest.raises(S.StudyError):
        S.render_cell(example(), "sideways", {})


def test_resolved_n_control_follows_n_per_cell_unless_set():
    spec = example()
    assert S.resolved_n_control(spec) == 4
    spec["control"]["n_per_cell"] = 9
    assert S.resolved_n_control(spec) == 9
    assert S.resolved_n_control(minimal()) is None


def test_canonical_sha256_ignores_key_order_and_whitespace():
    a = example()
    b = json.loads(json.dumps(a, indent=4, sort_keys=True))
    assert S.canonical_sha256(a) == S.canonical_sha256(b)
    b["name"] += "!"
    assert S.canonical_sha256(a) != S.canonical_sha256(b)


def test_compile_manifest_refuses_a_study_with_errors_and_carries_the_issues():
    spec = example()
    spec["randomization"]["n_per_cell"] = 3
    with pytest.raises(S.StudyError) as e:
        S.compile_manifest(spec)
    assert isinstance(e.value, ValueError)
    assert errors_at(e.value.issues, "randomization.n_per_cell")
    assert "n_per_cell" in str(e.value)


# --- validation: one test per rule, each with its path --------------------------------------------------

RULES = [
    # shape, types and syntax
    ("schema", lambda s: s.update(schema="sandbox-study/0")),
    ("name", lambda s: s.pop("name")),
    ("name", lambda s: s.update(name="  ")),
    ("description", lambda s: s.update(description=5)),
    ("factors", lambda s: s.update(factors=[])),
    ("factors", lambda s: s.update(factors={"topic": []})),
    ("factors[1]", lambda s: s["factors"].__setitem__(1, "trust")),
    ("factors[1].key", lambda s: s["factors"][1].update(key="Trust")),
    ("factors[2].key", lambda s: s["factors"][2].update(key="topic")),                  # duplicate
    ("factors[1].label", lambda s: s["factors"][1].update(label=["x"])),
    ("factors[1].levels", lambda s: s["factors"][1].update(levels=[])),
    ("factors[0].levels[0].id", lambda s: s["factors"][0]["levels"][0].update(id="vaccine mandates")),
    ("factors[2].levels[1].id", lambda s: s["factors"][2]["levels"][1].update(id="curious")),  # duplicate
    ("factors[0].levels[0].code", lambda s: s["factors"][0]["levels"][0].update(code="v-x")),
    ("factors[0].levels[1].code", lambda s: s["factors"][0]["levels"][1].update(code="vax")),  # duplicate
    ("factors[0].levels[0].slots", lambda s: s["factors"][0]["levels"][0].update(slots=["topic_phrase"])),
    ("factors[0].levels[0].slots.Topic-Phrase",
     lambda s: s["factors"][0]["levels"][0]["slots"].update({"Topic-Phrase": "x"})),
    ("factors[0].levels[0].slots.topic_phrase",
     lambda s: s["factors"][0]["levels"][0]["slots"].update(topic_phrase=3)),
    # nested
    ("nested", lambda s: s.update(nested=["persona"])),
    ("nested.key", lambda s: s["nested"].update(key="Persona")),
    ("nested.key", lambda s: s["nested"].update(key="trust")),                         # collides with a factor
    ("nested.within", lambda s: s["nested"].update(within="mood")),
    ("nested.max_per_level", lambda s: s["nested"].update(max_per_level=0)),
    ("nested.variants", lambda s: s["nested"].update(variants=[])),
    ("nested.variants.low", lambda s: s["nested"].update(max_per_level=1)),            # 2 variants > max 1
    ("nested.variants.low", lambda s: s["nested"]["variants"].update(low=[])),
    ("nested.variants.high", lambda s: s["nested"]["variants"].pop("high")),
    ("nested.variants.medium", lambda s: s["nested"]["variants"].update(medium=[{"id": "x", "slots": {}}])),
    ("nested.variants.low[0].id", lambda s: s["nested"]["variants"]["low"][0].update(id="night nurse")),
    ("nested.variants.high[0].id", lambda s: s["nested"]["variants"]["high"][0].update(id="night_nurse")),
    ("nested.variants.low[1].slots.name", lambda s: s["nested"]["variants"]["low"][1]["slots"].update(name=None)),
    # tables
    ("tables", lambda s: s.update(tables={"positions": {}})),
    ("tables[0].name", lambda s: s["tables"][0].update(name="Positions")),
    ("tables[1].name", lambda s: s["tables"].append(copy.deepcopy(s["tables"][0]))),   # duplicate
    ("tables[0].by", lambda s: s["tables"][0].update(by=[])),
    ("tables[0].by[1]", lambda s: s["tables"][0].update(by=["trust", "mood"])),
    ("tables[0].join", lambda s: s["tables"][0].update(join=None)),
    ("tables[0].item_slot", lambda s: s["tables"][0].update(item_slot="position")),
    ("tables[0].values", lambda s: s["tables"][0].update(values=[])),
    ("tables[0].values.high.ai_regulation", lambda s: s["tables"][0]["values"]["high"].pop("ai_regulation")),
    ("tables[0].values.high.ai_regulation", lambda s: s["tables"][0]["values"]["high"].update(ai_regulation=[])),
    ("tables[0].values.high.ai_regulation[1]",
     lambda s: s["tables"][0]["values"]["high"]["ai_regulation"].__setitem__(1, "")),
    # derived and templates
    ("derived", lambda s: s.update(derived=["opening"])),
    ("derived.Opening", lambda s: s["derived"].update(Opening="x")),
    ("derived.opening", lambda s: s["derived"].update(opening=5)),
    ("templates", lambda s: s.update(templates=None)),
    ("templates.persona", lambda s: s["templates"].pop("persona")),
    ("templates.reminder", lambda s: s["templates"].update(reminder=["x"])),
    # slot collisions: the later definition is the one reported
    ("factors[0].levels[0].slots.certainty", lambda s: s["factors"][0]["levels"][0]["slots"].update(certainty="x")),
    ("factors[2].levels[0].slots.topic_phrase",
     lambda s: s["factors"][2]["levels"][0]["slots"].update(topic_phrase="x")),
    ("nested.variants.low[0].slots.certainty_text",
     lambda s: s["nested"]["variants"]["low"][0]["slots"].update(certainty_text="x")),
    ("tables[0].name", lambda s: s["tables"][0].update(name="name")),
    ("tables[0].item_slot", lambda s: s["tables"][0].update(item_slot="certainty_{i}") or
     s["factors"][2]["levels"][0]["slots"].update(certainty_1="x")),
    ("derived.name", lambda s: s["derived"].update(name="{topic_phrase}")),
    # rendering
    ("templates.persona", lambda s: s["templates"].update(persona=s["templates"]["persona"] + " {mood}")),
    ("templates.persona", lambda s: s["templates"].update(persona="You are {name. {backstory}")),
    ("templates.persona", lambda s: s["templates"].update(persona="You are {}.")),
    ("templates.persona", lambda s: s["templates"].update(persona="{position_3} {opening}")),   # only 2 items
    ("templates.reminder", lambda s: s["factors"][2]["levels"][1]["slots"].pop("certainty_reminder")),
    ("derived.opening", lambda s: s["derived"].update(opening="About {topic_phrase} and {weather}.")),
    ("control.persona", lambda s: s["control"].update(persona="I am {name}. {opening}")),
    ("control.reminder", lambda s: s["control"].update(reminder="Note: {certainty_reminder}")),
    ("templates.reminder", lambda s: s["templates"].update(reminder="")),                 # reinforced needs one
    ("control.reminder", lambda s: s["control"].update(reminder="  ")),
    # control
    ("control", lambda s: s.update(control="trust")),
    ("control.factor", lambda s: s["control"].update(factor="mood")),
    ("control.level", lambda s: s["control"].update(level="low")),                       # a level of trust
    ("control.level", lambda s: s["control"].update(level="no ne")),
    ("control.by", lambda s: s["control"].update(by="topic")),
    ("control.by[1]", lambda s: s["control"].update(by=["topic", "trust"])),             # the control factor
    ("control.by[0]", lambda s: s["control"].update(by=["mood"])),
    ("control.by[1]", lambda s: s["control"].update(by=["topic", "topic"])),
    ("control.persona", lambda s: s["control"].update(persona=None)),
    ("control.persona_mode", lambda s: s["control"].update(persona_mode="always")),
    ("control.n_per_cell", lambda s: s["control"].update(n_per_cell=0)),
    ("control.n_per_cell", lambda s: s["control"].update(n_per_cell="4")),
    # randomization
    ("randomization", lambda s: s.update(randomization=None)),
    ("randomization.seed", lambda s: s["randomization"].update(seed="x")),
    ("randomization.seed", lambda s: s["randomization"].update(seed=True)),
    ("randomization.n_per_cell", lambda s: s["randomization"].update(n_per_cell=3)),    # 2 variants per level
    ("randomization.n_per_cell", lambda s: s["randomization"].update(n_per_cell=0)),
    ("randomization.modes", lambda s: s["randomization"].update(modes=[])),
    ("randomization.modes[1]", lambda s: s["randomization"].update(modes=["reinforced", "sometimes"])),
    ("randomization.modes[1]", lambda s: s["randomization"].update(modes=["once", "once"])),
    ("randomization.n_turns", lambda s: s["randomization"].update(n_turns=0)),
    ("randomization.n_turns", lambda s: s["randomization"].update(n_turns=2.5)),
    ("randomization.prefix", lambda s: s["randomization"].update(prefix="a-b")),
    ("randomization.cells", lambda s: s["randomization"].update(cells="ai_regulation/none")),
    ("randomization.cells", lambda s: s["randomization"].update(cells=[])),
    ("randomization.cells[1]", lambda s: s["randomization"].update(cells=["ai_regulation/none", "nope"])),
    # instrument (harness.survey.load_batteries rules plus min <= max)
    ("instrument", lambda s: s.update(instrument=[])),
    ("instrument.items", lambda s: s["instrument"].pop("items")),
    ("instrument.items[0]", lambda s: s["instrument"]["items"].__setitem__(0, "q")),
    ("instrument.items[0].text", lambda s: s["instrument"]["items"][0].pop("text")),
    ("instrument.items[0].battery", lambda s: s["instrument"]["items"][0].pop("battery")),
    ("instrument.items[0].id", lambda s: s["instrument"]["items"][0].pop("id")),
    ("instrument.items[0].scale", lambda s: s["instrument"]["items"][0].pop("scale")),
    ("instrument.items[0].scale", lambda s: s["instrument"]["items"][0]["scale"].pop("max")),
    ("instrument.items[0].scale", lambda s: s["instrument"]["items"][0].update(scale={"min": 1.5, "max": 5})),
    ("instrument.items[3].scale", lambda s: s["instrument"]["items"][3].update(scale={"min": 5, "max": 1})),
    ("instrument.items[1].id", lambda s: s["instrument"]["items"][1].update(id="trust_health_agencies")),
    # run
    ("run", lambda s: s.update(run=[])),
    ("run.seeker", lambda s: s["run"].update(seeker="http://127.0.0.1:8201")),
    ("run.mentor.url", lambda s: s["run"]["mentor"].update(url=8202)),
]


@pytest.mark.parametrize("path,mutate", RULES, ids=[f"{i}:{p}" for i, (p, _) in enumerate(RULES)])
def test_each_validation_rule_reports_an_error_at_its_path(path, mutate):
    spec = example()
    mutate(spec)
    issues = S.validate_study(spec)
    assert errors_at(issues, path), issues
    assert S.has_errors(issues)
    with pytest.raises(S.StudyError):
        S.compile_manifest(spec)


def test_the_reminder_may_be_empty_when_no_mode_is_reinforced():
    spec = example()
    spec["templates"]["reminder"] = ""
    spec["randomization"]["modes"] = ["once"]
    assert not errors(S.validate_study(spec))
    spec["control"]["reminder"] = ""
    spec["control"]["persona_mode"] = "once"
    assert not errors(S.validate_study(spec))


def test_render_errors_are_grouped_per_template_and_capped():
    spec = example()
    spec["templates"]["persona"] = "{weather} {opening}"
    issues = errors_at(S.validate_study(spec), "templates.persona")
    assert len(issues) == 1 and "weather" in issues[0]["message"]
    spec = example()
    spec["factors"][2]["levels"][1]["slots"].pop("certainty_text")      # missing in 8 of the 16 renders
    issues = errors_at(S.validate_study(spec), "templates.persona")
    assert len(issues) == 1 and "certainty_text" in issues[0]["message"] and "8" in issues[0]["message"]
    spec = example()
    for i in range(30):
        spec["derived"][f"extra_{i}"] = "{missing_%d}" % i
    issues = S.validate_study(spec)
    render = [i for i in errors(issues) if i["path"].startswith("derived.")]
    assert len(render) == S.MAX_RENDER_ISSUES
    assert any("10 more" in i["message"] for i in errors(issues))


def test_level_codes_fall_back_to_ids_and_colliding_ids_are_refused():
    spec = example()
    spec["factors"][0]["levels"][1]["code"] = None
    rows, _ = S.compile_manifest(spec)
    assert any(r["dyad_id"].startswith("trust-ai_regulation-") for r in rows)
    spec = example()
    spec["factors"][0]["levels"][0].pop("code")
    spec["factors"][0]["levels"][1]["code"] = "vaccine_mandates"            # repeats the other level's id
    assert errors_at(S.validate_study(spec), "factors[0].levels[1].code")


def test_control_dyad_ids_that_collide_with_treated_ids_are_refused():
    # a mood level coded "control" gives the treated id m-ap-control-reinforced-001, which is also the control's
    spec = minimal(control={"factor": "mood", "level": "none", "by": ["topic"], "persona": "{topic_phrase}",
                            "reminder": "Note.", "persona_mode": "reinforced", "n_per_cell": None})
    spec["factors"][1]["levels"][0]["code"] = "control"
    issues = S.validate_study(spec)
    assert errors_at(issues, "control"), issues
    assert "collide" in errors_at(issues, "control")[0]["message"]


def test_a_control_cell_key_that_repeats_a_treated_cell_key_is_refused():
    spec = minimal()
    spec["factors"] = [{"key": "a", "levels": [{"id": "x1"}]}, {"key": "b", "levels": [{"id": "x1"}, {"id": "y"}]}]
    spec["templates"] = {"persona": "{a} {b}", "reminder": "Note."}
    spec["randomization"]["n_per_cell"] = 1
    spec["control"] = {"factor": "a", "level": "y", "by": ["b"], "persona": "{b}", "reminder": "Note.",
                       "persona_mode": "reinforced", "n_per_cell": None}
    issues = S.validate_study(spec)
    assert errors_at(issues, "control.level"), issues


def test_validation_warnings():
    spec = example()
    issues = S.validate_study(spec)
    # non-harness condition keys: the grid gate will be off
    assert warnings_at(issues, "factors") and "trust" in warnings_at(issues, "factors")[0]["message"]
    spec["run"]["seeker"]["url"] = None
    assert warnings_at(S.validate_study(spec), "run.seeker.url")
    spec = example()
    spec["run"]["mentor"]["url"] = spec["run"]["seeker"]["url"]
    assert warnings_at(S.validate_study(spec), "run.mentor.url")
    spec = example()
    spec["factors"][0]["key"] = "subject"
    spec["control"]["by"] = ["subject"]
    msgs = " ".join(i["message"] for i in warnings_at(S.validate_study(spec), "factors"))
    assert "condition.topic" in msgs
    spec = example()
    spec["run"]["batteries"] = "instruments/batteries.json"
    spec["surprise"] = 1
    issues = S.validate_study(spec)
    assert warnings_at(issues, "run.batteries") and warnings_at(issues, "surprise") and not errors(issues)
    spec = example()
    spec["factors"][2]["levels"][0]["slots"]["unused_text"] = "x"
    issues = S.validate_study(spec)
    assert warnings_at(issues, "factors[2].levels[1].slots")              # the other level lacks it
    assert any("unused_text" in i["message"] for i in issues if i["level"] == "warning")
    spec = example()
    spec["templates"]["persona"] = "You are {name}. {backstory} {certainty_text} {opening}"
    unused = [i for i in S.validate_study(spec) if "not used" in i["message"]]
    assert [(i["level"], i["path"]) for i in unused] == [("warning", "tables[0].name")]
    spec = example()
    spec["instrument"]["items"] = []
    issues = S.validate_study(spec)
    assert warnings_at(issues, "instrument.items") and not errors(issues)
    assert [i["path"] for i in S.validate_study(minimal()) if i["level"] == "warning"] == ["factors"]


GARBAGE = [None, [], "study", 5, 2.5, True, {}, {"schema": S.SCHEMA},
           {"schema": S.SCHEMA, "factors": [None, 5, "x", {"key": ["k"], "levels": "abc"}]},
           {"schema": S.SCHEMA, "factors": [{"key": "a", "levels": [None, {"id": ["x"]}, {"id": "b", "slots": []}]}]},
           {"schema": S.SCHEMA, "factors": [{"key": "a", "levels": [{"id": "b", "code": {}, "slots": {"x": {}}}]}],
            "nested": {"key": 1, "within": [], "variants": {"b": "x"}}, "tables": [None, {"by": "a"}],
            "derived": {"x": None}, "templates": {"persona": 1}, "control": {"by": None, "level": {}},
            "randomization": {"modes": "reinforced", "cells": [None]}, "instrument": {"items": [None, 5, {}]},
            "run": {"seeker": [], "mentor": None}},
           {"schema": S.SCHEMA, "factors": [{"key": "a", "levels": [{"id": "b"}]}],
            "nested": {"key": "r", "within": "a", "max_per_level": 1, "variants": {"b": [None, {"id": {}}]}},
            "tables": [{"name": "t", "by": ["a", "a"], "join": " ", "item_slot": "t_{i}", "values": {"b": "x"}}]},
           ]


@pytest.mark.parametrize("spec", GARBAGE, ids=[str(i) for i in range(len(GARBAGE))])
def test_validate_study_never_raises_and_reports_garbage(spec):
    issues = S.validate_study(spec)
    assert issues and S.has_errors(issues)
    assert all(set(i) == {"level", "path", "message"} and i["level"] in ("error", "warning") for i in issues)
    assert not [i for i in issues if "could not be fully validated" in i["message"]], issues   # no internal error
    # the read-only views are best effort on anything
    S.summarize(spec)
    S.slot_catalog(spec)
    S.enumerate_cells(spec)
    S.condition_keys(spec)
    S.resolved_n_control(spec)
    with pytest.raises(S.StudyError):
        S.compile_manifest(spec)


def test_validate_study_on_missing_top_level_keys_names_each():
    issues = S.validate_study({"schema": S.SCHEMA})
    for key in ("name", "factors", "templates", "randomization", "instrument", "run"):
        assert errors_at(issues, key), key
