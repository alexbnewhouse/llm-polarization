"""The repo study (sandbox/repo_study.py): built from prompts/grid.json, the persona catalogue, the batteries and
the config; compiled by sandbox.study to exactly the rows harness.randomize.build_manifest writes; mapped back
to the files on disk."""
import copy
import json
import shutil
import pytest
from harness import randomize as RZ
from harness.grid import load_grid
from sandbox import repo_study as R, study as S


def files(root):
    src = R.repo_sources(root)
    return load_grid(src["grid"]), RZ.load_catalogue(src["catalogue"])


def three_variant_catalogue(catalogue):
    """The example catalogue with three role variants under every level but one: the variant loop's order and
    the per-variant split are exercised, as the real catalogue will."""
    c = copy.deepcopy(catalogue)
    for level, roles in c["roles"].items():
        if level == "moderate":
            continue
        base = roles[0]
        roles.extend(dict(base, slug=f"{base['slug']}_{n}", name=f"{base['name']} {n}") for n in (2, 3))
    return c


def dumped(rows):
    """Rows as the JSONL lines write_manifest writes: key order counts."""
    return [json.dumps(r, ensure_ascii=False) for r in rows]


def same_assignment(ours, theirs):
    for key, value in theirs.items():
        if key == "generated":
            continue
        assert json.dumps(ours[key]) == json.dumps(value), key
    assert ours["engine"] == "sandbox.study"


# --- sources and loading ------------------------------------------------------------------------------------

def test_repo_sources_fall_back_to_the_example_catalogue_and_config(repo_root, tmp_path):
    src = R.repo_sources(repo_root)
    assert set(src) == {"grid", "catalogue", "batteries", "config"}
    assert all(p.is_absolute() and p.exists() for p in src.values())
    assert src["grid"] == repo_root / "prompts" / "grid.json"
    assert src["batteries"] == repo_root / "instruments" / "batteries.json"
    real_catalogue = repo_root / "prompts" / "personas" / "catalogue.json"
    example_catalogue = repo_root / "prompts" / "personas" / "catalogue.example.json"
    expected = real_catalogue if real_catalogue.exists() else example_catalogue
    assert src["catalogue"] == expected
    # a root with the real catalogue and a config.json uses them
    for rel in ("prompts/grid.json", "prompts/personas/catalogue.example.json", "instruments/batteries.json",
                "harness/config.example.json"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(repo_root / rel, tmp_path / rel)
    src = R.repo_sources(tmp_path)
    assert src["catalogue"].name == "catalogue.example.json" and src["config"].name == "config.example.json"
    shutil.copy(tmp_path / "prompts/personas/catalogue.example.json", tmp_path / "prompts/personas/catalogue.json")
    shutil.copy(tmp_path / "harness/config.example.json", tmp_path / "config.json")
    src = R.repo_sources(tmp_path)
    assert src["catalogue"] == tmp_path.resolve() / "prompts/personas/catalogue.json"
    assert src["config"] == tmp_path.resolve() / "config.json"
    spec = R.load_repo_study(tmp_path)
    assert spec["repo"]["sources"] == {"grid": "prompts/grid.json", "catalogue": "prompts/personas/catalogue.json",
                                       "batteries": "instruments/batteries.json", "config": "config.json"}


def test_load_repo_study_maps_the_files_and_validates_clean(repo_root):
    spec = R.load_repo_study(repo_root)
    issues = S.validate_study(spec)
    assert not S.has_errors(issues), issues
    grid, catalogue = files(repo_root)
    assert [f["key"] for f in spec["factors"]] == ["topic", "ideology", "openness"]
    assert spec["factors"][0]["levels"][0] == {"id": "immigration_enforcement", "code": "immigr",
                                               "slots": {"topic_phrase": "immigration enforcement"}}
    assert all("code" not in lv for f in spec["factors"][1:] for lv in f["levels"])
    assert spec["nested"]["key"] == "role" and spec["nested"]["within"] == "ideology"
    assert spec["nested"]["max_per_level"] == grid["variants_per_level"]
    assert spec["tables"][0]["values"] == catalogue["anchors"]
    assert spec["derived"] == {"opening": catalogue["shared"]["opening"]}
    assert spec["control"]["level"] == "none" and spec["control"]["n_per_cell"] is None
    assert spec["randomization"] == {"seed": 20261005, "n_per_cell": 135, "modes": ["reinforced"], "n_turns": 40,
                                     "prefix": "w", "cells": None}
    batteries = json.loads((repo_root / "instruments" / "batteries.json").read_text())
    assert spec["instrument"] == batteries
    config = json.loads(R.repo_sources(repo_root)["config"].read_text())
    assert spec["run"] == {k: v for k, v in config.items() if k not in ("batteries", "grid")}
    assert "batteries" not in spec["run"] and "grid" not in spec["run"]
    assert spec["repo"]["grid"] == grid
    assert spec["repo"]["catalogue"] == {k: v for k, v in catalogue.items()
                                         if k not in ("template", "shared", "openness", "roles", "anchors", "control")}
    assert spec["repo"]["sources"]["grid"] == "prompts/grid.json"


def test_to_repo_files_round_trips_the_files_on_disk(repo_root):
    src = R.repo_sources(repo_root)
    grid_on_disk = json.loads(src["grid"].read_text(encoding="utf-8"))
    catalogue_on_disk = json.loads(src["catalogue"].read_text(encoding="utf-8"))
    spec = R.load_repo_study(repo_root)
    assert R.repo_shape_reason(spec) is None
    grid, catalogue = R.to_repo_files(spec)
    assert (grid, catalogue) == (grid_on_disk, catalogue_on_disk)
    assert list(catalogue) == list(catalogue_on_disk) and list(grid) == list(grid_on_disk)
    assert json.dumps(catalogue) == json.dumps(catalogue_on_disk)


def test_unknown_catalogue_keys_survive_the_round_trip(repo_root):
    grid, catalogue = files(repo_root)
    catalogue["provenance"] = {"author": "x"}
    spec = R.from_repo_files(grid, catalogue, {"items": []}, {"seeker": {}, "mentor": {}})
    _, back = R.to_repo_files(spec)
    assert back == catalogue and list(back)[-1] == "provenance"


def test_from_repo_files_refuses_a_catalogue_it_cannot_map(repo_root):
    grid, catalogue = files(repo_root)
    del catalogue["shared"]
    with pytest.raises(ValueError, match="shared"):
        R.from_repo_files(grid, catalogue, {"items": []}, {})


# --- the randomizer equivalence ----------------------------------------------------------------------------

def test_compile_manifest_equals_build_manifest_for_the_grid_defaults(repo_root):
    grid, catalogue = files(repo_root)
    spec = R.load_repo_study(repo_root)
    rows, assignment = S.compile_manifest(spec)
    cli_rows, cli_assignment = RZ.build_manifest(grid, catalogue, seed=20261005)     # randomize --seed 20261005
    assert len(rows) == 2970
    assert dumped(rows) == dumped(cli_rows)
    same_assignment(assignment, cli_assignment)
    kw_rows, kw_assignment = RZ.build_manifest(grid, catalogue, **R.build_manifest_kwargs(spec))
    assert dumped(kw_rows) == dumped(rows)
    same_assignment(assignment, kw_assignment)


@pytest.mark.parametrize("preset,cli", [
    ("pilot", {"seed": 20260918, "n_per_cell": 5, "modes": ("reinforced", "once"), "prefix": "p"}),
    ("wave1", {"seed": 20261005, "prefix": "w1"}),
])
def test_compile_manifest_equals_build_manifest_for_the_presets(repo_root, preset, cli):
    grid, catalogue = files(repo_root)
    spec = R.load_repo_study(repo_root)
    spec["randomization"] = copy.deepcopy(R.RANDOMIZATION_PRESETS[preset])
    rows, assignment = S.compile_manifest(spec)
    cli_rows, cli_assignment = RZ.build_manifest(grid, catalogue, **cli)      # the command in prompts/README.md
    assert len(rows) == {"pilot": 210, "wave1": 2970}[preset]
    assert dumped(rows) == dumped(cli_rows)
    same_assignment(assignment, cli_assignment)
    for key in ("rows_per_cell", "rows_per_variant", "rows_per_mode"):
        assert assignment[key] == cli_assignment[key]


def test_compile_manifest_equals_build_manifest_with_three_variants_per_level(repo_root):
    grid, catalogue = files(repo_root)
    catalogue = three_variant_catalogue(catalogue)
    config = json.loads(R.repo_sources(repo_root)["config"].read_text())
    spec = R.from_repo_files(grid, catalogue, {"items": []}, config)
    for randomization, cli in [
        (None, {"seed": 20261005}),
        (R.RANDOMIZATION_PRESETS["wave1"], {"seed": 20261005, "prefix": "w1"}),
        ({"seed": 3, "n_per_cell": 3, "modes": ["once", "reinforced"], "n_turns": 7, "prefix": "z", "cells": None},
         {"seed": 3, "n_per_cell": 3, "modes": ("once", "reinforced"), "n_turns": 7, "prefix": "z"}),
    ]:
        if randomization:
            spec["randomization"] = copy.deepcopy(randomization)
        rows, assignment = S.compile_manifest(spec)
        cli_rows, cli_assignment = RZ.build_manifest(grid, catalogue, **cli)
        assert dumped(rows) == dumped(cli_rows)
        same_assignment(assignment, cli_assignment)


def test_a_grid_control_size_of_its_own_is_kept(repo_root):
    grid, catalogue = files(repo_root)
    grid["control"]["n_per_cell"] = 60
    spec = R.from_repo_files(grid, catalogue, {"items": []}, {})
    assert spec["control"]["n_per_cell"] == 60 and S.resolved_n_control(spec) == 60
    rows, _ = S.compile_manifest(spec)
    assert dumped(rows) == dumped(RZ.build_manifest(grid, catalogue, seed=20261005)[0])
    assert R.build_manifest_kwargs(spec)["n_control"] == 60


def test_build_manifest_kwargs(repo_root):
    spec = R.load_repo_study(repo_root)
    spec["randomization"] = copy.deepcopy(R.RANDOMIZATION_PRESETS["pilot"])
    assert R.build_manifest_kwargs(spec) == {"seed": 20260918, "n_per_cell": 5, "n_control": 5,
                                             "modes": ("reinforced", "once"), "n_turns": 40, "prefix": "p"}


def test_randomization_presets_follow_prompts_readme():
    assert R.RANDOMIZATION_PRESETS["pilot"] == {"seed": 20260918, "n_per_cell": 5, "modes": ["reinforced", "once"],
                                                "n_turns": 40, "prefix": "p", "cells": None}
    assert R.RANDOMIZATION_PRESETS["wave1"] == {"seed": 20261005, "n_per_cell": 135, "modes": ["reinforced"],
                                                "n_turns": 40, "prefix": "w1", "cells": None}


def test_a_subset_of_the_repo_study_is_the_full_manifest_restricted_to_those_cells(repo_root):
    spec = R.load_repo_study(repo_root)
    full, _ = S.compile_manifest(spec)
    keys = ["decarbonization/moderate/closed", "immigration_enforcement/none"]
    spec["randomization"]["cells"] = keys
    sub, assignment = S.compile_manifest(spec)

    def key(r):
        c = r["condition"]
        return f"{c['topic']}/none" if c["ideology"] == "none" else f"{c['topic']}/{c['ideology']}/{c['openness']}"
    assert dumped(sub) == dumped([r for r in full if key(r) in keys])
    assert len(sub) == 270 and assignment["rows_per_cell"] == {k: 135 for k in keys}


# --- cells and rendering -------------------------------------------------------------------------------------

def test_enumerate_cells_and_summary_of_the_repo_study(repo_root):
    spec = R.load_repo_study(repo_root)
    cells = S.enumerate_cells(spec)
    assert len(cells) == 22
    assert sum(c["kind"] == "treated" for c in cells) == 20 and sum(c["kind"] == "control" for c in cells) == 2
    assert sum(c["n_rows"] for c in cells) == 2970
    assert cells[0]["key"] == "immigration_enforcement/strong_left/open"
    assert cells[-1] == {"key": "decarbonization/none", "kind": "control", "condition": {"topic": "decarbonization"},
                         "variants": [], "n_rows": 135, "selected": True}
    s = S.summarize(spec)
    assert s["cells_treated"] == 20 and s["cells_control"] == 2 and s["rows"] == 2970
    assert s["rows_per_mode"] == {"reinforced": 2970} and s["messages"] == 2970 * 40 * 2
    assert s["condition_keys"] == ["topic", "ideology", "openness", "role"]


def test_render_cell_matches_the_harness_renderers(repo_root):
    grid, catalogue = files(repo_root)
    catalogue = three_variant_catalogue(catalogue)
    spec = R.from_repo_files(grid, catalogue, {"items": []}, {})
    role = catalogue["roles"]["lean_right"][2]
    out = S.render_cell(spec, "treated", {"topic": "decarbonization", "ideology": "lean_right", "openness": "closed"},
                        role["slug"])
    assert (out["persona_text"], out["persona_reminder"]) == RZ.render_persona(
        catalogue, "lean_right", role, "decarbonization", "closed")
    ctl = S.render_cell(spec, "control", {"topic": "immigration_enforcement", "ideology": "none", "openness": None,
                                          "role": None})
    assert (ctl["persona_text"], ctl["persona_reminder"]) == RZ.render_control(catalogue, "immigration_enforcement")


# --- repo shape --------------------------------------------------------------------------------------------

def test_editing_text_keeps_the_study_repo_shaped_and_the_files_follow(repo_root):
    spec = R.load_repo_study(repo_root)
    spec["templates"]["persona"] = "You are {name}, {role}. {backstory} {anchor_1} {openness_text} {opening}"
    spec["factors"][2]["levels"][1]["slots"]["openness_text"] = "You are not going to change your mind."
    spec["nested"]["variants"]["moderate"][0]["slots"]["name"] = "Sam Ortiz"
    spec["control"]["reminder"] = "Note to self: ask about {topic_phrase}."
    spec["name"] = "edited"
    assert R.repo_shape_reason(spec) is None
    grid, catalogue = R.to_repo_files(spec)
    assert catalogue["template"]["persona"] == spec["templates"]["persona"]
    assert catalogue["openness"]["closed"]["text"] == "You are not going to change your mind."
    assert catalogue["roles"]["moderate"][0]["name"] == "Sam Ortiz"
    # the files the export would write give the same manifest as the spec
    spec["randomization"] = copy.deepcopy(R.RANDOMIZATION_PRESETS["pilot"])
    rows, _ = S.compile_manifest(spec)
    assert dumped(rows) == dumped(RZ.build_manifest(grid, catalogue, **R.build_manifest_kwargs(spec))[0])


def test_changing_levels_and_variants_keeps_it_repo_shaped(repo_root):
    spec = R.load_repo_study(repo_root)
    spec["factors"][1]["levels"].pop()                                   # drop strong_right
    del spec["nested"]["variants"]["strong_right"]
    del spec["tables"][0]["values"]["strong_right"]
    spec["control"]["level"] = "no_persona"
    assert R.repo_shape_reason(spec) is None
    grid, catalogue = R.to_repo_files(spec)
    assert grid["factors"]["ideology"] == ["strong_left", "lean_left", "moderate", "lean_right"]
    assert grid["control"]["ideology"] == "no_persona" and "strong_right" not in catalogue["roles"]
    rows, _ = S.compile_manifest(spec)
    assert dumped(rows) == dumped(RZ.build_manifest(grid, catalogue, **R.build_manifest_kwargs(spec))[0])


def _fourth_factor(spec):
    spec["factors"].append({"key": "tone", "levels": [{"id": "warm", "slots": {}}]})


@pytest.mark.parametrize("mutate,words", [
    (_fourth_factor, "factor"),
    (lambda s: s.update(repo=None), "repo"),
    (lambda s: s.update(nested=None), "nested"),
    (lambda s: s.update(control=None), "control"),
    (lambda s: s["control"].update(persona_mode="once"), "reinforced"),
    (lambda s: s["control"].update(by=[]), "by"),
    (lambda s: s["tables"][0].update(join=" | "), "join"),
    (lambda s: s["tables"][0].update(name="stances"), "anchors"),
    (lambda s: s["tables"].append(dict(s["tables"][0], name="more")), "table"),
    (lambda s: s["derived"].update(closing="Bye."), "derived"),
    (lambda s: s["derived"].update(opening="Hi {name}, {topic_phrase}."), "opening"),
    (lambda s: s["factors"][0]["levels"][0].update(code="imm"), "code"),
    (lambda s: s["factors"][1]["levels"][0].update(code="sl"), "code"),
    (lambda s: s["factors"][0]["levels"][0]["slots"].update(extra="x"), "slots"),
    (lambda s: s["nested"].update(key="persona"), "role"),
    (lambda s: s["nested"]["variants"]["moderate"][0]["slots"].pop("backstory"), "backstory"),
    (lambda s: s["nested"]["variants"]["moderate"][0]["slots"].update(age="51") or
     s["templates"].update(persona=s["templates"]["persona"] + " {age}"), "age"),
    (lambda s: s["repo"]["grid"]["control"].update(openness="open"), "openness"),
    (lambda s: s["tables"][0]["values"]["moderate"].pop("decarbonization"), "anchor"),
])
def test_repo_shape_reason_says_why_a_study_is_not_the_repos(repo_root, mutate, words):
    spec = R.load_repo_study(repo_root)
    mutate(spec)
    reason = R.repo_shape_reason(spec)
    assert reason and words in reason, reason
    with pytest.raises(R.NotRepoShaped, match=words):
        R.to_repo_files(spec)


@pytest.mark.parametrize("spec", [None, [], {}, {"repo": {}}, {"repo": {"grid": {}, "catalogue": {}}, "factors": 3}])
def test_repo_shape_reason_never_raises(spec):
    assert isinstance(R.repo_shape_reason(spec), str)


def test_the_example_study_is_not_repo_shaped(repo_root):
    spec = json.loads((repo_root / "studies" / "example-institutional-trust.study.json").read_text())
    assert "repo" in R.repo_shape_reason(spec)
