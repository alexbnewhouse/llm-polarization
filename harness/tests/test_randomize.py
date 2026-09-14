"""The condition randomizer: prompts/grid.json + a persona catalogue -> a dyad manifest and its assignment log."""
import json
from pathlib import Path
import pytest
from harness import grid as G, log, randomize as RZ

ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "prompts" / "grid.json"
EXAMPLE_CATALOGUE = ROOT / "prompts" / "personas" / "catalogue.example.json"


def grid():
    return G.load_grid(GRID)


def catalogue():
    return RZ.load_catalogue(EXAMPLE_CATALOGUE)


# --- the grid gate -------------------------------------------------------------------------------------

def test_check_conditions_accepts_grid_levels_and_the_control_shape():
    rows = [{"condition": {"topic": "decarbonization", "ideology": "moderate", "openness": "closed", "role": "x"}},
            {"condition": {"topic": "immigration_enforcement", "ideology": "none", "openness": None, "role": None}}]
    G.check_conditions(rows, grid())            # no raise


@pytest.mark.parametrize("bad", [
    {"topic": "healthcare", "ideology": "moderate", "openness": "open", "role": "x"},
    {"topic": "decarbonization", "ideology": "centrist", "openness": "open", "role": "x"},
    {"topic": "decarbonization", "ideology": "moderate", "openness": "curious", "role": "x"},
    {"topic": "decarbonization", "ideology": "moderate", "openness": "open", "role": None},
    {"topic": "decarbonization", "ideology": "none", "openness": "open", "role": None},
    {"topic": "decarbonization", "ideology": "moderate", "openness": "open"},
])
def test_check_conditions_rejects_a_level_outside_the_grid(bad):
    with pytest.raises(ValueError):
        G.check_conditions([{"dyad_id": "d", "condition": bad}], grid())


# --- the catalogue ---------------------------------------------------------------------------------------

def test_example_catalogue_validates_against_the_grid():
    RZ.validate_catalogue(catalogue(), grid())


def test_catalogue_slugs_must_belong_to_a_grid_level_and_be_unique():
    c = catalogue()
    c["roles"]["centrist"] = [{"slug": "z", "name": "Z", "backstory": "b", "reminder_self": "r"}]
    with pytest.raises(ValueError, match="centrist"):
        RZ.validate_catalogue(c, grid())
    c = catalogue()
    c["roles"]["lean_left"].append(dict(c["roles"]["strong_left"][0]))
    with pytest.raises(ValueError, match="slug"):
        RZ.validate_catalogue(c, grid())
    c = catalogue()
    c["roles"]["moderate"] = c["roles"]["moderate"] * 4          # more variants than the grid allows
    with pytest.raises(ValueError, match="variants"):
        RZ.validate_catalogue(c, grid())


def test_catalogue_needs_an_anchor_set_for_every_ideology_and_topic_and_both_openness_texts():
    c = catalogue()
    del c["anchors"]["lean_right"]["decarbonization"]
    with pytest.raises(ValueError, match="lean_right"):
        RZ.validate_catalogue(c, grid())
    c = catalogue()
    del c["openness"]["closed"]
    with pytest.raises(ValueError, match="closed"):
        RZ.validate_catalogue(c, grid())


def test_render_persona_fills_every_slot_and_the_opening_line_is_identical_in_every_cell():
    c = catalogue()
    role = c["roles"]["strong_left"][0]
    text, reminder = RZ.render_persona(c, "strong_left", role, "immigration_enforcement", "open")
    assert role["name"] in text and role["backstory"] in text
    assert all(a in text for a in c["anchors"]["strong_left"]["immigration_enforcement"])
    assert c["openness"]["open"]["text"] in text and c["openness"]["open"]["reminder"] in reminder
    assert reminder.startswith("Note to self")
    assert "{" not in text and "{" not in reminder
    opening = RZ.opening_line(c, "immigration_enforcement")
    assert text.endswith(opening)
    other, _ = RZ.render_persona(c, "strong_right", c["roles"]["strong_right"][0], "immigration_enforcement", "closed")
    assert other.endswith(opening)
    ctext, creminder = RZ.render_control(c, "immigration_enforcement")
    assert ctext.endswith(opening) and role["name"] not in ctext and "Note to self" in creminder


def test_render_persona_refuses_an_unfilled_slot():
    c = catalogue()
    c["template"]["persona"] += " {hobby}"
    with pytest.raises(ValueError, match="hobby"):
        RZ.render_persona(c, "moderate", c["roles"]["moderate"][0], "decarbonization", "open")


# --- the manifest ------------------------------------------------------------------------------------------

def build(**kw):
    kw.setdefault("seed", 20260918)
    return RZ.build_manifest(grid(), catalogue(), **kw)


def test_pilot_manifest_has_the_pilot_grid():
    # 20 treated cells x 5 x 2 delivery modes + 2 control x 5 = 210 (docs/decisions/factorial.md, "The pilot")
    rows, assignment = build(n_per_cell=5, modes=("reinforced", "once"), prefix="p")
    assert len(rows) == 210
    assert sum(1 for r in rows if r["condition"]["ideology"] == "none") == 10
    assert all(r["n_turns"] == 40 for r in rows)
    G.check_conditions(rows, grid())
    assert assignment["rows"] == 210 and assignment["n_per_cell"] == 5 and assignment["modes"] == ["reinforced", "once"]


def test_wave_manifest_splits_each_cell_evenly_across_the_levels_variants():
    rows, assignment = build()                     # grid defaults: 135 per cell, reinforced, 40 turns
    assert len(rows) == grid()["dialogues_per_arm"] == 2970
    treated = [r for r in rows if r["condition"]["ideology"] != "none"]
    cells = {}
    for r in treated:
        c = r["condition"]
        cells.setdefault((c["topic"], c["ideology"], c["openness"]), {}).setdefault(c["role"], 0)
        cells[(c["topic"], c["ideology"], c["openness"])][c["role"]] += 1
    assert len(cells) == 20
    n_variants = {lvl: len(v) for lvl, v in catalogue()["roles"].items()}
    for (topic, ideology, openness), by_role in cells.items():
        assert len(by_role) == n_variants[ideology]
        assert set(by_role.values()) == {135 // n_variants[ideology]}
    assert assignment["rows_per_cell"][f"immigration_enforcement/strong_left/open"] == 135
    assert sum(assignment["rows_per_cell"].values()) == 2970
    assert sum(assignment["rows_per_variant"].values()) == 2700
    assert assignment["rows_per_mode"] == {"reinforced": 2970}


def test_n_per_cell_must_split_evenly_across_the_variants():
    c = catalogue()
    c["roles"]["moderate"] = c["roles"]["moderate"] + [dict(c["roles"]["moderate"][0], slug="second_moderate")]
    with pytest.raises(ValueError, match="evenly"):
        RZ.build_manifest(grid(), c, seed=1, n_per_cell=5)


def test_control_rows_are_bare_and_reinforced_with_the_framing_only_reminder():
    rows, _ = build(n_per_cell=3, modes=("reinforced", "once"))
    control = [r for r in rows if r["condition"]["ideology"] == "none"]
    assert len(control) == 6 and {r["condition"]["topic"] for r in control} == set(grid()["factors"]["topic"])
    assert all(r["condition"] == {"topic": r["condition"]["topic"], "ideology": "none", "openness": None, "role": None} for r in control)
    assert all(r["persona_mode"] == "reinforced" for r in control)    # the control is always reinforced
    assert all(r["persona_reminder"].startswith("Note to self") for r in control)
    assert all(r["persona_text"] == RZ.render_control(catalogue(), r["condition"]["topic"])[0] for r in control)


def test_two_rows_in_the_same_cell_differ_only_in_seed_role_and_id():
    rows, _ = build(n_per_cell=3)
    same = [r for r in rows if r["condition"]["topic"] == "decarbonization" and r["condition"]["ideology"] == "lean_left"
            and r["condition"]["openness"] == "closed"]
    assert len(same) == 3 and len({r["seed"] for r in same}) == 3
    a, b = same[0], same[1]
    for key in ("persona_mode", "n_turns"):
        assert a[key] == b[key]
    if a["condition"]["role"] == b["condition"]["role"]:
        assert a["persona_text"] == b["persona_text"] and a["persona_reminder"] == b["persona_reminder"]


def test_seeds_are_distinct_reproducible_from_the_rng_seed_and_ids_unique():
    rows1, a1 = build(n_per_cell=6)
    rows2, a2 = build(n_per_cell=6)
    assert rows1 == rows2 and a1["rng_seed"] == a2["rng_seed"] == 20260918
    assert len({r["seed"] for r in rows1}) == len(rows1)
    assert len({r["dyad_id"] for r in rows1}) == len(rows1)
    rows3, _ = build(n_per_cell=6, seed=1)
    assert [r["seed"] for r in rows3] != [r["seed"] for r in rows1]
    assert all(0 < r["seed"] < 2 ** 31 for r in rows1)


def test_rows_are_shuffled_so_cells_interleave():
    # A sequential file would run one cell at a time across the 8 slots and confound cell with wall-clock
    # drift on the box. The first 16 rows must span more than two cells.
    rows, _ = build(n_per_cell=6)
    first_cells = {(r["condition"]["topic"], r["condition"]["ideology"], r["condition"]["openness"]) for r in rows[:16]}
    assert len(first_cells) > 2


def test_assignment_log_pins_grid_catalogue_and_rng(tmp_path):
    rows, assignment = build(n_per_cell=3, prefix="t")
    out = tmp_path / "t-dyads.jsonl"
    written = RZ.write_manifest(out, rows, assignment, grid_path=GRID, catalogue_path=EXAMPLE_CATALOGUE)
    assert written == tmp_path / "t-assignment.json"
    a = json.loads(written.read_text())
    assert a["grid"]["sha256"] == log.sha256_file(GRID) and a["grid"]["frozen"] == "2026-09-11"
    assert a["catalogue"]["sha256"] == log.sha256_file(EXAMPLE_CATALOGUE) and a["catalogue"]["version"]
    assert a["output"]["sha256"] == log.sha256_file(out) and a["rng_seed"] == 20260918
    assert a["variants_per_level"]["strong_left"] == [r["slug"] for r in catalogue()["roles"]["strong_left"]]
    assert a["harness_version"] and a["generated"]
    back = log.read_jsonl(out)
    assert back == rows
    # the written file passes the same gate the harness runs, and the run's own manifest validator
    G.check_conditions(back, grid())
    from harness.run import validate_manifest_rows
    validate_manifest_rows(back)


def test_cli_writes_manifest_and_sidecar_and_refuses_a_bad_catalogue(tmp_path, capsys):
    out = tmp_path / "pilot-dyads.jsonl"
    rc = RZ.main(["--grid", str(GRID), "--catalogue", str(EXAMPLE_CATALOGUE), "--out", str(out),
                  "--seed", "20260918", "--n-per-cell", "5", "--modes", "reinforced,once", "--prefix", "p"])
    assert rc == 0 and out.exists() and (tmp_path / "pilot-assignment.json").exists()
    assert len(log.read_jsonl(out)) == 210
    assert "210" in capsys.readouterr().out
    bad = tmp_path / "bad.json"
    c = catalogue(); c["roles"]["centrist"] = c["roles"].pop("moderate")
    bad.write_text(json.dumps(c))
    rc = RZ.main(["--grid", str(GRID), "--catalogue", str(bad), "--out", str(tmp_path / "x.jsonl"), "--seed", "1"])
    assert rc == 1 and not (tmp_path / "x.jsonl").exists()
    assert "centrist" in capsys.readouterr().err


def test_cli_check_validates_the_catalogue_without_writing_anything(tmp_path, capsys):
    rc = RZ.main(["--grid", str(GRID), "--catalogue", str(EXAMPLE_CATALOGUE), "--check"])
    assert rc == 0 and not list(tmp_path.iterdir())
    assert "valid" in capsys.readouterr().out
    bad = tmp_path / "bad.json"
    c = catalogue(); del c["anchors"]["moderate"]["decarbonization"]
    bad.write_text(json.dumps(c))
    assert RZ.main(["--grid", str(GRID), "--catalogue", str(bad), "--check"]) == 1
    assert "moderate" in capsys.readouterr().err
