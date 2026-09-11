"""Pins the frozen factorial (docs/decisions/factorial.md, 2026-09-11). A change to prompts/grid.json that
alters the cell count, the per-arm total or the level codes fails here, so the freeze cannot drift silently."""
import json
from pathlib import Path

from harness import log

ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "prompts" / "grid.json"
EXAMPLE = ROOT / "harness" / "dyads.example.jsonl"


def grid():
    return json.loads(GRID.read_text())


def test_three_mentor_arms():
    assert grid()["arms"] == ["qwen3.6:35b-a3b", "gpt-oss:20b", "Olmo-3-7B-Instruct"]


def test_fixed_factor_levels():
    f = grid()["factors"]
    assert f["topic"] == ["immigration_enforcement", "decarbonization"]
    assert f["ideology"] == ["strong_left", "lean_left", "moderate", "lean_right", "strong_right"]
    assert f["openness"] == ["open", "closed"]


def test_twenty_treated_cells_plus_two_control():
    g = grid()
    f = g["factors"]
    treated = len(f["topic"]) * len(f["ideology"]) * len(f["openness"])
    assert treated == 20
    assert g["control"] == {"ideology": "none", "openness": None, "role": None, "n_per_cell": 135}
    assert len(f["topic"]) == 2  # one control cell per topic


def test_per_arm_total_matches_the_record():
    g = grid()
    f = g["factors"]
    treated = len(f["topic"]) * len(f["ideology"]) * len(f["openness"]) * g["n_per_cell"]
    control = len(f["topic"]) * g["control"]["n_per_cell"]
    assert treated == 2700 and control == 270 and treated + control == g["dialogues_per_arm"] == 2970


def test_n_per_cell_splits_evenly_across_variants():
    g = grid()
    assert g["variants_per_level"] == 3
    assert g["n_per_cell"] % g["variants_per_level"] == 0
    assert g["n_per_cell"] // g["variants_per_level"] == 45


def test_fixed_features_match_the_decisions():
    fixed = grid()["fixed"]
    assert fixed["n_turns"] == 40 and fixed["persona_mode"] == "reinforced"
    assert fixed["temperature"] == 0.7 and fixed["top_p"] == 0.95


def test_stated_once_extension_is_not_in_the_grid():
    e = grid()["extensions"]["E1_stated_once"]
    assert e["arms"] == ["qwen3.6:35b-a3b"] and e["persona_mode"] == "once" and e["n_per_cell"] == 45
    assert e["dialogues"] == 20 * 45


def test_example_manifest_uses_only_grid_levels():
    g = grid()
    f = g["factors"]
    rows = log.read_jsonl(EXAMPLE)
    assert rows, "dyads.example.jsonl is empty"
    for row in rows:
        c = row["condition"]
        assert set(c) == {"topic", "ideology", "openness", "role"}, c
        assert c["topic"] in f["topic"], c
        if c["ideology"] == g["control"]["ideology"]:
            assert c["openness"] is None and c["role"] is None, c
        else:
            assert c["ideology"] in f["ideology"] and c["openness"] in f["openness"], c
            assert isinstance(c["role"], str) and c["role"], c


def test_example_manifest_shows_a_control_row():
    rows = log.read_jsonl(EXAMPLE)
    assert any(r["condition"]["ideology"] == "none" for r in rows)
