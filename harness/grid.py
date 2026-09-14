"""The frozen factorial as code reads it: load prompts/grid.json and check manifest rows against it, so a
manifest that uses a level outside the grid never reaches the box (docs/decisions/factorial.md)."""
from __future__ import annotations
import json
from pathlib import Path

CONDITION_KEYS = ("topic", "ideology", "openness", "role")


def load_grid(path: str | Path) -> dict:
    """Read prompts/grid.json (or any file of the same shape)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def treated_cells(grid: dict) -> list[tuple[str, str, str]]:
    """Every (topic, ideology, openness) cell of the crossed factors, in grid order."""
    f = grid["factors"]
    return [(t, i, o) for t in f["topic"] for i in f["ideology"] for o in f["openness"]]


def check_condition(condition: dict, grid: dict, where: str = "") -> None:
    """Raise ValueError unless `condition` names a grid cell: all four keys, topic and levels from the
    grid, or the control shape (`ideology` = the control's, openness and role null)."""
    f = grid["factors"]
    control = grid["control"]["ideology"]
    if not isinstance(condition, dict) or set(condition) != set(CONDITION_KEYS):
        raise ValueError(f"{where}condition must have exactly the keys {CONDITION_KEYS}, got {condition!r}")
    if condition["topic"] not in f["topic"]:
        raise ValueError(f"{where}topic {condition['topic']!r} is not in the grid {f['topic']}")
    if condition["ideology"] == control:
        if condition["openness"] is not None or condition["role"] is not None:
            raise ValueError(f"{where}a control row has openness and role null, got {condition!r}")
        return
    if condition["ideology"] not in f["ideology"]:
        raise ValueError(f"{where}ideology {condition['ideology']!r} is not in the grid {f['ideology']}")
    if condition["openness"] not in f["openness"]:
        raise ValueError(f"{where}openness {condition['openness']!r} is not in the grid {f['openness']}")
    if not isinstance(condition["role"], str) or not condition["role"]:
        raise ValueError(f"{where}a treated row needs a role slug, got {condition['role']!r}")


def check_conditions(rows: list[dict], grid: dict) -> None:
    """check_condition over every manifest row, naming the line and dyad_id in the error."""
    for i, row in enumerate(rows, 1):
        check_condition(row.get("condition"), grid, where=f"dyad manifest line {i} ({row.get('dyad_id')}): ")
