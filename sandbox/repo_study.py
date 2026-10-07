"""The repo's own study as a sandbox spec (docs/superpowers/specs/2026-10-07-sandbox-gui-design.md section 4).

`load_repo_study` builds the spec live from prompts/grid.json, the persona catalogue, instruments/batteries.json
and the config on every call, so the GUI never holds a stale copy and nothing about the study is saved twice.
`to_repo_files` inverts the mapping into (grid, catalogue) for a spec that is still repo-shaped, so an export
of the repo study goes through `harness.randomize.build_manifest` itself; `repo_shape_reason` says in words
why a spec no longer is (a fourth factor, a renamed table, a template slot the catalogue cannot fill)."""
from __future__ import annotations
import copy
import json
import re
from pathlib import Path
from harness.grid import load_grid
from harness.randomize import CONTROL_MODE, _topic_code, load_catalogue, validate_catalogue
from sandbox.study import SCHEMA, resolved_n_control, template_slots

REPO_STUDY_NAME = "LLM political polarization (repo study)"
REPO_STUDY_DESCRIPTION = ("The study this repository implements, built live from prompts/grid.json, the persona "
                          "catalogue, instruments/batteries.json and the config. Exported unchanged, it is written "
                          "by harness.randomize itself.")
# prompts/README.md: the pilot and wave 1 commands. The repo study's own default is the grid's defaults with the
# randomizer's default prefix (seed 20261005, prefix w), which is `randomize --seed 20261005` with no options.
RANDOMIZATION_PRESETS: dict = {
    "pilot": {"seed": 20260918, "n_per_cell": 5, "modes": ["reinforced", "once"], "n_turns": 40, "prefix": "p",
              "cells": None},
    "wave1": {"seed": 20261005, "n_per_cell": 135, "modes": ["reinforced"], "n_turns": 40, "prefix": "w1",
              "cells": None},
}
DEFAULT_SEED = 20261005
DEFAULT_PREFIX = "w"

REPO_FACTORS = ["topic", "ideology", "openness"]
LEVEL_SLOTS = {"topic": ["topic_phrase"], "ideology": [], "openness": ["openness_text", "openness_reminder"]}
ROLE_FIELDS = ("name", "backstory", "reminder_self")
MAPPED_CATALOGUE_KEYS = ("template", "shared", "openness", "roles", "anchors", "control")
CATALOGUE_ORDER = ("version", "_note", *MAPPED_CATALOGUE_KEYS)
# The slots harness.randomize.render_persona / render_control / opening_line fill: a template using anything
# else renders in the sandbox but would be refused by the repo's randomizer.
TREATED_SLOTS = {"topic", "topic_phrase", "opening", "ideology", "role", "name", "backstory", "reminder_self",
                 "anchors", "openness", "openness_text", "openness_reminder"}
CONTROL_SLOTS = {"topic", "topic_phrase", "opening"}
OPENING_SLOTS = {"topic", "topic_phrase"}
_ANCHOR_ITEM = re.compile(r"anchor_[1-9][0-9]*")


class NotRepoShaped(ValueError):
    """A spec that cannot be written as the repo's grid.json and catalogue; the message says why."""


def repo_sources(root) -> dict[str, Path]:
    """The four files the repo study is built from, as absolute paths: the persona catalogue is
    prompts/personas/catalogue.json once it is written, else catalogue.example.json; the config is config.json
    at the root when there is one, else harness/config.example.json."""
    root = Path(root).resolve()
    catalogue = root / "prompts" / "personas" / "catalogue.json"
    if not catalogue.exists():
        catalogue = root / "prompts" / "personas" / "catalogue.example.json"
    config = root / "config.json"
    if not config.exists():
        config = root / "harness" / "config.example.json"
    return {"grid": root / "prompts" / "grid.json", "catalogue": catalogue,
            "batteries": root / "instruments" / "batteries.json", "config": config}


def load_repo_study(root) -> dict:
    """The repo study, read from the files on disk now (repo_sources), with root-relative source paths."""
    root = Path(root).resolve()
    src = repo_sources(root)
    sources = {}
    for key, path in src.items():
        try:
            sources[key] = path.relative_to(root).as_posix()
        except ValueError:
            sources[key] = str(path)
    return from_repo_files(load_grid(src["grid"]), load_catalogue(src["catalogue"]),
                           json.loads(src["batteries"].read_text(encoding="utf-8")),
                           json.loads(src["config"].read_text(encoding="utf-8")), sources)


def from_repo_files(grid: dict, catalogue: dict, batteries: dict, config: dict, sources: dict | None = None) -> dict:
    """Map the repo's files onto a spec (the table in spec section 4). The inputs are copied, never aliased.
    Raises ValueError naming what is missing when a file lacks a part the mapping reads."""
    try:
        return _from_repo_files(copy.deepcopy(grid), copy.deepcopy(catalogue), copy.deepcopy(batteries),
                                copy.deepcopy(config), dict(sources or {}))
    except (KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"cannot build the repo study from these files: missing or malformed {e}") from None


def _from_repo_files(grid: dict, catalogue: dict, batteries: dict, config: dict, sources: dict) -> dict:
    f, shared, openness = grid["factors"], catalogue["shared"], catalogue["openness"]
    control = grid["control"]
    return {
        "schema": SCHEMA,
        "name": REPO_STUDY_NAME,
        "description": REPO_STUDY_DESCRIPTION,
        "factors": [
            {"key": "topic", "label": "Topic",
             "levels": [{"id": t, "code": _topic_code(t), "slots": {"topic_phrase": shared["topic_phrase"][t]}}
                        for t in f["topic"]]},
            {"key": "ideology", "label": "Ideology", "levels": [{"id": i, "slots": {}} for i in f["ideology"]]},
            {"key": "openness", "label": "Openness",
             "levels": [{"id": o, "slots": {"openness_text": openness[o]["text"],
                                            "openness_reminder": openness[o]["reminder"]}} for o in f["openness"]]},
        ],
        "nested": {"key": "role", "within": "ideology", "max_per_level": grid["variants_per_level"],
                   "variants": {level: [{"id": r["slug"], "slots": {k: v for k, v in r.items() if k != "slug"}}
                                        for r in roles] for level, roles in catalogue["roles"].items()}},
        "tables": [{"name": "anchors", "by": ["ideology", "topic"], "join": " ", "item_slot": "anchor_{i}",
                    "values": catalogue["anchors"]}],
        "derived": {"opening": shared["opening"]},
        "templates": {"persona": catalogue["template"]["persona"], "reminder": catalogue["template"]["reminder"]},
        "control": {"factor": "ideology", "level": control["ideology"], "by": ["topic"],
                    "persona": catalogue["control"]["persona"], "reminder": catalogue["control"]["reminder"],
                    "persona_mode": CONTROL_MODE,
                    # build_manifest sizes the control like a treated cell unless the grid gives it its own size
                    "n_per_cell": None if control["n_per_cell"] == grid["n_per_cell"] else control["n_per_cell"]},
        "randomization": {"seed": DEFAULT_SEED, "n_per_cell": grid["n_per_cell"],
                          "modes": [grid["fixed"]["persona_mode"]], "n_turns": grid["fixed"]["n_turns"],
                          "prefix": DEFAULT_PREFIX, "cells": None},
        "instrument": batteries,
        "run": {k: v for k, v in config.items() if k not in ("batteries", "grid")},
        "repo": {"grid": copy.deepcopy(grid),
                 "catalogue": {k: v for k, v in catalogue.items() if k not in MAPPED_CATALOGUE_KEYS},
                 "sources": sources},
    }


def repo_shape_reason(spec) -> str | None:
    """None when to_repo_files(spec) would succeed, else a sentence saying why the spec is not the repo's
    shape. Text (templates, level slot values, persona variants, anchors) may change freely; structure (the
    three factors, the role nesting, the one anchors table, the opening, the control) may not, and every
    template may use only the slots the repo's randomizer fills. Never raises."""
    try:
        reason = _shape_reason(spec)
        if reason:
            return reason
        grid, catalogue = _build_files(spec)
        validate_catalogue(catalogue, grid)
    except ValueError as e:
        return f"the repo's randomizer would refuse the files: {e}"
    except Exception as e:                       # a malformed draft: say so rather than fail the request
        return f"the study is not well-formed enough to map onto the repo's files ({type(e).__name__}: {e})"
    return None


def _shape_reason(spec) -> str | None:
    if not isinstance(spec, dict):
        return "the study is not an object"
    repo = spec.get("repo")
    if not (isinstance(repo, dict) and isinstance(repo.get("grid"), dict) and isinstance(repo.get("catalogue"), dict)):
        return "the study has no repo block: it was not loaded from the repo's files"
    if "version" not in repo["catalogue"]:
        return "repo.catalogue has no version"
    stored_control = repo["grid"].get("control")
    if not isinstance(stored_control, dict):
        return "repo.grid has no control"
    for key in ("openness", "role"):
        if stored_control.get(key) is not None:
            return f"the stored grid's control.{key} is {stored_control.get(key)!r}; a control row has {key} null"
    factors = spec.get("factors")
    keys = [f.get("key") if isinstance(f, dict) else None for f in factors] if isinstance(factors, list) else None
    if keys != REPO_FACTORS:
        return (f"the repo's grid crosses exactly the factors {', '.join(REPO_FACTORS)} (in that order); this "
                f"study's factors are {keys}")
    for f in factors:
        levels = f.get("levels")
        if not isinstance(levels, list) or not levels:
            return f"factor {f['key']!r} has no levels"
        for lv in levels:
            if not isinstance(lv, dict) or not isinstance(lv.get("id"), str):
                return f"factor {f['key']!r} has a level without an id"
            slots = lv.get("slots") or {}
            if not isinstance(slots, dict) or sorted(slots) != sorted(LEVEL_SLOTS[f["key"]]):
                names = sorted(slots) if isinstance(slots, dict) else slots
                return (f"level {f['key']}={lv['id']} has slots {names}; the repo's files hold exactly "
                        f"{LEVEL_SLOTS[f['key']] or 'none'} for a {f['key']} level")
            if not all(isinstance(v, str) for v in slots.values()):
                return f"level {f['key']}={lv['id']} has a slot value that is not text"
            code = lv.get("code") or lv["id"]
            expected = _topic_code(lv["id"]) if f["key"] == "topic" else lv["id"]
            if code != expected:
                return (f"level {f['key']}={lv['id']} has code {code!r}; the repo's randomizer always uses "
                        f"{expected!r} in dyad ids")
    nested = spec.get("nested")
    if not isinstance(nested, dict):
        return "the repo nests role variants within ideology; this study has no nested variants"
    if nested.get("key") != "role" or nested.get("within") != "ideology":
        return (f"the repo nests 'role' within 'ideology'; this study nests {nested.get('key')!r} within "
                f"{nested.get('within')!r}")
    if not isinstance(nested.get("max_per_level"), int) or not isinstance(nested.get("variants"), dict):
        return "nested needs max_per_level and variants"
    for level, variants in nested["variants"].items():
        if not isinstance(variants, list):
            return f"nested.variants.{level} is not a list"
        for v in variants:
            slots = v.get("slots") if isinstance(v, dict) else None
            if not isinstance(slots, dict) or not isinstance(v.get("id"), str):
                return f"a role variant under {level!r} has no id or slots"
            if "slug" in slots:
                return f"role variant {v['id']!r} has a slot named 'slug', which the catalogue uses for the id"
            missing = [k for k in ROLE_FIELDS if not slots.get(k)]
            if missing:
                return (f"role variant {v['id']!r} has no {', '.join(missing)}; every catalogue role needs "
                        f"{', '.join(ROLE_FIELDS)}")
    tables = spec.get("tables")
    if not isinstance(tables, list) or len(tables) != 1 or not isinstance(tables[0], dict):
        n = len(tables) if isinstance(tables, list) else tables
        return f"the repo has exactly one table, anchors; this study has {n} tables"
    t = tables[0]
    for key, expected in (("name", "anchors"), ("by", ["ideology", "topic"]), ("join", " "),
                          ("item_slot", "anchor_{i}")):
        if t.get(key) != expected:
            return f"the repo's anchors table has {key} {expected!r}; this study's table has {t.get(key)!r}"
    derived = spec.get("derived")
    if not isinstance(derived, dict) or list(derived) != ["opening"]:
        return (f"the repo has exactly one derived slot, opening; this study's derived slots are "
                f"{list(derived) if isinstance(derived, dict) else derived}")
    used, problem = template_slots(derived["opening"])
    if problem or not set(used) <= OPENING_SLOTS:
        return (f"the repo fills the opening from {{topic}} and {{topic_phrase}} only; this study's opening uses "
                f"{', '.join('{' + u + '}' for u in used if u not in OPENING_SLOTS) or problem}")
    control = spec.get("control")
    if not isinstance(control, dict):
        return "the repo has a control cell per topic; this study has no control"
    if control.get("factor") != "ideology" or control.get("by") != ["topic"]:
        return (f"the repo's control is an ideology level by topic; this study's is factor "
                f"{control.get('factor')!r} by {control.get('by')!r}")
    if control.get("persona_mode") != CONTROL_MODE:
        return f"the repo's control always runs {CONTROL_MODE}; this study's runs {control.get('persona_mode')!r}"
    if not isinstance(control.get("level"), str):
        return "control.level is not a level id"
    templates = spec.get("templates")
    if not isinstance(templates, dict):
        return "the study has no templates"
    for path, template, allowed in (("templates.persona", templates.get("persona"), TREATED_SLOTS),
                                    ("templates.reminder", templates.get("reminder"), TREATED_SLOTS),
                                    ("control.persona", control.get("persona"), CONTROL_SLOTS),
                                    ("control.reminder", control.get("reminder"), CONTROL_SLOTS)):
        used, problem = template_slots(template)
        if problem:
            return f"{path}: {problem}"
        extra = [u for u in used if u not in allowed and not (allowed is TREATED_SLOTS and _ANCHOR_ITEM.fullmatch(u))]
        if extra:
            return (f"{path} uses {', '.join('{' + u + '}' for u in extra)}, which the repo's randomizer does not "
                    f"fill (it fills {', '.join(sorted(allowed))}{', anchor_1..' if allowed is TREATED_SLOTS else ''})")
    return None


def _build_files(spec: dict) -> tuple[dict, dict]:
    """(grid, catalogue) for a spec that passed _shape_reason: the stored grid with factors, variants_per_level
    and control.ideology from the spec; the catalogue with its keys in the file's order."""
    repo = spec["repo"]
    grid = copy.deepcopy(repo["grid"])
    levels = {f["key"]: f["levels"] for f in spec["factors"]}
    grid["factors"] = {key: [lv["id"] for lv in levels[key]] for key in REPO_FACTORS}
    grid["variants_per_level"] = spec["nested"]["max_per_level"]
    grid["control"]["ideology"] = spec["control"]["level"]
    stored = repo["catalogue"]
    mapped = {
        "version": stored["version"],
        "template": {"persona": spec["templates"]["persona"], "reminder": spec["templates"]["reminder"]},
        "shared": {"topic_phrase": {lv["id"]: lv["slots"]["topic_phrase"] for lv in levels["topic"]},
                   "opening": spec["derived"]["opening"]},
        "openness": {lv["id"]: {"text": lv["slots"]["openness_text"], "reminder": lv["slots"]["openness_reminder"]}
                     for lv in levels["openness"]},
        "roles": {level: [{"slug": v["id"], **v["slots"]} for v in variants]
                  for level, variants in spec["nested"]["variants"].items()},
        "anchors": copy.deepcopy(spec["tables"][0]["values"]),
        "control": {"persona": spec["control"]["persona"], "reminder": spec["control"]["reminder"]},
    }
    if "_note" in stored:
        mapped["_note"] = stored["_note"]
    catalogue = {k: copy.deepcopy(mapped[k]) for k in CATALOGUE_ORDER if k in mapped}
    for k, v in stored.items():
        if k not in catalogue:
            catalogue[k] = copy.deepcopy(v)
    return grid, catalogue


def to_repo_files(spec) -> tuple[dict, dict]:
    """The (grid, catalogue) the repo's randomizer would read to produce this spec's manifest. Raises
    NotRepoShaped with repo_shape_reason's sentence when the spec cannot be written as those files.
    to_repo_files(load_repo_study(root)) equals the files on disk."""
    reason = repo_shape_reason(spec)
    if reason:
        raise NotRepoShaped(reason)
    return _build_files(spec)


def build_manifest_kwargs(spec) -> dict:
    """The keyword arguments of harness.randomize.build_manifest for this spec's randomization. n_control is
    always explicit (the spec's resolved control size), so the result does not depend on the grid's defaults."""
    r = spec["randomization"]
    return {"seed": r["seed"], "n_per_cell": r["n_per_cell"], "n_control": resolved_n_control(spec),
            "modes": tuple(r["modes"]), "n_turns": r["n_turns"], "prefix": r["prefix"]}
