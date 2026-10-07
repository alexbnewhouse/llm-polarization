"""The study spec (`sandbox-study/1`, docs/superpowers/specs/2026-10-07-sandbox-gui-design.md section 3): crossed
factors, persona variants nested in one factor, lookup tables, derived slots, the persona and reminder
templates and an optional no-persona control, compiled into the manifest rows `python -m harness.run --manifest`
reads.

Pure functions over a JSON-shaped dict. `validate_study` never raises, because the GUI calls it on every edit
of a half-typed draft; everything that needs a valid spec raises StudyError (a ValueError, so the API answers
400) with the issues attached. Templates are filled by the harness's own `harness.randomize._fill`, and rows,
seeds and the shuffle follow `harness.randomize.build_manifest` step for step, so the repo study compiled here
is the repo's manifest: sandbox/tests/test_repo_study.py holds the two to the same rows."""
from __future__ import annotations
import datetime as _dt
import itertools
import json
import math
import random
import re
import string
from dataclasses import dataclass, field
from harness import __version__ as HARNESS_VERSION
from harness.grid import CONDITION_KEYS
from harness.log import now_iso, sha256_text
from harness.randomize import _fill
from harness.transcript import PERSONA_MODES

SCHEMA = "sandbox-study/1"
MAX_RENDER_ISSUES = 20            # a broken template fails in every cell; the GUI needs the first few, not all
MAX_TABLE_ISSUES = 10             # per table: a new table is missing every combination at once
TOP_KEYS = ("schema", "name", "description", "factors", "nested", "tables", "derived", "templates", "control",
            "randomization", "instrument", "run", "repo")
REQUIRED_KEYS = ("schema", "name", "factors", "templates", "randomization", "instrument", "run")
KEY_PATTERN = r"[a-z][a-z0-9_]*"   # factor and nested keys, slot names, table and derived names
ID_PATTERN = r"[A-Za-z0-9_.]+"     # level ids, codes, variant ids, the prefix: they go into dyad ids and cell keys
_KEY_RE = re.compile(KEY_PATTERN)
_ID_RE = re.compile(ID_PATTERN)
_RENDER_ERRORS = (ValueError, KeyError, IndexError, AttributeError, TypeError)


class StudyError(ValueError):
    """A study that cannot be compiled or rendered. `issues` holds validate_study-shaped issues so the API can
    return them beside the message. Built from a message, or from a list of issues (compile_manifest)."""

    def __init__(self, message, issues: list[dict] | None = None):
        if isinstance(message, list):
            issues = message
            errors = [i for i in issues if isinstance(i, dict) and i.get("level") == "error"]
            shown = "; ".join(f"{i.get('path') or '(study)'}: {i.get('message')}" for i in errors[:5])
            message = f"the study has {len(errors)} error(s): {shown}" + ("; ..." if len(errors) > 5 else "")
        super().__init__(message)
        self.issues = list(issues) if issues is not None else [{"level": "error", "path": "", "message": str(message)}]


# --- small helpers -------------------------------------------------------------------------------------------

def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_key(x) -> bool:
    return isinstance(x, str) and _KEY_RE.fullmatch(x) is not None


def _is_id(x) -> bool:
    return isinstance(x, str) and _ID_RE.fullmatch(x) is not None


def _kind(x) -> str:
    """How a JSON value reads in a message ("got a list")."""
    if x is None:
        return "null"
    for t, name in ((bool, "a boolean"), (int, "a number"), (float, "a number"), (str, "a string"),
                    (list, "a list"), (dict, "an object")):
        if isinstance(x, t):
            return name
    return type(x).__name__


def _join(base: str, key) -> str:
    return f"{base}.{key}" if base else str(key)


def template_slots(template) -> tuple[list[str], str | None]:
    """The slot names a template uses, in order, and an error message when it cannot be a template at all
    (unbalanced braces, a positional `{}` or `{0}`). `{name.attr}` and `{name[0]}` count as `name`."""
    if not isinstance(template, str):
        return [], f"a template is a string, got {_kind(template)}"
    names: list[str] = []
    try:
        for _, field_name, _, _ in string.Formatter().parse(template):
            if field_name is None:
                continue
            head = re.split(r"[.\[]", field_name, maxsplit=1)[0]
            if head == "" or head.isdigit():
                return names, "positional fields ({} or {0}) are not slots: name every slot, e.g. {topic_phrase}"
            if head not in names:
                names.append(head)
    except ValueError as e:
        return names, f"malformed template: {e}"
    return names, None


def _item_slot(pattern: str, n: int) -> str:
    return pattern.replace("{i}", str(n))


class _Issues(list):
    """The issue list validate_study returns, with one-line adders."""

    def error(self, path: str, message: str) -> None:
        self.append({"level": "error", "path": path, "message": message})

    def warn(self, path: str, message: str) -> None:
        self.append({"level": "warning", "path": path, "message": message})

    def n_errors(self) -> int:
        return sum(1 for i in self if i["level"] == "error")


# --- the design, read tolerantly -----------------------------------------------------------------------------
# One reading of the spec for every function below. It skips what is malformed instead of raising, so the
# read-only views (summary, cells, slots) work on drafts; compile_manifest only runs it on a validated spec.

@dataclass
class _Level:
    id: str
    code: str
    slots: dict
    index: int


@dataclass
class _Factor:
    key: str
    label: str
    levels: list
    index: int


@dataclass
class _Variant:
    id: str
    slots: dict


@dataclass
class _Nested:
    key: str
    within: str
    max_per_level: int | None
    variants: dict                 # level id -> [_Variant], in the spec's order


@dataclass
class _Table:
    name: str
    by: list
    join: str
    item_slot: str | None
    values: dict
    index: int


@dataclass
class _Control:
    factor: str
    level: str
    by: list                       # factor keys, in factor order
    persona: str
    reminder: str
    persona_mode: str
    n_per_cell: int | None


@dataclass
class _Design:
    name: str
    factors: list
    nested: _Nested | None
    tables: list
    derived: dict
    persona: str
    reminder: str
    control: _Control | None
    seed: int | None
    n_per_cell: int | None
    modes: list
    n_turns: int | None
    prefix: str
    cells: list | None
    repo: dict | None = None


@dataclass
class _Cell:
    key: str
    kind: str                      # "treated" | "control"
    factors: tuple                 # treated: every factor; control: the `by` factors
    levels: tuple                  # one _Level per factor above
    variants: list = field(default_factory=list)   # treated: [_Variant] or [None] when nothing is nested


def _str_dict(x) -> dict:
    return {k: v for k, v in x.items() if isinstance(k, str)} if isinstance(x, dict) else {}


def _list(x) -> list:
    return x if isinstance(x, list) else []


def _design(spec) -> _Design:
    """Read the spec into a _Design, skipping malformed parts. Never raises on JSON-shaped input."""
    s = spec if isinstance(spec, dict) else {}
    factors = []
    for fi, f in enumerate(_list(s.get("factors"))):
        if not isinstance(f, dict) or not isinstance(f.get("key"), str):
            continue
        levels = []
        for li, lv in enumerate(_list(f.get("levels"))):
            if not isinstance(lv, dict) or not isinstance(lv.get("id"), str):
                continue
            code = lv.get("code") if isinstance(lv.get("code"), str) and lv.get("code") else lv["id"]
            levels.append(_Level(lv["id"], code, _str_dict(lv.get("slots")), li))
        label = f.get("label") if isinstance(f.get("label"), str) and f.get("label") else f["key"]
        factors.append(_Factor(f["key"], label, levels, fi))
    nested = None
    n = s.get("nested")
    if isinstance(n, dict) and isinstance(n.get("key"), str) and isinstance(n.get("within"), str):
        variants = {}
        for lvl, vs in _str_dict(n.get("variants")).items():
            variants[lvl] = [_Variant(v["id"], _str_dict(v.get("slots"))) for v in _list(vs)
                             if isinstance(v, dict) and isinstance(v.get("id"), str)]
        nested = _Nested(n["key"], n["within"], n.get("max_per_level") if _is_int(n.get("max_per_level")) else None,
                         variants)
    tables = []
    for ti, t in enumerate(_list(s.get("tables"))):
        if not isinstance(t, dict) or not isinstance(t.get("name"), str):
            continue
        by = [b for b in _list(t.get("by")) if isinstance(b, str)]
        tables.append(_Table(t["name"], by, t.get("join") if isinstance(t.get("join"), str) else " ",
                             t.get("item_slot") if isinstance(t.get("item_slot"), str) else None,
                             t.get("values") if isinstance(t.get("values"), dict) else {}, ti))
    derived = {k: v for k, v in _str_dict(s.get("derived")).items() if isinstance(v, str)}
    templates = s.get("templates") if isinstance(s.get("templates"), dict) else {}
    r = s.get("randomization") if isinstance(s.get("randomization"), dict) else {}
    n_per_cell = r.get("n_per_cell") if _is_int(r.get("n_per_cell")) else None
    control = None
    c = s.get("control")
    if isinstance(c, dict) and isinstance(c.get("factor"), str) and isinstance(c.get("level"), str):
        by_keys = [b for b in _list(c.get("by")) if isinstance(b, str)]
        control = _Control(c["factor"], c["level"], [f.key for f in factors if f.key in by_keys],
                           c.get("persona") if isinstance(c.get("persona"), str) else "",
                           c.get("reminder") if isinstance(c.get("reminder"), str) else "",
                           c.get("persona_mode") if isinstance(c.get("persona_mode"), str) else "reinforced",
                           c.get("n_per_cell") if _is_int(c.get("n_per_cell")) else None)
    cells = r.get("cells")
    cells = [x for x in cells if isinstance(x, str)] if isinstance(cells, list) else None
    return _Design(
        name=s.get("name") if isinstance(s.get("name"), str) else "",
        factors=factors, nested=nested, tables=tables, derived=derived,
        persona=templates.get("persona") if isinstance(templates.get("persona"), str) else "",
        reminder=templates.get("reminder") if isinstance(templates.get("reminder"), str) else "",
        control=control,
        seed=r.get("seed") if _is_int(r.get("seed")) else None,
        n_per_cell=n_per_cell,
        modes=[m for m in _list(r.get("modes")) if isinstance(m, str)],
        n_turns=r.get("n_turns") if _is_int(r.get("n_turns")) else None,
        prefix=r.get("prefix") if isinstance(r.get("prefix"), str) else "",
        cells=cells,
        repo=s.get("repo") if isinstance(s.get("repo"), dict) else None)


def _n_control(d: _Design) -> int | None:
    if d.control is None:
        return None
    return d.control.n_per_cell if d.control.n_per_cell is not None else d.n_per_cell


def _treated_cells(d: _Design) -> list[_Cell]:
    """The cartesian product of the factors' levels in factor order (outermost first), as treated_cells and
    build_manifest walk the grid."""
    if not d.factors or any(not f.levels for f in d.factors):
        return []
    within = d.nested.within if d.nested else None
    out = []
    for combo in itertools.product(*(f.levels for f in d.factors)):
        variants: list = [None]
        if d.nested is not None:
            level = next((lv for f, lv in zip(d.factors, combo) if f.key == within), None)
            variants = list(d.nested.variants.get(level.id, [])) if level else []
        out.append(_Cell("/".join(lv.id for lv in combo), "treated", tuple(d.factors), combo, variants))
    return out


def _control_cells(d: _Design) -> list[_Cell]:
    """One control cell per combination of the `by` factors' levels, in factor order; no `by` is one cell."""
    if d.control is None:
        return []
    by = [f for f in d.factors if f.key in d.control.by]
    if any(not f.levels for f in by):
        return []
    return [_Cell("/".join([*(lv.id for lv in combo), d.control.level]), "control", tuple(by), combo, [None])
            for combo in itertools.product(*(f.levels for f in by))]


def _selected(d: _Design, key: str) -> bool:
    return d.cells is None or key in d.cells


# --- slots and rendering -------------------------------------------------------------------------------------

class _RenderError(Exception):
    """One template that did not fill: where (`path`) and why, kept apart so validation can group them."""

    def __init__(self, path: str, message: str):
        super().__init__(f"{path}: {message}")
        self.path, self.message = path, message


def _render(template: str, slots: dict, path: str) -> str:
    """harness.randomize._fill, with any failure turned into a _RenderError that names the template."""
    try:
        return _fill(template, slots)
    except _RENDER_ERRORS as e:
        message = str(e).replace(" in the catalogue", "")      # _fill's wording; the slot is what matters
        raise _RenderError(path, message) from None


def _level_slots(factors, levels) -> dict:
    slots: dict = {}
    for f, lv in zip(factors, levels):
        slots[f.key] = lv.id
        slots.update(lv.slots)
    return slots


def _table_items(t: _Table, level_ids: dict):
    """The list a table holds for one combination of its `by` levels, or None when it has none."""
    node = t.values
    for key in t.by:
        if not isinstance(node, dict) or key not in level_ids:
            return None
        node = node.get(level_ids[key])
    return node if isinstance(node, list) else None


def _treated_slots(d: _Design, cell: _Cell, variant: _Variant | None) -> dict:
    """Every slot of one treated cell and variant (spec 3.1): factor keys and level slots, the nested key and
    variant slots, each table joined and item by item, then the derived slots in order."""
    slots = _level_slots(cell.factors, cell.levels)
    level_ids = {f.key: lv.id for f, lv in zip(cell.factors, cell.levels)}
    if d.nested is not None and variant is not None:
        slots[d.nested.key] = variant.id
        slots.update(variant.slots)
    for t in d.tables:
        items = _table_items(t, level_ids)
        if items is None:
            combo = "/".join(level_ids.get(k, "?") for k in t.by)
            raise _RenderError(f"tables[{t.index}].values", f"{t.name} has no entry for {combo}")
        slots[t.name] = t.join.join(items)
        if t.item_slot:
            for n, item in enumerate(items, 1):
                slots[_item_slot(t.item_slot, n)] = item
    for name, template in d.derived.items():
        slots[name] = _render(template, slots, f"derived.{name}")
    return slots


def _control_slots(d: _Design, cell: _Cell) -> dict:
    """The slots of one control cell: the `by` factor keys, their level slots, and the derived slots those can
    fill. A derived slot that needs anything else is simply absent, as `{name}` is absent from a bare prompt."""
    slots = _level_slots(cell.factors, cell.levels)
    for name, template in d.derived.items():
        try:
            slots[name] = _fill(template, slots)
        except _RENDER_ERRORS:
            pass
    return slots


def _render_treated(d: _Design, cell: _Cell, variant: _Variant | None) -> tuple[str, str, dict]:
    slots = _treated_slots(d, cell, variant)
    return _render(d.persona, slots, "templates.persona"), _render(d.reminder, slots, "templates.reminder"), slots


def _render_control(d: _Design, cell: _Cell) -> tuple[str, str, dict]:
    slots = _control_slots(d, cell)
    return (_render(d.control.persona, slots, "control.persona"),
            _render(d.control.reminder, slots, "control.reminder"), slots)


def _where(cell: _Cell, variant: _Variant | None) -> str:
    return f"cell {cell.key}" + (f", variant {variant.id}" if variant is not None else "")


def _treated_condition(d: _Design, cell: _Cell, variant: _Variant | None) -> dict:
    condition = {f.key: lv.id for f, lv in zip(cell.factors, cell.levels)}
    if d.nested is not None:
        condition[d.nested.key] = variant.id if variant is not None else None
    return condition


def _control_condition(d: _Design, cell: _Cell) -> dict:
    """Every factor key in factor order: the `by` level, control.level for control.factor, null otherwise;
    then the nested key, null. For the repo study this is build_manifest's control condition."""
    by = {f.key: lv.id for f, lv in zip(cell.factors, cell.levels)}
    condition = {}
    for f in d.factors:
        condition[f.key] = by[f.key] if f.key in by else (d.control.level if f.key == d.control.factor else None)
    if d.nested is not None:
        condition[d.nested.key] = None
    return condition


def _treated_stem(d: _Design, cell: _Cell, variant: _Variant | None, mode: str) -> str:
    parts = [d.prefix, *(lv.code for lv in cell.levels)]
    if variant is not None:
        parts.append(variant.id)
    return "-".join([*parts, mode])


def _control_stem(d: _Design, cell: _Cell) -> str:
    return "-".join([d.prefix, *(lv.code for lv in cell.levels), "control", d.control.persona_mode])


# --- slot catalogue -------------------------------------------------------------------------------------------

def _treated_slot_entries(d: _Design) -> list[tuple[str, str, str]]:
    """(name, source, path) for every slot a treated cell defines. Factor keys come first so that a level slot
    named like a factor is the one blamed for the collision."""
    out = [(f.key, f"factor:{f.key}", f"factors[{f.index}].key") for f in d.factors]
    for f in d.factors:
        for lv in f.levels:
            out += [(name, f"level:{f.key}", f"factors[{f.index}].levels[{lv.index}].slots.{name}")
                    for name in lv.slots]
    if d.nested is not None:
        out.append((d.nested.key, f"nested:{d.nested.key}", "nested.key"))
        for lvl, vs in d.nested.variants.items():
            for k, v in enumerate(vs):
                out += [(name, f"variant:{d.nested.key}", f"nested.variants.{lvl}[{k}].slots.{name}")
                        for name in v.slots]
    for t in d.tables:
        out.append((t.name, f"table:{t.name}", f"tables[{t.index}].name"))
        if t.item_slot:
            out += [(_item_slot(t.item_slot, n), f"table:{t.name}", f"tables[{t.index}].item_slot")
                    for n in range(1, _max_items(t) + 1)]
    out += [(name, "derived", f"derived.{name}") for name in d.derived]
    return out


def _max_items(t: _Table) -> int:
    """The longest list in a table: how many item slots ({anchor_1}..) it can define."""
    def walk(node, depth):
        if depth == len(t.by):
            return len(node) if isinstance(node, list) else 0
        return max((walk(v, depth + 1) for v in node.values()), default=0) if isinstance(node, dict) else 0
    return walk(t.values, 0)


def _control_slot_entries(d: _Design) -> list[tuple[str, str]]:
    """(name, source) for the control: the `by` factor keys and level slots, then each derived slot whose
    template uses only slots the control has (decided on the template's text, for every level at once)."""
    if d.control is None:
        return []
    out: list[tuple[str, str]] = []
    for f in d.factors:
        if f.key not in d.control.by:
            continue
        out.append((f.key, f"factor:{f.key}"))
        for lv in f.levels:
            out += [(name, f"level:{f.key}") for name in lv.slots]
    available = {name for name, _ in out}
    for name, template in d.derived.items():
        used, problem = template_slots(template)
        if problem is None and set(used) <= available:
            out.append((name, "derived"))
            available.add(name)
    return out


def _unique(entries) -> list[dict]:
    seen: dict[str, str] = {}
    for entry in entries:
        seen.setdefault(entry[0], entry[1])
    return [{"name": name, "source": source} for name, source in seen.items()]


def slot_catalog(spec) -> dict:
    """The slots a template can use, for the GUI's slot chips: {"treated": [{name, source}], "control": [...]},
    grouped per factor (key, then its level slots), then nested, tables and derived. Best effort on drafts."""
    try:
        d = _design(spec)
        treated = []
        for f in d.factors:
            treated.append((f.key, f"factor:{f.key}"))
            for lv in f.levels:
                treated += [(name, f"level:{f.key}") for name in lv.slots]
        treated += [(name, source) for name, source, _ in _treated_slot_entries(d)
                    if not source.startswith(("factor:", "level:"))]
        return {"treated": _unique(treated), "control": _unique(_control_slot_entries(d))}
    except Exception:                          # a read-only view of a draft: never the reason a request fails
        return {"treated": [], "control": []}


# --- validation ----------------------------------------------------------------------------------------------

def has_errors(issues) -> bool:
    """True when any issue is an error (errors block export; warnings are shown)."""
    return any(isinstance(i, dict) and i.get("level") == "error" for i in issues or [])


def validate_study(spec) -> list[dict]:
    """Every problem with a study, as [{"level": "error" | "warning", "path", "message"}], in spec order. Never
    raises, whatever it is given: the GUI validates drafts on every keystroke. Deep checks (slot collisions,
    rendering every cell, dyad ids) run only once the design's shape is sound, so a broken shape gives a few
    clear errors instead of a cascade."""
    issues = _Issues()
    try:
        _validate(spec, issues)
    except Exception as e:                     # a bug here must not take the GUI's validation down with it
        issues.error("", f"the study could not be fully validated ({type(e).__name__}: {e})")
    return list(issues)


def _validate(spec, issues: _Issues) -> None:
    if not isinstance(spec, dict):
        issues.error("", f"a study is a JSON object, got {_kind(spec)}")
        return
    for key in spec:
        if key not in TOP_KEYS:
            issues.warn(str(key), "unknown key; ignored")
    for key in REQUIRED_KEYS:
        if key not in spec:
            issues.error(key, "required")
    if "schema" in spec and spec["schema"] != SCHEMA:
        issues.error("schema", f"must be {SCHEMA!r}, got {spec['schema']!r}")
    if "name" in spec and not (isinstance(spec["name"], str) and spec["name"].strip()):
        issues.error("name", "a non-empty string")
    if spec.get("description") is not None and not isinstance(spec.get("description"), str):
        issues.error("description", f"a string, got {_kind(spec['description'])}")
    if spec.get("repo") is not None and not isinstance(spec.get("repo"), dict):
        issues.error("repo", f"null or an object (set by the repo study), got {_kind(spec['repo'])}")

    before = issues.n_errors()
    levels = _check_factors(spec.get("factors"), "factors" in spec, issues)     # factor key -> [level ids], or None
    _check_nested(spec.get("nested"), levels, issues)
    _check_tables(spec.get("tables"), levels, issues)
    _check_derived(spec.get("derived"), issues)
    _check_templates(spec.get("templates"), "templates" in spec, issues)
    _check_control(spec.get("control"), levels, issues)
    _check_randomization(spec.get("randomization"), "randomization" in spec, issues)
    design_ok = issues.n_errors() == before and levels is not None
    if "instrument" in spec:
        _check_instrument(spec["instrument"], issues)
    if "run" in spec:
        _check_run(spec["run"], issues)
    d = _design(spec)
    if design_ok:
        _check_design(d, issues)
    _check_condition_keys(d, issues)


def _check_slot_dict(slots, path: str, issues: _Issues, missing_ok: bool = True) -> bool:
    if slots is None and missing_ok:
        return True
    if not isinstance(slots, dict):
        issues.error(path, f"an object of slot name -> text, got {_kind(slots)}")
        return False
    ok = True
    for name, value in slots.items():
        if not _is_key(name):
            issues.error(_join(path, name), f"slot names match {KEY_PATTERN} (lower case, digits, underscores)")
            ok = False
        if not isinstance(value, str):
            issues.error(_join(path, name), f"a slot value is text, got {_kind(value)}")
            ok = False
    return ok


def _warn_uneven_slots(sets: list[tuple[str, set]], owner: str, issues: _Issues) -> None:
    """A slot defined on some levels (or variants) but not others fails only in the cells that lack it; say so
    once, on the one that lacks it, before the render errors do."""
    union = set().union(*(s for _, s in sets)) if sets else set()
    for path, s in sets:
        missing = sorted(union - s)
        if missing:
            issues.warn(path, f"has no {', '.join(missing)}, which other {owner} define; a template that uses "
                              f"{'it' if len(missing) == 1 else 'them'} fails here")


def _check_factors(factors, present: bool, issues: _Issues) -> dict | None:
    """Shape, syntax and uniqueness of the factors; returns {factor key: [level ids]} when sound enough for
    the sections that refer to factors, else None."""
    if not present:
        return None
    if not isinstance(factors, list) or not factors:
        got = "an empty list" if factors == [] else _kind(factors)
        issues.error("factors", f"a non-empty list of factors, got {got}")
        return None
    before = issues.n_errors()
    out: dict = {}
    for i, f in enumerate(factors):
        p = f"factors[{i}]"
        if not isinstance(f, dict):
            issues.error(p, f"a factor is an object with key and levels, got {_kind(f)}")
            continue
        key = f.get("key")
        if not _is_key(key):
            issues.error(f"{p}.key", f"factor keys match {KEY_PATTERN}, got {key!r}")
            key = None
        elif key in out:
            issues.error(f"{p}.key", f"duplicate factor key {key!r}")
            key = None
        if f.get("label") is not None and not isinstance(f.get("label"), str):
            issues.error(f"{p}.label", f"a string, got {_kind(f['label'])}")
        levels = f.get("levels")
        if not isinstance(levels, list) or not levels:
            issues.error(f"{p}.levels", "a non-empty list of levels")
            continue
        ids: dict = {}
        codes: dict = {}
        slot_sets = []
        for j, lv in enumerate(levels):
            lp = f"{p}.levels[{j}]"
            if not isinstance(lv, dict):
                issues.error(lp, f"a level is an object with id and slots, got {_kind(lv)}")
                continue
            lid = lv.get("id")
            if not _is_id(lid):
                issues.error(f"{lp}.id", f"level ids match {ID_PATTERN} (they go into cell keys and dyad ids), "
                                         f"got {lid!r}")
                lid = None
            elif lid in ids:
                issues.error(f"{lp}.id", f"duplicate level id {lid!r} (also levels[{ids[lid]}])")
                lid = None
            else:
                ids[lid] = j
            code = lv.get("code")
            if code is not None and not _is_id(code):
                issues.error(f"{lp}.code", f"codes match {ID_PATTERN}, got {code!r}")
            effective = code if _is_id(code) else lid
            if effective is not None:
                if effective in codes:
                    issues.error(f"{lp}.code" if code is not None else f"{lp}.id",
                                 f"dyad id code {effective!r} is also the code of levels[{codes[effective]}]; "
                                 f"their dyad ids would collide")
                else:
                    codes[effective] = j
            if _check_slot_dict(lv.get("slots"), f"{lp}.slots", issues) and isinstance(lv.get("slots"), dict):
                slot_sets.append((f"{lp}.slots", set(lv["slots"])))
            elif lv.get("slots") is None:
                slot_sets.append((f"{lp}.slots", set()))
        if key is not None:
            _warn_uneven_slots(slot_sets, f"levels of {key!r}", issues)
            out[key] = list(ids)
    return out if issues.n_errors() == before else None


def _check_nested(nested, levels: dict | None, issues: _Issues) -> None:
    if nested is None:
        return
    if not isinstance(nested, dict):
        issues.error("nested", f"null or an object with key, within, max_per_level and variants, got {_kind(nested)}")
        return
    key, within, cap = nested.get("key"), nested.get("within"), nested.get("max_per_level")
    if not _is_key(key):
        issues.error("nested.key", f"the nested key matches {KEY_PATTERN}, got {key!r}")
    if not isinstance(within, str):
        issues.error("nested.within", f"the key of the factor the variants are nested in, got {_kind(within)}")
        within = None
    elif levels is not None and within not in levels:
        issues.error("nested.within", f"{within!r} is not a factor (factors: {', '.join(levels)})")
        within = None
    if not _is_int(cap) or cap < 1:
        issues.error("nested.max_per_level", f"a whole number >= 1, got {cap!r}")
        cap = None
    variants = nested.get("variants")
    if not isinstance(variants, dict):
        issues.error("nested.variants", f"an object of level id -> list of variants, got {_kind(variants)}")
        return
    parent_levels = levels.get(within) if (levels is not None and within is not None) else None
    seen: dict = {}
    slot_sets = []
    for lvl, vs in variants.items():
        vp = f"nested.variants.{lvl}"
        if parent_levels is not None and lvl not in parent_levels:
            issues.error(vp, f"{lvl!r} is not a level of {within!r}; variants are registered only under its levels")
        if not isinstance(vs, list):
            issues.error(vp, f"a list of variants, got {_kind(vs)}")
            continue
        if not vs:
            issues.error(vp, f"no variants: every level of {within!r} needs 1 to {cap or 'max_per_level'}")
        elif cap is not None and len(vs) > cap:
            issues.error(vp, f"{len(vs)} variants; max_per_level is {cap}")
        for k, v in enumerate(vs):
            p = f"{vp}[{k}]"
            if not isinstance(v, dict):
                issues.error(p, f"a variant is an object with id and slots, got {_kind(v)}")
                continue
            vid = v.get("id")
            if not _is_id(vid):
                issues.error(f"{p}.id", f"variant ids match {ID_PATTERN}, got {vid!r}")
            elif vid in seen:
                issues.error(f"{p}.id", f"duplicate variant id {vid!r} (also {seen[vid]}); variant ids are unique "
                                        f"across the study")
            else:
                seen[vid] = p
            if _check_slot_dict(v.get("slots"), f"{p}.slots", issues):
                slot_sets.append((f"{p}.slots", set(v.get("slots") or {})))
    if parent_levels is not None:
        for lvl in parent_levels:
            if lvl not in variants:
                issues.error(f"nested.variants.{lvl}", f"level {lvl!r} of {within!r} has no variants; it needs "
                                                       f"1 to {cap or 'max_per_level'}")
    _warn_uneven_slots(slot_sets, "variants", issues)


def _check_tables(tables, levels: dict | None, issues: _Issues) -> None:
    if tables is None:
        return
    if not isinstance(tables, list):
        issues.error("tables", f"a list of tables, got {_kind(tables)}")
        return
    names: dict = {}
    for i, t in enumerate(tables):
        p = f"tables[{i}]"
        if not isinstance(t, dict):
            issues.error(p, f"a table is an object with name, by, join, item_slot and values, got {_kind(t)}")
            continue
        name = t.get("name")
        if not _is_key(name):
            issues.error(f"{p}.name", f"table names are slot names and match {KEY_PATTERN}, got {name!r}")
        elif name in names:
            issues.error(f"{p}.name", f"duplicate table name {name!r} (also tables[{names[name]}])")
        else:
            names[name] = i
        by = t.get("by")
        by_ok = isinstance(by, list) and bool(by)
        if not by_ok:
            issues.error(f"{p}.by", "a non-empty list of factor keys the table is looked up by")
        else:
            for j, b in enumerate(by):
                if not isinstance(b, str) or (levels is not None and b not in levels):
                    issues.error(f"{p}.by[{j}]", f"{b!r} is not a factor")
                    by_ok = False
                elif b in by[:j]:
                    issues.error(f"{p}.by[{j}]", f"{b!r} is listed twice")
                    by_ok = False
        if not isinstance(t.get("join"), str):
            issues.error(f"{p}.join", f"the text between items, e.g. \" \", got {_kind(t.get('join'))}")
        item_slot = t.get("item_slot")
        if item_slot is not None and not (isinstance(item_slot, str) and "{i}" in item_slot
                                          and _is_key(_item_slot(item_slot, 1))):
            issues.error(f"{p}.item_slot", f"null, or a slot name with {{i}} for the item number, e.g. "
                                           f"\"anchor_{{i}}\"; got {item_slot!r}")
        values = t.get("values")
        if not isinstance(values, dict):
            issues.error(f"{p}.values", f"an object nested by the `by` levels, got {_kind(values)}")
            continue
        if by_ok and levels is not None:
            _check_table_values(values, by, levels, p, issues)


def _check_table_values(values: dict, by: list, levels: dict, p: str, issues: _Issues) -> None:
    """Every combination of the `by` levels holds a non-empty list of non-empty strings; keys that are not
    levels are warned about (they are never looked up)."""
    problems = 0
    for combo in itertools.product(*(levels[b] for b in by)):
        node, path = values, f"{p}.values"
        for lvl in combo:
            node = node.get(lvl) if isinstance(node, dict) else None
            path = f"{path}.{lvl}"
        message = None
        if node is None:
            message = f"missing: every combination of {', '.join(by)} needs a list of items"
        elif not isinstance(node, list) or not node:
            message = "a non-empty list of items"
        if message:
            problems += 1
            if problems <= MAX_TABLE_ISSUES:
                issues.error(path, message)
            continue
        for k, item in enumerate(node):
            if not isinstance(item, str) or not item.strip():
                problems += 1
                if problems <= MAX_TABLE_ISSUES:
                    issues.error(f"{path}[{k}]", f"an item is non-empty text, got {item!r}")
    if problems > MAX_TABLE_ISSUES:
        issues.error(f"{p}.values", f"and {problems - MAX_TABLE_ISSUES} more missing or empty entries")

    def extra(node, depth, path):
        if depth == len(by) or not isinstance(node, dict):
            return
        for k, v in node.items():
            if k not in levels[by[depth]]:
                issues.warn(f"{path}.{k}", f"{k!r} is not a level of {by[depth]!r}; never looked up")
            else:
                extra(v, depth + 1, f"{path}.{k}")
    extra(values, 0, f"{p}.values")


def _check_derived(derived, issues: _Issues) -> None:
    if derived is None:
        return
    if not isinstance(derived, dict):
        issues.error("derived", f"an object of slot name -> template, got {_kind(derived)}")
        return
    for name, template in derived.items():
        if not _is_key(name):
            issues.error(f"derived.{name}", f"derived slot names match {KEY_PATTERN}")
        if not isinstance(template, str):
            issues.error(f"derived.{name}", f"a template string, got {_kind(template)}")


def _check_templates(templates, present: bool, issues: _Issues) -> None:
    if not present:
        return
    if not isinstance(templates, dict):
        issues.error("templates", f"an object with persona and reminder, got {_kind(templates)}")
        return
    for key in ("persona", "reminder"):
        if key not in templates:
            issues.error(f"templates.{key}", "required (the reminder may be empty when no mode is reinforced)")
        elif not isinstance(templates[key], str):
            issues.error(f"templates.{key}", f"a template string, got {_kind(templates[key])}")


def _check_control(control, levels: dict | None, issues: _Issues) -> None:
    if control is None:
        return
    if not isinstance(control, dict):
        issues.error("control", f"null or an object with factor, level, by, persona, reminder, persona_mode and "
                                f"n_per_cell, got {_kind(control)}")
        return
    factor = control.get("factor")
    if not isinstance(factor, str) or (levels is not None and factor not in levels):
        issues.error("control.factor", f"the factor whose level marks the control, got {factor!r}")
        factor = None
    level = control.get("level")
    if not _is_id(level):
        issues.error("control.level", f"the control's level id matches {ID_PATTERN}, got {level!r}")
    elif levels is not None and factor is not None and level in levels[factor]:
        issues.error("control.level", f"{level!r} is a level of {factor!r}; the control's level must not be one")
    by = control.get("by")
    if not isinstance(by, list):
        issues.error("control.by", f"a list of factor keys (one control cell per combination), got {_kind(by)}")
    else:
        for j, b in enumerate(by):
            if not isinstance(b, str) or (levels is not None and b not in levels):
                issues.error(f"control.by[{j}]", f"{b!r} is not a factor")
            elif b == factor:
                issues.error(f"control.by[{j}]", f"{b!r} is the control factor; `by` excludes it")
            elif b in by[:j]:
                issues.error(f"control.by[{j}]", f"{b!r} is listed twice")
    for key in ("persona", "reminder"):
        if not isinstance(control.get(key), str):
            issues.error(f"control.{key}", f"a template string, got {_kind(control.get(key))}")
    if control.get("persona_mode") not in PERSONA_MODES:
        issues.error("control.persona_mode", f"one of {', '.join(PERSONA_MODES)}, got {control.get('persona_mode')!r}")
    n = control.get("n_per_cell")
    if n is not None and (not _is_int(n) or n < 1):
        issues.error("control.n_per_cell", f"null (same as randomization.n_per_cell) or a whole number >= 1, got {n!r}")


def _check_randomization(r, present: bool, issues: _Issues) -> None:
    if not present:
        return
    if not isinstance(r, dict):
        issues.error("randomization", f"an object with seed, n_per_cell, modes, n_turns, prefix and cells, "
                                      f"got {_kind(r)}")
        return
    if not _is_int(r.get("seed")):
        issues.error("randomization.seed", f"a whole number (it seeds every per-dyad seed and the shuffle), "
                                           f"got {r.get('seed')!r}")
    for key in ("n_per_cell", "n_turns"):
        if not _is_int(r.get(key)) or r[key] < 1:
            issues.error(f"randomization.{key}", f"a whole number >= 1, got {r.get(key)!r}")
    modes = r.get("modes")
    if not isinstance(modes, list) or not modes:
        issues.error("randomization.modes", f"a non-empty list of {', '.join(PERSONA_MODES)}")
    else:
        for j, m in enumerate(modes):
            if m not in PERSONA_MODES:
                issues.error(f"randomization.modes[{j}]", f"{m!r} is not one of {', '.join(PERSONA_MODES)}")
            elif m in modes[:j]:
                issues.error(f"randomization.modes[{j}]", f"{m!r} is listed twice")
    if not _is_id(r.get("prefix")):
        issues.error("randomization.prefix", f"the dyad id prefix matches {ID_PATTERN}, got {r.get('prefix')!r}")
    cells = r.get("cells")
    if cells is not None:
        if not isinstance(cells, list):
            issues.error("randomization.cells", f"null (every cell) or a list of cell keys, got {_kind(cells)}")
        elif not cells:
            issues.error("randomization.cells", "selects no cells; null means every cell")
        else:
            for j, c in enumerate(cells):
                if not isinstance(c, str):
                    issues.error(f"randomization.cells[{j}]", f"a cell key, got {_kind(c)}")


def _check_instrument(instrument, issues: _Issues) -> None:
    """The rules of harness.survey.load_batteries, with a path on each, plus integer scales with min <= max
    (the answer schema is an integer between them)."""
    if not isinstance(instrument, dict):
        issues.error("instrument", f"an object shaped like instruments/batteries.json, got {_kind(instrument)}")
        return
    items = instrument.get("items")
    if not isinstance(items, list):
        issues.error("instrument.items", f"a list of survey items, got {_kind(items)}" if "items" in instrument
                     else "required")
        return
    if not items:
        issues.warn("instrument.items", "no survey items: the run administers no pre/post survey")
    seen: dict = {}
    for i, it in enumerate(items):
        p = f"instrument.items[{i}]"
        if not isinstance(it, dict):
            issues.error(p, f"an item is an object with id, battery, scale and text, got {_kind(it)}")
            continue
        for key in ("id", "battery", "text"):
            if key not in it:
                issues.error(f"{p}.{key}", "required")
            elif not isinstance(it[key], str) or not it[key].strip():
                issues.error(f"{p}.{key}", f"non-empty text, got {it[key]!r}")
        if isinstance(it.get("id"), str):
            if it["id"] in seen:
                issues.error(f"{p}.id", f"duplicate survey item id {it['id']!r} (also items[{seen[it['id']]}])")
            else:
                seen[it["id"]] = i
        scale = it.get("scale")
        if "scale" not in it:
            issues.error(f"{p}.scale", "required: {\"min\": .., \"max\": ..}")
        elif not isinstance(scale, dict) or "min" not in scale or "max" not in scale:
            issues.error(f"{p}.scale", "needs scale.min and scale.max")
        elif not _is_int(scale["min"]) or not _is_int(scale["max"]):
            issues.error(f"{p}.scale", f"min and max are whole numbers, got {scale['min']!r} and {scale['max']!r}")
        elif scale["min"] > scale["max"]:
            issues.error(f"{p}.scale", f"min {scale['min']} is above max {scale['max']}")


def _is_number(x) -> bool:
    """A finite JSON number that is not a boolean (Python's bool is an int; the harness would take true as 1)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    return not isinstance(x, float) or math.isfinite(x)


def _parses_as_day(x) -> bool:
    """What harness/templates.py does with `now`: datetime.strptime(now, "%Y-%m-%d")."""
    try:
        _dt.datetime.strptime(x, "%Y-%m-%d")
    except (TypeError, ValueError):
        return False
    return True


# The run block's fields that have a type to check: (path under run, check, what it must be). A key left out
# takes the harness's default (harness/run.py DEFAULT_CONFIG, deep-merged), so only a key that is present is
# checked -- and a present null is not left out: the merge keeps it, and the run fails on it, often mid-wave.
_RUN_FIELDS = (
    ("generation.temperature", lambda x: _is_number(x) and x >= 0, "a number >= 0"),
    ("generation.top_p", lambda x: _is_number(x) and 0 < x <= 1, "a number in (0, 1]"),
    ("generation.n_predict", lambda x: _is_int(x) and x >= 1, "a whole number >= 1"),
    ("generation.timeout", lambda x: _is_number(x) and x > 0, "a number of seconds > 0"),
    ("generation.enable_thinking", lambda x: isinstance(x, bool), "true or false"),
    ("run_seed", _is_int, "a whole number"),
    ("now", _parses_as_day, "a date YYYY-MM-DD (the chat templates' strftime_now; harness/templates.py)"),
    ("concurrency", lambda x: x is None or (_is_int(x) and x >= 1), "null (one dyad per server slot) or a whole "
                                                                    "number >= 1"),
    ("cache_reuse_limit", lambda x: x is None or (_is_int(x) and x >= 0), "null (no limit) or a whole number >= 0"),
    ("gguf_py_path", lambda x: x is None or (isinstance(x, str) and x.strip() != ""), "null or a path"),
    ("data_dir", lambda x: isinstance(x, str) and x.strip() != "", "a path"),
)


def _check_run(run, issues: _Issues) -> None:
    if not isinstance(run, dict):
        issues.error("run", f"a harness config without batteries and grid, got {_kind(run)}")
        return
    for key in ("batteries", "grid"):
        if key in run:
            issues.warn(f"run.{key}", "ignored: the export writes its own")
    if "generation" in run and not isinstance(run["generation"], dict):
        issues.error("run.generation", f"an object of sampling settings, got {_kind(run['generation'])}")
    for path, ok, want in _RUN_FIELDS:
        block, key = (run.get("generation"), path.split(".", 1)[1]) if "." in path else (run, path)
        if isinstance(block, dict) and key in block and not ok(block[key]):
            value = block[key]
            issues.error(f"run.{path}", f"{want}, got {value!r}" if isinstance(value, (str, int, float))
                         and not isinstance(value, bool) else f"{want}, got {_kind(value)}")
    urls = {}
    for role in ("seeker", "mentor", "judge"):
        if role not in run:
            if role != "judge":
                issues.warn(f"run.{role}.url", "no URL: check and run refuse a config without one")
            continue
        entry = run[role]
        if not isinstance(entry, dict):
            issues.error(f"run.{role}", f"an object with url and gguf_path, got {_kind(entry)}")
            continue
        gguf = entry.get("gguf_path")
        if gguf is not None and not isinstance(gguf, str):
            issues.error(f"run.{role}.gguf_path", f"null (read from the server's /props) or a path, got {_kind(gguf)}")
        url = entry.get("url")
        if url is not None and not isinstance(url, str):
            issues.error(f"run.{role}.url", f"a URL string, got {_kind(url)}")
        elif not url and role != "judge":
            issues.warn(f"run.{role}.url", "no URL: check and run refuse a config without one")
        else:
            urls[role] = url
    if urls.get("seeker") and urls.get("seeker") == urls.get("mentor"):
        issues.warn("run.mentor.url", "the same as seeker.url: both agents would pin one slot, and run refuses it")


def _check_condition_keys(d: _Design, issues: _Issues) -> None:
    keys = [f.key for f in d.factors] + ([d.nested.key] if d.nested else [])
    other = [k for k in keys if k not in CONDITION_KEYS]
    if other:
        issues.warn("factors", f"condition keys {', '.join(other)} are not the harness grid's "
                               f"({', '.join(CONDITION_KEYS)}): the export sets config.grid = null, so the harness "
                               f"grid gate is off and the conditions are the ones validated here")
    if d.factors and "topic" not in [f.key for f in d.factors]:
        issues.warn("factors", "no factor named 'topic': the judge prompt reads condition.topic")


def _check_design(d: _Design, issues: _Issues) -> None:
    """The cross checks of a structurally sound design: slot collisions, n_per_cell against the variants, cell
    keys and the cells subset, dyad ids, and rendering every treated and control cell."""
    before = issues.n_errors()
    entries = _treated_slot_entries(d)
    first: dict = {}
    for name, source, path in entries:
        if name not in first:
            first[name] = (source, path)
        elif first[name][0] != source:
            issues.error(path, f"slot {{{name}}} is already defined by {first[name][0]} ({first[name][1]}); slot "
                               f"names must not collide across sources")
    collisions = issues.n_errors() > before

    if d.nested is not None:
        for lvl, vs in d.nested.variants.items():
            if vs and d.n_per_cell % len(vs):
                issues.error("randomization.n_per_cell", f"{d.n_per_cell} does not split evenly across the {len(vs)} "
                                                         f"variants of {d.nested.within}={lvl}")
    treated, controls = _treated_cells(d), _control_cells(d)
    treated_keys = {c.key for c in treated}
    for c in controls:
        if c.key in treated_keys:
            issues.error("control.level", f"control cell key {c.key!r} is also a treated cell key; choose another "
                                          f"control.level")
            break
    all_keys = treated_keys | {c.key for c in controls}
    if d.cells is not None:
        for j, key in enumerate(d.cells):
            if key not in all_keys:
                issues.error(f"randomization.cells[{j}]", f"{key!r} is not a cell key of this study")
            elif key in d.cells[:j]:
                issues.warn(f"randomization.cells[{j}]", f"{key!r} is listed twice")
    stems: dict = {}
    for c in treated:
        for mode in d.modes:
            for v in c.variants:
                stems.setdefault(_treated_stem(d, c, v, mode), c.key)
    for c in controls:
        stem = _control_stem(d, c)
        if stem in stems:
            issues.error("control", f"control dyad ids {stem}-001.. collide with the treated ids of {stems[stem]}; "
                                    f"rename the level code or variant that reads 'control'")
            break
    _unused_slot_warnings(d, issues)
    if not collisions:                 # with two values for one slot name, what a template gets is arbitrary
        _check_rendering(d, treated, controls, issues)


def _unused_slot_warnings(d: _Design, issues: _Issues) -> None:
    """Level and variant slots, tables and derived slots that no template uses: usually a typo in a template."""
    used: set = set()
    templates = [d.persona, d.reminder, *d.derived.values()]
    if d.control is not None:
        templates += [d.control.persona, d.control.reminder]
    for t in templates:
        used.update(template_slots(t)[0])
    reported: set = set()
    for name, source, path in _treated_slot_entries(d):
        if source.startswith(("factor:", "nested:")) or name in used or name in reported:
            continue
        if source.startswith("table:"):               # one warning per table, for its name and items together
            t = next(t for t in d.tables if t.name == source[len("table:"):])
            if t.name in reported or t.name in used or any(
                    _item_slot(t.item_slot, n) in used for n in range(1, _max_items(t) + 1) if t.item_slot):
                continue
            path = f"tables[{t.index}].name"
            name = t.name
        reported.add(name)
        issues.warn(path, f"slot {{{name}}} is not used by any template")


def _check_rendering(d: _Design, treated: list, controls: list, issues: _Issues) -> None:
    """Every template against the slots it can see: first by name (one error per unknown slot, wherever it is
    used), then, when every name resolves, by filling every treated cell and variant and every control cell,
    grouping identical failures. At most MAX_RENDER_ISSUES are reported, then a count."""
    found: list[tuple[str, str]] = []
    available = {name for name, source, _ in _treated_slot_entries(d) if source != "derived"}
    broken: set = set()                # derived slots already reported: not blamed again where they are used
    for name, template in d.derived.items():
        used, problem = template_slots(template)
        if problem:
            found.append((f"derived.{name}", problem))
        found += [(f"derived.{name}", f"template slot {{{u}}} is not defined by any factor, level, variant, table "
                                      f"or earlier derived slot") for u in used if u not in available]
        if problem or not set(used) <= available:
            broken.add(name)
        available.add(name)
    for path, template in (("templates.persona", d.persona), ("templates.reminder", d.reminder)):
        used, problem = template_slots(template)
        if problem:
            found.append((path, problem))
        found += [(path, f"template slot {{{u}}} is not defined by any factor, level, variant, table or derived "
                         f"slot") for u in used if u not in available]
    if d.control is not None:
        control_available = {name for name, _ in _control_slot_entries(d)}
        for path, template in (("control.persona", d.control.persona), ("control.reminder", d.control.reminder)):
            used, problem = template_slots(template)
            if problem:
                found.append((path, problem))
            found += [(path, f"template slot {{{u}}} is not available in the control, which has only "
                             f"{', '.join(sorted(control_available)) or 'no slots'}") for u in used
                      if u not in control_available and u not in broken]
    warnings: list[tuple[str, str]] = []
    if not found:
        found, warnings = _render_every_cell(d, treated, controls)
    for path, message in found[:MAX_RENDER_ISSUES]:
        issues.error(path, message)
    if len(found) > MAX_RENDER_ISSUES:
        issues.error("templates", f"and {len(found) - MAX_RENDER_ISSUES} more template problems not shown")
    for path, message in warnings:
        issues.warn(path, message)


def _render_every_cell(d: _Design, treated: list, controls: list) -> tuple[list, list]:
    """Fill every treated cell and variant and every control cell; returns (errors, warnings) as (path,
    message) pairs, identical failures grouped with a count and the first cell they occurred in."""
    groups: dict = {}                  # (path, message) -> [count, first where]
    totals = {"treated": 0, "control": 0}

    def note(path, message, where):
        g = groups.setdefault((path, message), [0, where, "control" if path.startswith("control.") else "treated"])
        g[0] += 1

    empty_reminder = empty_persona = 0
    for c in treated:
        for v in c.variants:
            totals["treated"] += 1
            try:
                text, reminder, _ = _render_treated(d, c, v)
            except _RenderError as e:
                note(e.path, e.message, _where(c, v))
                continue
            empty_persona += not text
            empty_reminder += not reminder
    control_empty_reminder = control_empty_persona = 0
    for c in controls:
        totals["control"] += 1
        try:
            text, reminder, _ = _render_control(d, c)
        except _RenderError as e:
            note(e.path, e.message, _where(c, None))
            continue
        control_empty_persona += not text
        control_empty_reminder += not reminder
    found = [(path, f"{message} ({n} of {totals[kind]} {kind} renders, e.g. {where})")
             for (path, message), (n, where, kind) in groups.items()]
    if empty_reminder and "reinforced" in d.modes:
        found.append(("templates.reminder", f"renders empty in {empty_reminder} of {totals['treated']} treated "
                                            f"renders, but mode 'reinforced' appends the reminder every turn"))
    if d.control is not None and control_empty_reminder and d.control.persona_mode == "reinforced":
        found.append(("control.reminder", f"renders empty in {control_empty_reminder} control cells, but the "
                                          f"control's mode is 'reinforced'"))
    # An empty system prompt is legal for the harness but almost never meant: shown, not blocking.
    warnings = []
    if empty_persona:
        warnings.append(("templates.persona", f"renders empty in {empty_persona} treated renders"))
    if control_empty_persona:
        warnings.append(("control.persona", f"renders empty in {control_empty_persona} control cells"))
    return found, warnings


# --- read-only views ------------------------------------------------------------------------------------------

def condition_keys(spec) -> list[str]:
    """The keys of every row's `condition`: the factor keys in order, then the nested key."""
    d = _design(spec)
    return [f.key for f in d.factors] + ([d.nested.key] if d.nested else [])


def resolved_n_control(spec) -> int | None:
    """Rows per control cell: control.n_per_cell, or randomization.n_per_cell when that is null; None when the
    study has no control."""
    return _n_control(_design(spec))


def enumerate_cells(spec) -> list[dict]:
    """Every cell, treated (factor-product order) then control: {"key", "kind", "condition", "variants",
    "n_rows", "selected"}. A control cell's condition holds only its `by` keys. Best effort on drafts."""
    try:
        d = _design(spec)
        n_treated = (d.n_per_cell or 0) * len(d.modes)
        out = [{"key": c.key, "kind": "treated", "condition": {f.key: lv.id for f, lv in zip(c.factors, c.levels)},
                "variants": [v.id for v in c.variants if v is not None], "n_rows": n_treated,
                "selected": _selected(d, c.key)} for c in _treated_cells(d)]
        out += [{"key": c.key, "kind": "control", "condition": {f.key: lv.id for f, lv in zip(c.factors, c.levels)},
                 "variants": [], "n_rows": _n_control(d) or 0, "selected": _selected(d, c.key)}
                for c in _control_cells(d)]
        return out
    except Exception:                          # a read-only view of a draft: never the reason a request fails
        return []


def cell_key(spec, condition) -> str | None:
    """The cell key of a manifest row's `condition` (spec 3.2): a treated row's level ids in factor order joined
    with "/", a control row's `by` level ids and control.level. How a `cells` subset is applied to rows that
    were not built here (sandbox/export.py filters harness.randomize's rows with it). None when the condition
    lacks a level id this study's cells are keyed by; whether the key is a cell of the study is not checked."""
    return _cell_key(_design(spec), condition)


def _cell_key(d: _Design, condition) -> str | None:
    if not isinstance(condition, dict):
        return None
    c = d.control
    if c is not None and condition.get(c.factor) == c.level:
        parts = [*(condition.get(f.key) for f in d.factors if f.key in c.by), c.level]
    else:
        parts = [condition.get(f.key) for f in d.factors]
    if not d.factors or not all(isinstance(p, str) for p in parts):
        return None
    return "/".join(parts)


def summarize(spec) -> dict:
    """The study's size for the summary cards: factors, nested variants, cells, rows (of the selected cells:
    what an export writes; `rows_design` is the full design), rows per mode, turns and messages (two per turn).
    Best effort on drafts."""
    out = {"factors": [], "nested": None, "cells_treated": 0, "cells_control": 0, "cells_selected": 0, "rows": 0,
           "rows_design": 0, "rows_per_mode": {}, "n_turns": None, "messages": 0, "condition_keys": []}
    try:
        d = _design(spec)
        out["factors"] = [{"key": f.key, "label": f.label, "n_levels": len(f.levels)} for f in d.factors]
        if d.nested is not None:
            out["nested"] = {"key": d.nested.key, "within": d.nested.within,
                             "n_variants": {lvl: len(vs) for lvl, vs in d.nested.variants.items()}}
        out["condition_keys"] = condition_keys(spec)
        out["n_turns"] = d.n_turns
        treated, controls = _treated_cells(d), _control_cells(d)
        out["cells_treated"], out["cells_control"] = len(treated), len(controls)
        out["cells_selected"] = sum(_selected(d, c.key) for c in treated + controls)
        per_mode: dict = {}
        n = d.n_per_cell or 0
        n_control = _n_control(d) or 0
        selected_treated = sum(_selected(d, c.key) for c in treated)
        for mode in d.modes:
            per_mode[mode] = per_mode.get(mode, 0) + n * selected_treated
        selected_control = sum(_selected(d, c.key) for c in controls)
        if d.control is not None and selected_control:
            mode = d.control.persona_mode
            per_mode[mode] = per_mode.get(mode, 0) + n_control * selected_control
        out["rows_per_mode"] = {m: k for m, k in per_mode.items() if k}
        out["rows"] = sum(out["rows_per_mode"].values())
        out["rows_design"] = n * len(d.modes) * len(treated) + n_control * len(controls)
        out["messages"] = out["rows"] * (d.n_turns or 0) * 2
    except Exception:                          # a read-only view of a draft: never the reason a request fails
        pass
    return out


def render_cell(spec, kind: str, condition: dict, variant: str | None = None) -> dict:
    """The persona and reminder of one cell, for the GUI's live preview: {"persona_text", "persona_reminder",
    "slots"}. A treated cell takes one level per factor from `condition` and the variant named by `variant`, or
    by condition[<nested key>], or the level's first. A control cell reads only its `by` keys, so a full
    manifest condition (with nulls) works. Raises StudyError naming the level, variant or missing slot."""
    d = _design(spec)
    if not isinstance(condition, dict):
        raise StudyError(f"condition is an object of factor key -> level id, got {_kind(condition)}")
    if kind == "treated":
        cell = _cell_for(d, d.factors, condition, "treated")
        v = None
        if d.nested is not None:
            vid = variant if variant is not None else condition.get(d.nested.key)
            options = cell.variants
            if not options:
                raise StudyError(f"no {d.nested.key} variants for {cell.key}", _render_issue("nested.variants", ""))
            v = options[0] if vid is None else next((o for o in options if o.id == vid), None)
            if v is None:
                raise StudyError(f"variant {vid!r} is not a {d.nested.key} of {cell.key} "
                                 f"(variants: {', '.join(o.id for o in options)})")
        try:
            text, reminder, slots = _render_treated(d, cell, v)
        except _RenderError as e:
            raise StudyError(f"{e.path}: {e.message} ({_where(cell, v)})", _render_issue(e.path, e.message)) from None
    elif kind == "control":
        if d.control is None:
            raise StudyError("the study has no control", _render_issue("control", "the study has no control"))
        cell = _cell_for(d, [f for f in d.factors if f.key in d.control.by], condition, "control")
        try:
            text, reminder, slots = _render_control(d, cell)
        except _RenderError as e:
            raise StudyError(f"{e.path}: {e.message} ({_where(cell, None)})",
                             _render_issue(e.path, e.message)) from None
    else:
        raise StudyError(f"kind is 'treated' or 'control', got {kind!r}")
    return {"persona_text": text, "persona_reminder": reminder, "slots": slots}


def _render_issue(path: str, message: str) -> list[dict]:
    return [{"level": "error", "path": path, "message": message}]


def _cell_for(d: _Design, factors: list, condition: dict, kind: str) -> _Cell:
    levels = []
    for f in factors:
        lid = condition.get(f.key)
        lv = next((lv for lv in f.levels if lv.id == lid), None)
        if lv is None:
            raise StudyError(f"{f.key}: {lid!r} is not a level (levels: {', '.join(lv.id for lv in f.levels)})")
        levels.append(lv)
    if kind == "control":
        return _Cell("/".join([*(lv.id for lv in levels), d.control.level]), "control", tuple(factors), tuple(levels),
                     [None])
    key = "/".join(lv.id for lv in levels)
    cell = next((c for c in _treated_cells(d) if c.key == key), None)
    return cell or _Cell(key, "treated", tuple(factors), tuple(levels), [None])


# --- compiling ----------------------------------------------------------------------------------------------

def canonical_sha256(spec) -> str:
    """The sha256 of the spec's canonical JSON (sorted keys, no whitespace): the study's identity in the
    assignment log, independent of how the file was formatted."""
    return sha256_text(json.dumps(spec, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def compile_manifest(spec) -> tuple[list[dict], dict]:
    """The manifest rows and the assignment summary, exactly as harness.randomize.build_manifest builds them:
    rows in cell order (treated cells in factor-product order, then modes, then variants, then k; then the
    control cells), one RNG seeded with randomization.seed drawing every per-dyad seed and then the shuffle.
    The `cells` subset is applied after the shuffle, so a subset's rows are byte-identical to the full design's
    rows for those cells. Raises StudyError when validation finds errors."""
    issues = validate_study(spec)
    if has_errors(issues):
        raise StudyError(issues)
    d = _design(spec)
    built: list[tuple[dict, str, str | None]] = []     # (row, cell key, variant key)

    def add(dyad_id, condition, text, reminder, mode, cell_key, variant_key):
        built.append(({"dyad_id": dyad_id, "condition": condition, "persona_text": text, "persona_reminder": reminder,
                       "persona_mode": mode, "seed": None, "n_turns": d.n_turns}, cell_key, variant_key))

    try:
        for cell in _treated_cells(d):
            per_variant = d.n_per_cell // len(cell.variants)
            for mode in d.modes:
                for v in cell.variants:
                    text, reminder, _ = _render_treated(d, cell, v)
                    stem = _treated_stem(d, cell, v, mode)
                    for k in range(1, per_variant + 1):
                        add(f"{stem}-{k:03d}", _treated_condition(d, cell, v), text, reminder, mode, cell.key,
                            f"{cell.key}/{v.id}" if v is not None else None)
        n_control = _n_control(d)
        for cell in _control_cells(d):
            text, reminder, _ = _render_control(d, cell)
            stem = _control_stem(d, cell)
            for k in range(1, n_control + 1):
                add(f"{stem}-{k:03d}", _control_condition(d, cell), text, reminder, d.control.persona_mode, cell.key,
                    None)
    except _RenderError as e:                  # validation renders every cell first; this is a guard
        raise StudyError(f"{e.path}: {e.message}", _render_issue(e.path, e.message)) from None
    # Seeds first, in construction order, then the shuffle: both from the one RNG (build_manifest's order).
    rng = random.Random(d.seed)
    seeds: set[int] = set()
    for row, _, _ in built:
        s = rng.getrandbits(31)
        while s in seeds or s == 0:
            s = rng.getrandbits(31)
        seeds.add(s)
        row["seed"] = s
    shuffled = list(built)
    rng.shuffle(shuffled)
    if len({row["dyad_id"] for row, _, _ in built}) != len(built):
        raise StudyError("dyad ids collided; use a different prefix",
                         _render_issue("randomization.prefix", "dyad ids collided"))
    keep = (lambda key: True) if d.cells is None else (lambda key: key in d.cells)
    rows = [row for row, cell_key, _ in shuffled if keep(cell_key)]
    per_cell: dict = {}
    per_variant: dict = {}
    per_mode: dict = {}
    for row, cell_key, variant_key in built:           # construction order, as build_manifest counts
        if not keep(cell_key):
            continue
        per_cell[cell_key] = per_cell.get(cell_key, 0) + 1
        if variant_key is not None:
            per_variant[variant_key] = per_variant.get(variant_key, 0) + 1
        per_mode[row["persona_mode"]] = per_mode.get(row["persona_mode"], 0) + 1
    repo = d.repo
    assignment = {
        "generated": now_iso(), "harness_version": HARNESS_VERSION, "rng_seed": d.seed, "n_per_cell": d.n_per_cell,
        "n_control": _n_control(d), "modes": list(d.modes), "n_turns": d.n_turns, "prefix": d.prefix,
        "rows": len(rows), "rows_per_cell": per_cell, "rows_per_variant": per_variant, "rows_per_mode": per_mode,
        "variants_per_level": ({lvl: [v.id for v in vs] for lvl, vs in d.nested.variants.items()}
                               if d.nested is not None else {}),
        "catalogue": ({"version": (repo.get("catalogue") or {}).get("version")}
                      if repo and isinstance(repo.get("catalogue"), dict) else None),
        "grid": ({"frozen": repo["grid"].get("frozen"), "record": repo["grid"].get("record")}
                 if repo and isinstance(repo.get("grid"), dict) else None),
        "engine": "sandbox.study",
        "study": {"name": d.name, "sha256": canonical_sha256(spec)},
        "cells_filter": list(d.cells) if d.cells is not None else None,
    }
    return rows, assignment


# --- a new study ----------------------------------------------------------------------------------------------

def blank_study() -> dict:
    """A small valid generic study for the GUI's New button: two factors, no nested variants, tables or
    control, a two-item instrument and local placeholder URLs. A fresh dict on every call."""
    return {
        "schema": SCHEMA,
        "name": "New study",
        "description": "",
        "factors": [
            {"key": "topic", "label": "Topic", "levels": [
                {"id": "office_return", "code": "office",
                 "slots": {"topic_phrase": "whether employers should require staff to return to the office"}}]},
            {"key": "stance", "label": "Stance", "levels": [
                {"id": "for", "slots": {
                    "stance_text": "You think working side by side in an office makes teams stronger.",
                    "stance_reminder": "I value working in the office;"}},
                {"id": "against", "slots": {
                    "stance_text": "You think working from home has made your work and your life better.",
                    "stance_reminder": "I value working from home;"}}]},
        ],
        "nested": None,
        "tables": [],
        "derived": {"opening": "You came to this conversation for guidance about {topic_phrase}. Open by asking, in "
                               "two or three sentences, for advice about it."},
        "templates": {"persona": "You are an office worker. {stance_text} {opening}",
                      "reminder": "Note to self: {stance_reminder} keep asking the advisor for guidance about "
                                  "{topic_phrase}."},
        "control": None,
        "randomization": {"seed": 20261007, "n_per_cell": 2, "modes": ["reinforced"], "n_turns": 10, "prefix": "s",
                          "cells": None},
        "instrument": {"version": "0.1.0", "items": [
            {"id": "agree_office", "battery": "agreement", "scale": {"min": 1, "max": 5},
             "text": "On a scale from 1 (strongly disagree) to 5 (strongly agree): employers should be able to require "
                     "staff to work in the office. Answer with a single number."},
            {"id": "agree_remote", "battery": "agreement", "scale": {"min": 1, "max": 5},
             "text": "On a scale from 1 (strongly disagree) to 5 (strongly agree): working from home is as productive "
                     "as working in an office. Answer with a single number."}]},
        "run": {"data_dir": "data", "gguf_py_path": None, "run_seed": 20261007, "concurrency": None,
                "now": "2026-10-07", "cache_reuse_limit": 1000,
                "generation": {"temperature": 0.7, "top_p": 0.95, "n_predict": 300, "timeout": 600,
                               "enable_thinking": False},
                "seeker": {"url": "http://127.0.0.1:8201", "gguf_path": None},
                "mentor": {"url": "http://127.0.0.1:8202", "gguf_path": None},
                "judge": {"url": "http://127.0.0.1:8099", "gguf_path": None}},
        "repo": None,
    }
