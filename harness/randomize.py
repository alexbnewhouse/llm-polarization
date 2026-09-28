"""The condition randomizer: prompts/grid.json + the persona catalogue -> one *-dyads.jsonl manifest for
`harness.run --manifest`, plus the *-assignment.json log that is what "randomization" means in the
design section. Notion task "Build the condition randomizer and assignment log" (2026-09-14).

    python -m harness.randomize --grid prompts/grid.json --catalogue prompts/personas/catalogue.json \\
        --out pilot-dyads.jsonl --seed 20260918 --n-per-cell 6 --modes reinforced,once --prefix p
    python -m harness.randomize --subset-of w1-dyads.jsonl --per-variant 30 --control 90 \\
        --out w1-n90-dyads.jsonl

Every treated cell (topic x ideology x openness) gets n_per_cell rows split evenly across that ideology
level's role variants from the catalogue; each topic gets a bare control cell; every row has its own
seed from one seeded RNG; the rows are shuffled so cells interleave across the server's slots.

--subset-of is the pre-analysis plan's descope rule (docs/pap/pre-analysis-plan.md section 11): it keeps
rows of an existing manifest by dyad_id and never re-randomizes, so a descoped arm runs the same dyads,
with the same seeds, as an arm already run at full size."""
from __future__ import annotations
import argparse
import json
import random
import re
import string
import sys
from pathlib import Path
from harness import __version__
from harness.grid import check_conditions, load_grid, treated_cells
from harness.log import now_iso, sha256_file
from harness.transcript import PERSONA_MODES

CONTROL_MODE = "reinforced"       # the control always runs reinforced so its mechanics match the treated cells
SLOT_FIELDS = ("slug", "name", "backstory", "reminder_self")   # every role variant must fill all four


def load_catalogue(path: str | Path) -> dict:
    """Read the persona catalogue JSON. Its shape is documented in prompts/README.md and shown by
    prompts/personas/catalogue.example.json."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_catalogue(catalogue: dict, grid: dict) -> None:
    """Raise ValueError unless the catalogue can fill every cell of the grid: role slugs registered only
    under grid ideology levels, unique, at most variants_per_level each; an anchor set for every
    (ideology, topic); both openness texts; the shared strings; the control."""
    f = grid["factors"]
    for key in ("version", "template", "shared", "openness", "roles", "anchors", "control"):
        if key not in catalogue:
            raise ValueError(f"catalogue is missing {key!r}")
    seen: set[str] = set()
    for ideology, variants in catalogue["roles"].items():
        if ideology not in f["ideology"]:
            raise ValueError(f"catalogue registers roles under {ideology!r}, which is not a grid ideology level")
        if not variants:
            raise ValueError(f"catalogue has no role variant for {ideology!r}")
        if len(variants) > grid["variants_per_level"]:
            raise ValueError(f"{ideology!r} has {len(variants)} variants; the grid allows {grid['variants_per_level']}")
        for v in variants:
            for field in SLOT_FIELDS:
                if not v.get(field):
                    raise ValueError(f"role variant {v.get('slug')!r} under {ideology!r} is missing {field!r}")
            if v["slug"] in seen:
                raise ValueError(f"duplicate role slug {v['slug']!r}")
            seen.add(v["slug"])
    for ideology in f["ideology"]:
        if ideology not in catalogue["roles"]:
            raise ValueError(f"catalogue has no role variant for {ideology!r}")
        for topic in f["topic"]:
            anchors = (catalogue["anchors"].get(ideology) or {}).get(topic)
            if not anchors or not all(isinstance(a, str) and a for a in anchors):
                raise ValueError(f"catalogue has no anchor set for ({ideology}, {topic})")
    for level in f["openness"]:
        o = catalogue["openness"].get(level) or {}
        if not o.get("text") or not o.get("reminder"):
            raise ValueError(f"catalogue openness {level!r} needs 'text' and 'reminder'")
    for topic in f["topic"]:
        if topic not in catalogue["shared"].get("topic_phrase", {}):
            raise ValueError(f"catalogue shared.topic_phrase has no entry for {topic!r}")
    for key in ("opening",):
        if not catalogue["shared"].get(key):
            raise ValueError(f"catalogue shared.{key} is missing")
    for key in ("persona", "reminder"):
        if not catalogue["template"].get(key) or not catalogue["control"].get(key):
            raise ValueError(f"catalogue template.{key} and control.{key} are both required")


class _Slots(dict):
    """format_map helper that names the missing slot instead of raising a bare KeyError."""
    def __missing__(self, key):
        raise ValueError(f"template slot {{{key}}} has no value in the catalogue")


def _fill(template: str, slots: dict) -> str:
    """Fill {slot} placeholders; a slot with no value is an error, never left in the prompt."""
    return template.format_map(_Slots(slots)).strip()


def opening_line(catalogue: dict, topic: str) -> str:
    """The setting line and opening request: one shared string, identical in every cell, control included."""
    return _fill(catalogue["shared"]["opening"], {"topic_phrase": catalogue["shared"]["topic_phrase"][topic], "topic": topic})


def _slots(catalogue: dict, topic: str) -> dict:
    """The slots shared by every prompt for one topic, treated and control: topic, topic_phrase, opening."""
    return {"topic": topic, "topic_phrase": catalogue["shared"]["topic_phrase"][topic],
            "opening": opening_line(catalogue, topic)}


def render_persona(catalogue: dict, ideology: str, role: dict, topic: str, openness: str) -> tuple[str, str]:
    """The seeker system prompt and the compact reminder for one (ideology, role, topic, openness)."""
    anchors = list(catalogue["anchors"][ideology][topic])
    o = catalogue["openness"][openness]
    slots = {**_slots(catalogue, topic), "ideology": ideology, "role": role["slug"], "name": role["name"],
             "backstory": role["backstory"], "reminder_self": role["reminder_self"],
             "anchors": " ".join(anchors), "openness": openness, "openness_text": o["text"],
             "openness_reminder": o["reminder"]}
    for i, a in enumerate(anchors, 1):
        slots[f"anchor_{i}"] = a
    return _fill(catalogue["template"]["persona"], slots), _fill(catalogue["template"]["reminder"], slots)


def render_control(catalogue: dict, topic: str) -> tuple[str, str]:
    """The bare control prompt and its framing-only reminder for one topic: no name, role or openness."""
    slots = _slots(catalogue, topic)
    return _fill(catalogue["control"]["persona"], slots), _fill(catalogue["control"]["reminder"], slots)


def _topic_code(topic: str) -> str:
    """A short id-safe code for a topic: the first word, six letters ('immig', 'decarb')."""
    return topic.split("_")[0][:6]


def build_manifest(grid: dict, catalogue: dict, *, seed: int, n_per_cell: int | None = None,
                   n_control: int | None = None, modes: tuple[str, ...] | None = None,
                   n_turns: int | None = None, prefix: str = "w") -> tuple[list[dict], dict]:
    """Expand the grid against the catalogue into manifest rows and the assignment summary. Defaults come
    from the grid: n_per_cell (135), the control's n_per_cell, persona_mode (reinforced) and n_turns (40).
    The pilot passes n_per_cell=5 and modes=('reinforced', 'once'); the control cells are always
    reinforced and are not multiplied by the delivery modes. One RNG seeded with `seed` draws every
    per-dyad seed and the final shuffle, so the same arguments give the same file."""
    validate_catalogue(catalogue, grid)
    n_per_cell = grid["n_per_cell"] if n_per_cell is None else int(n_per_cell)
    # The control is sized at one treated cell per topic (factorial.md), so it follows n_per_cell unless
    # overridden; the grid's own control n_per_cell applies only when nothing is overridden.
    if n_control is None:
        n_control = n_per_cell if n_per_cell != grid["n_per_cell"] else grid["control"]["n_per_cell"]
    n_control = int(n_control)
    modes = tuple(modes) if modes else (grid["fixed"]["persona_mode"],)
    n_turns = int(grid["fixed"]["n_turns"] if n_turns is None else n_turns)
    rng = random.Random(seed)
    rows: list[dict] = []
    per_cell: dict[str, int] = {}
    per_variant: dict[str, int] = {}
    per_mode: dict[str, int] = {}

    def add(dyad_id, condition, text, reminder, mode):
        rows.append({"dyad_id": dyad_id, "condition": condition, "persona_text": text, "persona_reminder": reminder,
                     "persona_mode": mode, "seed": None, "n_turns": n_turns})
        per_mode[mode] = per_mode.get(mode, 0) + 1

    for topic, ideology, openness in treated_cells(grid):
        variants = catalogue["roles"][ideology]
        k = len(variants)
        if n_per_cell % k:
            lower, upper = n_per_cell // k * k, (n_per_cell // k + 1) * k
            raise ValueError(f"n_per_cell {n_per_cell} does not split evenly across the {k} role variant(s) "
                             f"of {ideology!r}; use a multiple of {k} (--n-per-cell "
                             + (f"{lower} or {upper}" if lower else f"{upper}")
                             + "), or a catalogue with one variant per level")
        per_var = n_per_cell // len(variants)
        cell = f"{topic}/{ideology}/{openness}"
        for mode in modes:
            for role in variants:
                text, reminder = render_persona(catalogue, ideology, role, topic, openness)
                for k in range(1, per_var + 1):
                    dyad_id = f"{prefix}-{_topic_code(topic)}-{ideology}-{openness}-{role['slug']}-{mode}-{k:03d}"
                    add(dyad_id, {"topic": topic, "ideology": ideology, "openness": openness, "role": role["slug"]},
                        text, reminder, mode)
                    per_cell[cell] = per_cell.get(cell, 0) + 1
                    per_variant[f"{cell}/{role['slug']}"] = per_variant.get(f"{cell}/{role['slug']}", 0) + 1
    control = grid["control"]
    for topic in grid["factors"]["topic"]:
        text, reminder = render_control(catalogue, topic)
        cell = f"{topic}/{control['ideology']}"
        for k in range(1, n_control + 1):
            add(f"{prefix}-{_topic_code(topic)}-control-{CONTROL_MODE}-{k:03d}",
                {"topic": topic, "ideology": control["ideology"], "openness": control["openness"], "role": control["role"]},
                text, reminder, CONTROL_MODE)
            per_cell[cell] = per_cell.get(cell, 0) + 1
    # Seeds first, in grid order, then the shuffle: both from the one RNG, so the file is a pure function
    # of (grid, catalogue, arguments, seed).
    seeds: set[int] = set()
    for row in rows:
        s = rng.getrandbits(31)
        while s in seeds or s == 0:
            s = rng.getrandbits(31)
        seeds.add(s)
        row["seed"] = s
    rng.shuffle(rows)
    if len({r["dyad_id"] for r in rows}) != len(rows):
        raise ValueError("dyad ids collided; use a different --prefix")
    check_conditions(rows, grid)
    assignment = {"generated": now_iso(), "harness_version": __version__, "rng_seed": seed,
                  "n_per_cell": n_per_cell, "n_control": n_control, "modes": list(modes), "n_turns": n_turns,
                  "prefix": prefix, "rows": len(rows), "rows_per_cell": per_cell, "rows_per_variant": per_variant,
                  "rows_per_mode": per_mode,
                  "variants_per_level": {lvl: [v["slug"] for v in vs] for lvl, vs in catalogue["roles"].items()},
                  "catalogue": {"version": catalogue["version"]},
                  "grid": {"frozen": grid.get("frozen"), "record": grid.get("record")}}
    return rows, assignment


def assignment_log_path(out: str | Path) -> Path:
    """Where the assignment log of the manifest `out` lives: <stem>-assignment.json beside it, where <stem>
    is the name without its extension and any trailing "-dyads" (pilot-dyads.jsonl ->
    pilot-assignment.json)."""
    out = Path(out)
    stem = out.name[: -len(out.suffix)] if out.suffix else out.name
    stem = stem[: -len("-dyads")] if stem.endswith("-dyads") else stem
    return out.with_name(f"{stem}-assignment.json")


def write_manifest(out: str | Path, rows: list[dict], assignment: dict, *, grid_path: str | Path,
                   catalogue_path: str | Path) -> Path:
    """Write the manifest as JSONL and its assignment log beside it, with the grid, catalogue and output
    file hashes; return the log's path. The log is <stem>-assignment.json, where <stem> is the output name
    without its extension and any trailing "-dyads": pilot-dyads.jsonl -> pilot-assignment.json."""
    out = Path(out)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    sidecar = assignment_log_path(out)
    a = dict(assignment)
    a["grid"] = {**a.get("grid", {}), "path": str(grid_path), "sha256": sha256_file(Path(grid_path))}
    a["catalogue"] = {**a.get("catalogue", {}), "path": str(catalogue_path), "sha256": sha256_file(Path(catalogue_path))}
    a["output"] = {"path": str(out), "sha256": sha256_file(out)}
    sidecar.write_text(json.dumps(a, indent=2, ensure_ascii=False), encoding="utf-8")
    return sidecar


# The within-variant index k: the last three digits of a dyad_id build_manifest wrote. It is assigned in grid
# order before the shuffle, so it is independent of seeds and outcomes.
_INDEX = re.compile(r"-(\d{3})$")


def subset_lines(lines: list[bytes], *, per_variant: int | None = None, control: int | None = None,
                 ideology: tuple[str, ...] | None = None, topic: tuple[str, ...] | None = None
                 ) -> tuple[list[bytes], dict]:
    """The descope rule over a manifest's raw lines: keep a treated row when its index k is at most
    `per_variant`, a control row when k is at most `control`, and (when given) only the listed treated
    ideology levels and topics; the control is never cut by level. Lines are returned byte for byte, in
    order. Also returns the per-cell counts of what was kept."""
    kept, per_cell = [], {}
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        m = _INDEX.search(str(row.get("dyad_id", "")))
        if not m:
            raise ValueError(f"line {n}: dyad_id {row.get('dyad_id')!r} does not end in a three-digit index; "
                             "only a manifest harness.randomize wrote can be subset")
        k = int(m.group(1))
        c = row.get("condition") or {}
        is_control = c.get("ideology") == "none"
        limit = control if is_control else per_variant
        if limit is not None and k > limit:
            continue
        if topic and c.get("topic") not in topic:
            continue
        if ideology and not is_control and c.get("ideology") not in ideology:
            continue
        kept.append(line if line.endswith(b"\n") else line + b"\n")
        cell = "/".join(str(c.get(x)) for x in ("topic", "ideology", "openness") if c.get(x) is not None)
        per_cell[cell] = per_cell.get(cell, 0) + 1
    return kept, per_cell


def write_subset(parent: str | Path, out: str | Path, **flt) -> Path:
    """Write the subset of `parent` that `flt` keeps (subset_lines) to `out`, and its assignment log beside
    it: the parent's path and sha256 (and its own assignment log's), the filter, and the counts. Returns
    the log's path. Refuses to write over the parent or its assignment log."""
    parent, out = Path(parent), Path(out)
    if all(v is None for v in flt.values()):
        raise ValueError("no filter given: pass --per-variant, --control, --ideology or --topic")
    if out.resolve() == parent.resolve() or assignment_log_path(out).resolve() == \
            assignment_log_path(parent).resolve():
        raise ValueError(f"--out {out} would overwrite the parent manifest or its assignment log")
    lines = parent.read_bytes().splitlines(keepends=True)
    kept, per_cell = subset_lines(lines, **flt)
    if not kept:
        raise ValueError("the filter keeps no rows")
    out.write_bytes(b"".join(kept))
    parent_log = assignment_log_path(parent)
    a = {"generated": now_iso(), "harness_version": __version__, "kind": "subset",
         "rule": "docs/pap/pre-analysis-plan.md section 11: rows kept by dyad_id, never re-randomized",
         "parent": {"path": str(parent), "sha256": sha256_file(parent),
                    "assignment": ({"path": str(parent_log), "sha256": sha256_file(parent_log)}
                                   if parent_log.exists() else None)},
         "filter": {k: (list(v) if isinstance(v, tuple) else v) for k, v in flt.items()},
         "rows": len(kept), "rows_dropped": sum(1 for l in lines if l.strip()) - len(kept),
         "rows_per_cell": per_cell, "output": {"path": str(out), "sha256": sha256_file(out)}}
    sidecar = assignment_log_path(out)
    sidecar.write_text(json.dumps(a, indent=2, ensure_ascii=False), encoding="utf-8")
    return sidecar


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns 0 on success, 1 with one error line on stderr for a bad grid or catalogue."""
    ap = argparse.ArgumentParser(prog="harness.randomize", description=__doc__.split("\n\n")[0])
    ap.add_argument("--grid", default="prompts/grid.json")
    ap.add_argument("--catalogue", help="the persona catalogue (required unless --subset-of)")
    ap.add_argument("--out", help="the *-dyads.jsonl to write; the assignment log lands beside it")
    ap.add_argument("--seed", type=int, help="the RNG seed; recorded in the assignment log")
    ap.add_argument("--check", action="store_true", help="validate the catalogue against the grid and write nothing")
    ap.add_argument("--n-per-cell", type=int, default=None, help="rows per treated cell (grid default: 135)")
    ap.add_argument("--n-control", type=int, default=None,
                    help="rows per control cell (default: --n-per-cell when that is given, else the grid's "
                         "control n_per_cell, 135)")
    ap.add_argument("--modes", default=None, help="comma-separated persona modes (grid default: reinforced)")
    ap.add_argument("--n-turns", type=int, default=None)
    ap.add_argument("--prefix", default="w", help="dyad_id prefix, e.g. p for the pilot, w1 for wave 1")
    sub = ap.add_argument_group("descope (docs/pap/pre-analysis-plan.md section 11)")
    sub.add_argument("--subset-of",
                     help="an existing manifest to keep rows of, by dyad_id; nothing is re-randomized")
    sub.add_argument("--per-variant", type=int,
                     help="keep treated rows whose within-variant index is at most this")
    sub.add_argument("--control", type=int, help="keep control rows whose index is at most this")
    sub.add_argument("--ideology", help="comma-separated treated levels to keep (the control is always kept)")
    sub.add_argument("--topic", help="comma-separated topics to keep")
    a = ap.parse_args(argv)
    if a.subset_of:
        if a.out is None or a.check or a.seed is not None or a.n_per_cell is not None or a.modes:
            ap.error("--subset-of takes --out and the filters (--per-variant, --control, --ideology, "
                     "--topic), and nothing that would randomize")
        split = lambda v: tuple(x.strip() for x in v.split(",") if x.strip()) if v else None  # noqa: E731
        try:
            sidecar = write_subset(a.subset_of, a.out, per_variant=a.per_variant, control=a.control,
                                   ideology=split(a.ideology), topic=split(a.topic))
        except (ValueError, OSError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        rec = json.loads(sidecar.read_text(encoding="utf-8"))
        print(f"kept {rec['rows']} of {rec['rows'] + rec['rows_dropped']} rows of {a.subset_of} in {a.out}; "
              f"assignment log {sidecar} (parent sha256 {rec['parent']['sha256'][:12]})")
        return 0
    if a.catalogue is None:
        ap.error("--catalogue is required unless --subset-of")
    if not a.check and (a.out is None or a.seed is None):
        ap.error("--out and --seed are required unless --check")
    modes = tuple(m.strip() for m in a.modes.split(",")) if a.modes else None
    if modes is not None and (not all(modes) or len(set(modes)) != len(modes)
                              or any(m not in PERSONA_MODES for m in modes)):
        ap.error(f"--modes must be distinct values from {', '.join(PERSONA_MODES)}, comma-separated; "
                 f"got {a.modes!r}")
    try:
        grid = load_grid(a.grid)
        catalogue = load_catalogue(a.catalogue)
        if a.check:
            validate_catalogue(catalogue, grid)
            n = {lvl: len(v) for lvl, v in catalogue["roles"].items()}
            print(f"{a.catalogue} (version {catalogue['version']}) is valid against {a.grid}: role variants per level {n}")
            return 0
        rows, assignment = build_manifest(grid, catalogue, seed=a.seed, n_per_cell=a.n_per_cell,
                                          n_control=a.n_control, modes=modes, n_turns=a.n_turns, prefix=a.prefix)
        sidecar = write_manifest(a.out, rows, assignment, grid_path=a.grid, catalogue_path=a.catalogue)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"wrote {len(rows)} dyads to {a.out} ({len(assignment['rows_per_cell'])} cells, modes {assignment['modes']}, "
          f"rng seed {a.seed}); assignment log {sidecar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
