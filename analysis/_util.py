"""Small shared helpers: files in and out, markdown tables, grouping."""
from __future__ import annotations
import csv
import json
from pathlib import Path


def write_json(path: Path, obj) -> Path:
    """Write `obj` as indented JSON, creating the parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_default) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, rows) -> Path:
    """Write rows as JSONL (replacing the file), creating the parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_default) + "\n")
    return path


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> Path:
    """Write rows as CSV with a header. Columns default to the union of keys in first-seen order; nested
    values (dicts, lists) are written as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        for r in rows:
            columns += [k for k in r if k not in columns]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow([_cell(r.get(c)) for c in columns])
    return path


def read_csv(path: Path) -> list[dict]:
    """Every row of a CSV as a dict of strings."""
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False)
    return v


def _default(o):
    """JSON fallback for numpy scalars and arrays."""
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o)}")


def md_table(headers: list[str], rows: list[list]) -> str:
    """A GitHub markdown table."""
    out = ["| " + " | ".join(str(h) for h in headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


def fmt(x, nd: int = 3) -> str:
    """A number for a table: fixed decimals, 'n/a' for None or NaN."""
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{nd}f}"


def pct(k: int, n: int) -> str:
    """'k/n (p%)', or 'k/0' when there is nothing to divide by."""
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else f"{k}/0"


def group_by(rows, key):
    """{key(row): [rows]} preserving first-seen order."""
    out: dict = {}
    for r in rows:
        out.setdefault(key(r), []).append(r)
    return out


def try_frame(rows: list[dict]):
    """A pandas DataFrame of `rows` when pandas is installed, else the rows unchanged."""
    try:
        import pandas as pd
    except ImportError:
        return rows
    return pd.DataFrame(rows)
