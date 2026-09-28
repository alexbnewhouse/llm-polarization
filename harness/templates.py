"""Chat-template rendering owned by the harness: read the template from the GGUF, render with jinja2,
and prove parity with the server's own rendering before a run."""
from __future__ import annotations
import datetime as _dt
import difflib
import os, sys
from dataclasses import dataclass
from pathlib import Path
import jinja2
from harness.log import sha256_text


class TemplateError(Exception):
    """Raised when template loading or rendering fails."""
    pass


@dataclass(frozen=True)
class ChatTemplate:
    """A chat template's jinja source, its BOS and EOS token strings, and the source's sha256."""
    source: str
    bos: str
    eos: str
    sha256: str

    @classmethod
    def from_source(cls, source: str, bos: str = "", eos: str = "") -> "ChatTemplate":
        """Create a ChatTemplate from source, computing its SHA256 hash."""
        return cls(source, bos, eos, sha256_text(source))


# The seeker's reinforced shape: system prompt, history, trailing reminder. `check` renders it whole to
# test the trailing system message, and without the reminder for the system-first parity test.
FIXTURE_MESSAGES = [
    {"role": "system", "content": "You are a fixture persona used only to check template rendering."},
    {"role": "user", "content": "Fixture user line one."},
    {"role": "assistant", "content": "Fixture assistant line one."},
    {"role": "user", "content": "Fixture user line two."},
    {"role": "system", "content": "Fixture trailing reminder."},
]

# The mentor's only shape: no system message at all. Many templates inject a default system block
# precisely when none is supplied, so parity proved on the system-first fixture proves nothing here.
FIXTURE_MESSAGES_USER_FIRST = [m for m in FIXTURE_MESSAGES if m["role"] != "system"]


def _gguf_module(gguf_py_path: str | None):
    """Import llama.cpp's gguf-py from `gguf_py_path` (else $GGUF_PY_PATH), putting it first on sys.path."""
    path = gguf_py_path or os.environ.get("GGUF_PY_PATH")
    if path and path not in sys.path:
        sys.path.insert(0, path)
    try:
        import gguf  # noqa: WPS433
    except ImportError as e:
        # gguf-py is not pip-installed and brings its own dependencies (numpy first of all). Without this,
        # the first command the researcher runs answers with a raw ModuleNotFoundError traceback.
        raise TemplateError(f"cannot import {e.name!r}, which llama.cpp's gguf-py needs "
                            f"(gguf_py_path={path!r}); pip install -r harness/requirements.txt") from e
    if not hasattr(gguf, "GGUFReader"):
        raise TemplateError("imported a 'gguf' module without GGUFReader; set gguf_py_path to llama.cpp's gguf-py")
    return gguf


# gguf-py stores each metadata field as a list of `parts` plus a `data` list of indices into it. For a
# string field there is one index and the part is raw bytes; for the BOS/EOS ids there is one index and
# the part is a one-element integer array. Hence the [f.data[0]].
def _field_str(reader, key: str) -> str | None:
    """A string metadata field from the GGUF, or None when the key is absent."""
    f = reader.fields.get(key)
    if f is None:
        return None
    return bytes(f.parts[f.data[0]]).decode("utf-8")


def _token_text(reader, id_key: str) -> str:
    """The vocabulary string for the token id stored under `id_key` (BOS or EOS); "" when either is absent."""
    f = reader.fields.get(id_key)
    if f is None:
        return ""
    tid = int(f.parts[f.data[0]][0])
    toks = reader.fields.get("tokenizer.ggml.tokens")
    if toks is None:
        return ""
    return bytes(toks.parts[toks.data[tid]]).decode("utf-8", errors="replace")


def read_template_from_gguf(path: str | Path, gguf_py_path: str | None = None) -> ChatTemplate:
    """Read the chat template and the BOS/EOS token strings out of a GGUF file's metadata."""
    path = Path(path)
    if not path.exists():
        raise TemplateError(f"GGUF not found: {path}")
    gguf = _gguf_module(gguf_py_path)
    reader = gguf.GGUFReader(str(path))
    source = _field_str(reader, "tokenizer.chat_template")
    if not source:
        raise TemplateError(f"{path} has no tokenizer.chat_template")
    bos = _token_text(reader, "tokenizer.ggml.bos_token_id")
    eos = _token_text(reader, "tokenizer.ggml.eos_token_id")
    return ChatTemplate.from_source(source, bos, eos)


def _env(now: str) -> jinja2.Environment:
    """A jinja2 environment that renders chat templates as llama.cpp does, with `strftime_now` pinned to
    the date `now` (YYYY-MM-DD) so a template that prints the date renders the same prompt every day."""
    # These three settings are what make our rendering byte-identical to Hugging Face's and llama.cpp's.
    # Do not change them without re-running `check`'s parity test against every arm.
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=False, keep_trailing_newline=True)
    day = _dt.datetime.strptime(now, "%Y-%m-%d")

    def raise_exception(msg):
        raise TemplateError(str(msg))

    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: day.strftime(fmt)
    return env


def render(tpl: ChatTemplate, messages: list[dict], *, add_generation_prompt: bool = True,
           now: str = "2026-09-08", enable_thinking: bool = False) -> str:
    """Render `messages` through the template. `now` pins strftime_now; `tools` is always None, which is
    what keeps Olmo-3's template from emitting a tools block."""
    try:
        return _env(now).from_string(tpl.source).render(
            messages=messages, add_generation_prompt=add_generation_prompt,
            bos_token=tpl.bos, eos_token=tpl.eos, tools=None, enable_thinking=enable_thinking)
    except jinja2.TemplateError as e:
        raise TemplateError(f"template render failed: {e}") from e


def parity_check(tpl: ChatTemplate, client, messages: list[dict], *, now: str,
                 enable_thinking: bool = False) -> tuple[bool, str, str]:
    """Compare our template rendering to the server's; returns (ok, ours, theirs). parity_detail says why."""
    ok, ours, theirs, _ = parity_detail(tpl, client, messages, now=now, enable_thinking=enable_thinking)
    return ok, ours, theirs


# The most characters one date print may change between two dates ("September 8" -> "October 18").
DATE_DIFF_MAX = 16


def date_only_diff(a: str, b: str, max_segments: int = 1) -> bool:
    """True when a and b differ only in at most `max_segments` short spans (DATE_DIFF_MAX characters each):
    what rendering one template on two dates may change, and nothing else. Differences a few characters
    apart are one span: 2026-09-08 against 2026-10-18 is several small edits inside one date."""
    spans: list[list[int]] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if spans and i1 - spans[-1][1] <= 4 and j1 - spans[-1][3] <= 4:
            spans[-1][1], spans[-1][3] = i2, j2
        else:
            spans.append([i1, i2, j1, j2])
    return len(spans) <= max_segments and all(max(i2 - i1, j2 - j1) <= DATE_DIFF_MAX
                                              for i1, i2, j1, j2 in spans)


def _server_dates() -> list[str]:
    """The dates a server on this box may print today: local and UTC, and the days either side of local
    for a check run across midnight."""
    local = _dt.datetime.now()
    day = _dt.timedelta(days=1)
    days = [local, _dt.datetime.now(_dt.timezone.utc), local - day, local + day]
    return list(dict.fromkeys(d.strftime("%Y-%m-%d") for d in days))


def _same(ours: str, theirs: str, bos: str) -> tuple[bool, bool]:
    """(equal, equal only once our leading BOS is dropped). llama.cpp's chat formatting strips a leading BOS
    from the rendered prompt when the vocabulary adds one itself (common/chat.cpp), so /apply-template
    lacks the bos_token the template printed."""
    if ours == theirs:
        return True, False
    if bos and ours.startswith(bos) and not theirs.startswith(bos) and ours[len(bos):] == theirs:
        return True, True
    return False, False


def parity_detail(tpl: ChatTemplate, client, messages: list[dict], *, now: str,
                  enable_thinking: bool = False) -> tuple[bool, str, str, list[str]]:
    """Compare our rendering with the server's /apply-template, as `check` does: (ok, ours, theirs, notes).
    Both sides render with `enable_thinking` (sent as chat_template_kwargs; red-team L4). Two differences
    are allowed and noted:

    - `bos_stripped`: the server's render is ours without its leading BOS (red-team L5);
    - `date_adjusted <day>`: a template that prints the date (strftime_now) renders the server's wall-clock
      date there, which cannot be overridden, so ours is rendered with that date for this comparison only,
      and must equal the pinned render but for the date (red-team H4). The prompts keep `now`."""
    theirs = client.apply_template(messages, chat_template_kwargs={"enable_thinking": enable_thinking})
    ours = render(tpl, messages, now=now, enable_thinking=enable_thinking)
    ok, bos = _same(ours, theirs, tpl.bos)
    notes = ["bos_stripped"] if bos else []
    if ok or "strftime_now" not in tpl.source:
        return ok, ours, theirs, notes
    for day in _server_dates():
        if day == now:
            continue
        alt = render(tpl, messages, now=day, enable_thinking=enable_thinking)
        ok, bos = _same(alt, theirs, tpl.bos)
        if not ok:
            continue
        if not date_only_diff(ours, alt, max_segments=tpl.source.count("strftime_now")):
            return False, ours, theirs, [f"matches only when rendered on {day}, and that render differs from "
                                         f"the one on {now} beyond the date"]
        return True, ours, theirs, (["bos_stripped"] if bos else []) + [f"date_adjusted {day}"]
    return False, ours, theirs, []
