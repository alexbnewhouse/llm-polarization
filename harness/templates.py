"""Chat-template rendering owned by the harness: read the template from the GGUF, render with jinja2,
and prove parity with the server's own rendering before a run."""
from __future__ import annotations
import datetime as _dt
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
    """Holds a chat template source and tokens; computes its SHA256 hash."""
    source: str
    bos: str
    eos: str
    sha256: str

    @classmethod
    def from_source(cls, source: str, bos: str = "", eos: str = "") -> "ChatTemplate":
        """Create a ChatTemplate from source, computing its SHA256 hash."""
        return cls(source, bos, eos, sha256_text(source))


FIXTURE_MESSAGES = [
    {"role": "system", "content": "You are a fixture persona used only to check template rendering."},
    {"role": "user", "content": "Fixture user line one."},
    {"role": "assistant", "content": "Fixture assistant line one."},
    {"role": "user", "content": "Fixture user line two."},
    {"role": "system", "content": "Fixture trailing reminder."},
]


def _gguf_module(gguf_py_path: str | None):
    path = gguf_py_path or os.environ.get("GGUF_PY_PATH")
    if path and path not in sys.path:
        sys.path.insert(0, path)
    import gguf  # noqa: WPS433
    if not hasattr(gguf, "GGUFReader"):
        raise TemplateError("imported a 'gguf' module without GGUFReader; set gguf_py_path to llama.cpp's gguf-py")
    return gguf


def _field_str(reader, key: str) -> str | None:
    f = reader.fields.get(key)
    if f is None:
        return None
    return bytes(f.parts[f.data[0]]).decode("utf-8")


def _token_text(reader, id_key: str) -> str:
    f = reader.fields.get(id_key)
    if f is None:
        return ""
    tid = int(f.parts[f.data[0]][0])
    toks = reader.fields.get("tokenizer.ggml.tokens")
    if toks is None:
        return ""
    return bytes(toks.parts[toks.data[tid]]).decode("utf-8", errors="replace")


def read_template_from_gguf(path: str | Path, gguf_py_path: str | None = None) -> ChatTemplate:
    """Read a GGUF file and extract its chat template and token strings."""
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
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=False, keep_trailing_newline=True)
    day = _dt.datetime.strptime(now, "%Y-%m-%d")

    def raise_exception(msg):
        raise TemplateError(str(msg))

    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: day.strftime(fmt)
    return env


def render(tpl: ChatTemplate, messages: list[dict], *, add_generation_prompt: bool = True,
           now: str = "2026-09-08", enable_thinking: bool = False) -> str:
    """Render a chat template with jinja2 using the given messages and options."""
    try:
        return _env(now).from_string(tpl.source).render(
            messages=messages, add_generation_prompt=add_generation_prompt,
            bos_token=tpl.bos, eos_token=tpl.eos, tools=None, enable_thinking=enable_thinking)
    except TemplateError:
        raise
    except jinja2.TemplateError as e:
        raise TemplateError(f"template render failed: {e}") from e


def parity_check(tpl: ChatTemplate, client, messages: list[dict], *, now: str) -> tuple[bool, str, str]:
    """Compare our template rendering to the server's; returns (ok, ours, theirs)."""
    ours = render(tpl, messages, now=now)
    theirs = client.apply_template(messages)
    return ours == theirs, ours, theirs
