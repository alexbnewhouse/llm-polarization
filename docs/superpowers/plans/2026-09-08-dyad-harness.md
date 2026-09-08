# Dyad Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Python package `harness/` that runs seeker/mentor LLM dialogues against two llama-server endpoints with egocentric context projection, persona modes and slot pinning, logs every generation with provenance, administers the mentor's pre/post surveys, and scores seeker adherence offline with a judge model.

**Architecture:** Seven focused modules (`log`, `transcript`, `templates`, `client`, `dialogue`, `survey`, `scorer`) plus a CLI (`run.py`). The harness renders each model's chat template itself (jinja2 + template read from the GGUF) and posts raw prompts to llama-server `/completion` with `id_slot` and `cache_prompt`. Everything is append-only JSONL under `data/<run_id>/`. A `FakeClient` drives all unit tests; live tests are opt-in via an environment variable.

**Tech Stack:** Python 3.11+ (3.12 on both boxes), stdlib (`urllib`, `json`, `threading`, `argparse`, `hashlib`, `dataclasses`), `jinja2` (3.1.x, installed on both boxes), llama.cpp `gguf-py` (vendored checkout: `~/llm-serving/llama.cpp/gguf-py` on the desktop, `~/.local/llamacpp/src/gguf-py` on the Framework Desktop), `pytest` 9.

**Spec:** `docs/superpowers/specs/2026-09-08-dyad-harness-design.md`

## Global Constraints

- Python 3.11+; only `jinja2` outside the standard library; `gguf-py` imported from a configured path (never pip-installed).
- Every request to a llama-server carries `id_slot` (int, the worker's slot) and `cache_prompt: true`; survey and judge calls add `json_schema`.
- Log files are append-only JSONL; nothing rewrites an existing row. Keys join on `run_id, dyad_id, attempt, turn`.
- Terms in code and docs: `seeker`, `mentor`, `dyad`. Turn = one exchange (seeker message, then mentor message), 1-based.
- Seeds: `derive_seed(run_seed, dyad_id, attempt, turn, agent)` = first 8 hex chars of SHA-256 of `f"{run_seed}|{dyad_id}|{attempt}|{turn}|{agent}"` as an int.
- Generation defaults: `temperature 0.7`, `top_p 0.95`, `n_predict 300`, timeout 600 s. Surveys and judge: `temperature 0`, `n_predict 32` (survey) / `160` (judge).
- Mentor gets no system prompt and no reminder, ever. Reminder is a trailing `system` message on seeker turns in `reinforced` mode only.
- Tests run from the repo root with `python -m pytest harness/tests -q`. Commit after every task with the attribution trailer already in use in this repo.

## File structure

```
harness/
  __init__.py            package marker, __version__
  log.py                 paths, JsonlWriter, read_jsonl, manifest, resume index, seeds, hashing, clock
  transcript.py          Message, Transcript, egocentric view_for()
  templates.py           ChatTemplate, read from GGUF, jinja2 render, parity_check
  client.py              LlamaClient (health, props, apply_template, tokenize, complete), Completion, ServerError
  dialogue.py            AgentHandle, GenSettings, DyadSpec, DialogueRunner, expected_new_tokens
  survey.py              load_batteries, answer_schema, parse_answer, SurveyRunner
  scorer.py              metrics, build_judge_messages, score_schema, select_targets, Scorer
  run.py                 config, build_agent, check/run/survey/score subcommands, main(argv)
  config.example.json    two-server config for the Framework Desktop
  requirements.txt       jinja2>=3.1
  tests/
    conftest.py          fixtures: tmp run dirs, fake client, fixture templates
    fakes.py             FakeClient
    test_log.py test_transcript.py test_templates.py test_client.py test_dialogue.py
    test_survey.py test_scorer.py test_run.py test_live.py
instruments/batteries.json   13 placeholder items
```

Data files per run (`data/<run_id>/`): `manifest.json`, `dyads.jsonl` (one row per attempt with the DyadSpec; the scorer reads persona and topic from here), `status.jsonl`, `turns.jsonl`, `surveys.jsonl`, `scores.jsonl`. `dyads.jsonl` is an addition to the spec's section 3 made in this plan; add it to the spec in Task 9.

---

### Task 1: Package skeleton and `log.py`

**Files:**
- Create: `harness/__init__.py`, `harness/log.py`, `harness/requirements.txt`, `harness/tests/__init__.py`, `harness/tests/test_log.py`
- Modify: `.gitignore` (add `harness/.pytest_cache/`)

**Interfaces:**
- Produces:
  - `RunPaths(root, manifest, dyads, status, turns, surveys, scores)` dataclass of `pathlib.Path`; `run_paths(data_dir: str | Path, run_id: str) -> RunPaths` (creates `root`).
  - `class JsonlWriter: __init__(path: Path); write(obj: dict) -> None` (append one line, flush, thread-safe).
  - `read_jsonl(path: Path) -> list[dict]` (empty list if missing).
  - `sha256_text(s: str) -> str`; `sha256_file(path: Path) -> str` (streamed, 1 MiB chunks).
  - `derive_seed(run_seed: int, dyad_id: str, attempt: int, turn: int, agent: str) -> int`.
  - `now_iso() -> str` (local time, seconds, with offset).
  - `class ManifestMismatch(Exception)`; `write_manifest(paths: RunPaths, manifest: dict) -> None` (writes if absent; if present and `manifest["config"]` differs, raises; if same, no-op).
  - `resume_index(status_rows: list[dict]) -> dict[str, dict]` mapping `dyad_id -> {"attempt": int, "status": str}` for the highest attempt seen.
  - `next_attempt(index: dict, dyad_id: str) -> int | None` (None when latest is `complete`, else latest+1 or 1).

- [ ] **Step 1: Write the failing tests**

```python
# harness/tests/test_log.py
import json, threading
from pathlib import Path
import pytest
from harness import log


def test_run_paths_creates_root(tmp_path):
    p = log.run_paths(tmp_path, "r1")
    assert p.root == tmp_path / "r1" and p.root.is_dir()
    assert p.turns.name == "turns.jsonl" and p.manifest.name == "manifest.json"
    assert p.dyads.name == "dyads.jsonl" and p.scores.name == "scores.jsonl"


def test_jsonl_writer_appends_and_reads_back(tmp_path):
    w = log.JsonlWriter(tmp_path / "t.jsonl")
    w.write({"a": 1}); w.write({"b": "x"})
    assert log.read_jsonl(tmp_path / "t.jsonl") == [{"a": 1}, {"b": "x"}]
    assert log.read_jsonl(tmp_path / "missing.jsonl") == []


def test_jsonl_writer_is_thread_safe(tmp_path):
    w = log.JsonlWriter(tmp_path / "t.jsonl")
    def work(i):
        for j in range(50):
            w.write({"i": i, "j": j})
    ths = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    [t.start() for t in ths]; [t.join() for t in ths]
    rows = log.read_jsonl(tmp_path / "t.jsonl")
    assert len(rows) == 400 and all(set(r) == {"i", "j"} for r in rows)


def test_derive_seed_is_stable_and_distinct():
    a = log.derive_seed(7, "d1", 1, 3, "seeker")
    assert a == log.derive_seed(7, "d1", 1, 3, "seeker")
    assert a != log.derive_seed(7, "d1", 1, 3, "mentor")
    assert a != log.derive_seed(7, "d1", 1, 4, "seeker")
    assert a != log.derive_seed(7, "d1", 2, 3, "seeker")
    assert 0 <= a < 2 ** 32


def test_sha256_helpers(tmp_path):
    assert log.sha256_text("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    f = tmp_path / "f.bin"; f.write_bytes(b"abc")
    assert log.sha256_file(f) == log.sha256_text("abc")


def test_write_manifest_refuses_changed_config(tmp_path):
    p = log.run_paths(tmp_path, "r1")
    log.write_manifest(p, {"run_id": "r1", "config": {"x": 1}})
    log.write_manifest(p, {"run_id": "r1", "config": {"x": 1}})   # same config: no-op
    with pytest.raises(log.ManifestMismatch):
        log.write_manifest(p, {"run_id": "r1", "config": {"x": 2}})
    assert json.loads(p.manifest.read_text())["config"] == {"x": 1}


def test_resume_index_and_next_attempt():
    rows = [
        {"dyad_id": "a", "attempt": 1, "status": "started"},
        {"dyad_id": "a", "attempt": 1, "status": "failed"},
        {"dyad_id": "a", "attempt": 2, "status": "started"},
        {"dyad_id": "b", "attempt": 1, "status": "started"},
        {"dyad_id": "b", "attempt": 1, "status": "complete"},
    ]
    idx = log.resume_index(rows)
    assert idx == {"a": {"attempt": 2, "status": "started"}, "b": {"attempt": 1, "status": "complete"}}
    assert log.next_attempt(idx, "a") == 3
    assert log.next_attempt(idx, "b") is None
    assert log.next_attempt(idx, "c") == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest harness/tests/test_log.py -q`
Expected: ImportError / ModuleNotFoundError for `harness.log`.

- [ ] **Step 3: Implement**

```python
# harness/__init__.py
"""Dyad harness for the LLM political polarization study."""
__version__ = "0.1.0"
```

```
# harness/requirements.txt
jinja2>=3.1
```

```python
# harness/log.py
"""Append-only JSONL logging, run manifest, resume index, seeds and hashing."""
from __future__ import annotations
import hashlib, json, threading, time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    root: Path
    manifest: Path
    dyads: Path
    status: Path
    turns: Path
    surveys: Path
    scores: Path


def run_paths(data_dir: str | Path, run_id: str) -> RunPaths:
    root = Path(data_dir) / run_id
    root.mkdir(parents=True, exist_ok=True)
    return RunPaths(root, root / "manifest.json", root / "dyads.jsonl", root / "status.jsonl",
                    root / "turns.jsonl", root / "surveys.jsonl", root / "scores.jsonl")


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def derive_seed(run_seed: int, dyad_id: str, attempt: int, turn: int, agent: str) -> int:
    return int(sha256_text(f"{run_seed}|{dyad_id}|{attempt}|{turn}|{agent}")[:8], 16)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class ManifestMismatch(Exception):
    pass


def write_manifest(paths: RunPaths, manifest: dict) -> None:
    if paths.manifest.exists():
        existing = json.loads(paths.manifest.read_text(encoding="utf-8"))
        if existing.get("config") != manifest.get("config"):
            raise ManifestMismatch(f"{paths.manifest} exists with a different config; use a new run_id")
        return
    paths.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def resume_index(status_rows: list[dict]) -> dict[str, dict]:
    idx: dict[str, dict] = {}
    for r in status_rows:
        cur = idx.get(r["dyad_id"])
        if cur is None or r["attempt"] >= cur["attempt"]:
            idx[r["dyad_id"]] = {"attempt": r["attempt"], "status": r["status"]}
    return idx


def next_attempt(index: dict, dyad_id: str) -> int | None:
    cur = index.get(dyad_id)
    if cur is None:
        return 1
    if cur["status"] == "complete":
        return None
    return cur["attempt"] + 1
```

Create empty `harness/tests/__init__.py`. Append `harness/.pytest_cache/` and `.pytest_cache/` to `.gitignore`.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest harness/tests/test_log.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add harness/__init__.py harness/log.py harness/requirements.txt harness/tests/__init__.py harness/tests/test_log.py .gitignore
git commit -m "harness: package skeleton and append-only log module"
```

---

### Task 2: `transcript.py` (canonical transcript and egocentric projection)

**Files:**
- Create: `harness/transcript.py`, `harness/tests/test_transcript.py`

**Interfaces:**
- Produces:
  - `SEEKER = "seeker"`, `MENTOR = "mentor"`, `AGENTS = (SEEKER, MENTOR)`; `partner_of(agent: str) -> str`.
  - `@dataclass Message(turn: int, agent: str, text: str)`.
  - `class Transcript: __init__(dyad_id: str, seeker_system: str, reminder: str | None, persona_mode: str)`; attributes `dyad_id, seeker_system, reminder, persona_mode, messages: list[Message]`; `append(turn, agent, text) -> Message`; `view_for(agent) -> list[dict]` (list of `{"role","content"}`); `lines_of(agent) -> list[str]`; `last_line_of(agent) -> str | None`; `n_messages` property.
  - `persona_mode` is one of `"once"`, `"reinforced"`; anything else raises `ValueError`.

- [ ] **Step 1: Write the failing tests**

```python
# harness/tests/test_transcript.py
import pytest
from harness.transcript import Transcript, SEEKER, MENTOR, partner_of


def make(mode="once", reminder="Remember: you are Dana."):
    t = Transcript("d1", "You are Dana, a rancher.", reminder, mode)
    t.append(1, SEEKER, "I need advice.")
    t.append(1, MENTOR, "Tell me more.")
    t.append(2, SEEKER, "The troops worry me.")
    return t


def test_partner_of():
    assert partner_of(SEEKER) == MENTOR and partner_of(MENTOR) == SEEKER


def test_persona_mode_validated():
    with pytest.raises(ValueError):
        Transcript("d1", "sys", None, "sometimes")


def test_seeker_view_once_mode():
    v = make("once").view_for(SEEKER)
    assert v == [
        {"role": "system", "content": "You are Dana, a rancher."},
        {"role": "assistant", "content": "I need advice."},
        {"role": "user", "content": "Tell me more."},
        {"role": "assistant", "content": "The troops worry me."},
    ]


def test_seeker_view_reinforced_appends_trailing_system_reminder():
    v = make("reinforced").view_for(SEEKER)
    assert v[0] == {"role": "system", "content": "You are Dana, a rancher."}
    assert v[-1] == {"role": "system", "content": "Remember: you are Dana."}
    assert [m["role"] for m in v] == ["system", "assistant", "user", "assistant", "system"]


def test_mentor_view_has_no_system_and_no_reminder_in_any_mode():
    for mode in ("once", "reinforced"):
        v = make(mode).view_for(MENTOR)
        assert v == [
            {"role": "user", "content": "I need advice."},
            {"role": "assistant", "content": "Tell me more."},
            {"role": "user", "content": "The troops worry me."},
        ]


def test_empty_seeker_view_is_system_only_and_reminder_not_duplicated():
    t = Transcript("d1", "sys", "rem", "reinforced")
    assert t.view_for(SEEKER) == [{"role": "system", "content": "sys"}, {"role": "system", "content": "rem"}]
    assert t.view_for(MENTOR) == []


def test_no_speaker_labels_in_content():
    for m in make().view_for(SEEKER) + make().view_for(MENTOR):
        assert "seeker:" not in m["content"].lower() and "mentor:" not in m["content"].lower()


def test_lines_and_last_line():
    t = make()
    assert t.lines_of(SEEKER) == ["I need advice.", "The troops worry me."]
    assert t.last_line_of(MENTOR) == "Tell me more."
    assert Transcript("d", "s", None, "once").last_line_of(SEEKER) is None
    assert t.n_messages == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest harness/tests/test_transcript.py -q`
Expected: ModuleNotFoundError for `harness.transcript`.

- [ ] **Step 3: Implement**

```python
# harness/transcript.py
"""Canonical transcript of one dyad and the egocentric projection each agent generates from."""
from __future__ import annotations
from dataclasses import dataclass, field

SEEKER = "seeker"
MENTOR = "mentor"
AGENTS = (SEEKER, MENTOR)
PERSONA_MODES = ("once", "reinforced")


def partner_of(agent: str) -> str:
    return MENTOR if agent == SEEKER else SEEKER


@dataclass
class Message:
    turn: int
    agent: str
    text: str


@dataclass
class Transcript:
    dyad_id: str
    seeker_system: str
    reminder: str | None
    persona_mode: str
    messages: list[Message] = field(default_factory=list)

    def __post_init__(self):
        if self.persona_mode not in PERSONA_MODES:
            raise ValueError(f"persona_mode must be one of {PERSONA_MODES}, got {self.persona_mode!r}")

    def append(self, turn: int, agent: str, text: str) -> Message:
        m = Message(turn, agent, text)
        self.messages.append(m)
        return m

    def view_for(self, agent: str) -> list[dict]:
        """Own lines as assistant, partner lines as user. Seeker gets its system prompt (and the
        reminder last, in reinforced mode); the mentor gets nothing beyond the history."""
        view: list[dict] = []
        if agent == SEEKER:
            view.append({"role": "system", "content": self.seeker_system})
        for m in self.messages:
            role = "assistant" if m.agent == agent else "user"
            view.append({"role": role, "content": m.text})
        if agent == SEEKER and self.persona_mode == "reinforced" and self.reminder:
            view.append({"role": "system", "content": self.reminder})
        return view

    def lines_of(self, agent: str) -> list[str]:
        return [m.text for m in self.messages if m.agent == agent]

    def last_line_of(self, agent: str) -> str | None:
        lines = self.lines_of(agent)
        return lines[-1] if lines else None

    @property
    def n_messages(self) -> int:
        return len(self.messages)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest harness/tests/test_transcript.py -q`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add harness/transcript.py harness/tests/test_transcript.py
git commit -m "harness: transcript with egocentric projection and persona modes"
```

---

### Task 3: `templates.py` (chat template rendering and parity check)

**Files:**
- Create: `harness/templates.py`, `harness/tests/test_templates.py`, `harness/tests/conftest.py`

**Interfaces:**
- Produces:
  - `@dataclass ChatTemplate(source: str, bos: str, eos: str, sha256: str)`; `ChatTemplate.from_source(source, bos="", eos="") -> ChatTemplate`.
  - `read_template_from_gguf(path: str | Path, gguf_py_path: str | None = None) -> ChatTemplate` (imports `gguf` from `gguf_py_path` or env `GGUF_PY_PATH`; raises `TemplateError` if the key is missing).
  - `render(tpl: ChatTemplate, messages: list[dict], *, add_generation_prompt=True, now="2026-09-08", enable_thinking=False) -> str`.
  - `parity_check(tpl: ChatTemplate, client, messages: list[dict], *, now) -> tuple[bool, str, str]` where `client.apply_template(messages) -> str` (returns ok, ours, theirs).
  - `FIXTURE_MESSAGES: list[dict]` (system, user, assistant, user, then a trailing system reminder) used by `run.py check`.
  - `class TemplateError(Exception)`.
- Rendering rules: jinja2 `Environment(trim_blocks=True, lstrip_blocks=False, keep_trailing_newline=True)`, globals `raise_exception` (raises `TemplateError`) and `strftime_now(fmt)` (formats the pinned `now` date), variables `messages, add_generation_prompt, bos_token, eos_token, tools=None, enable_thinking`.

- [ ] **Step 1: Write conftest with fixture templates and the failing tests**

```python
# harness/tests/conftest.py
import pytest

CHATML = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)

# Mirrors the part of Olmo-3's template that broke llama.cpp's parser: it serializes `tools`.
OLMO_LIKE = (
    "{%- if tools is not none -%}<functions>{{ tools | tojson }}</functions>{%- endif -%}"
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)

# Mirrors gpt-oss's use of strftime_now in its system preamble.
DATED = (
    "<|start|>system<|message|>Knowledge cutoff: 2024-06\\nCurrent date: {{ strftime_now('%Y-%m-%d') }}<|end|>"
    "{%- for message in messages %}<|start|>{{ message['role'] }}<|message|>{{ message['content'] }}<|end|>{%- endfor %}"
    "{%- if add_generation_prompt %}<|start|>assistant{%- endif %}"
)

REJECTS_TRAILING_SYSTEM = (
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' and not loop.first %}{{ raise_exception('system must be first') }}{%- endif %}"
    "{{- message['role'] + ': ' + message['content'] + '\\n' }}{%- endfor %}"
)


@pytest.fixture
def chatml():
    return CHATML


@pytest.fixture
def olmo_like():
    return OLMO_LIKE


@pytest.fixture
def dated():
    return DATED


@pytest.fixture
def rejects_trailing_system():
    return REJECTS_TRAILING_SYSTEM
```

```python
# harness/tests/test_templates.py
import pytest
from harness import templates
from harness.templates import ChatTemplate, render, TemplateError

MSGS = [
    {"role": "system", "content": "You are Dana."},
    {"role": "user", "content": "Advice?"},
    {"role": "assistant", "content": "Tell me more."},
    {"role": "user", "content": "The troops."},
]


def test_render_chatml(chatml):
    tpl = ChatTemplate.from_source(chatml)
    out = render(tpl, MSGS)
    assert out == (
        "<|im_start|>system\nYou are Dana.<|im_end|>\n<|im_start|>user\nAdvice?<|im_end|>\n"
        "<|im_start|>assistant\nTell me more.<|im_end|>\n<|im_start|>user\nThe troops.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert render(tpl, MSGS, add_generation_prompt=False).endswith("The troops.<|im_end|>\n")


def test_render_passes_tools_none_so_olmo_style_templates_work(olmo_like):
    out = render(ChatTemplate.from_source(olmo_like), MSGS)
    assert "<functions>" not in out and out.startswith("<|im_start|>system")


def test_render_pins_strftime_now(dated):
    out = render(ChatTemplate.from_source(dated), MSGS[1:], now="2026-09-08")
    assert "Current date: 2026-09-08" in out
    assert render(ChatTemplate.from_source(dated), MSGS[1:], now="2027-01-01") != out


def test_raise_exception_becomes_template_error(rejects_trailing_system):
    tpl = ChatTemplate.from_source(rejects_trailing_system)
    with pytest.raises(TemplateError):
        render(tpl, MSGS + [{"role": "system", "content": "reminder"}])


def test_sha256_is_of_source(chatml):
    tpl = ChatTemplate.from_source(chatml)
    from harness.log import sha256_text
    assert tpl.sha256 == sha256_text(chatml)


def test_fixture_messages_end_with_trailing_system():
    assert templates.FIXTURE_MESSAGES[0]["role"] == "system"
    assert templates.FIXTURE_MESSAGES[-1]["role"] == "system"
    assert len(templates.FIXTURE_MESSAGES) >= 4


class _Srv:
    def __init__(self, source):
        self.tpl = ChatTemplate.from_source(source)
    def apply_template(self, messages):
        return render(self.tpl, messages)


def test_parity_check_ok_and_mismatch(chatml, dated):
    tpl = ChatTemplate.from_source(chatml)
    ok, ours, theirs = templates.parity_check(tpl, _Srv(chatml), MSGS, now="2026-09-08")
    assert ok and ours == theirs
    ok, ours, theirs = templates.parity_check(tpl, _Srv(dated), MSGS, now="2026-09-08")
    assert not ok and ours != theirs


def test_read_template_from_gguf_missing_path_raises(tmp_path):
    with pytest.raises((TemplateError, OSError, ImportError)):
        templates.read_template_from_gguf(tmp_path / "nope.gguf", gguf_py_path=str(tmp_path))
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest harness/tests/test_templates.py -q`
Expected: ModuleNotFoundError for `harness.templates`.

- [ ] **Step 3: Implement**

```python
# harness/templates.py
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
    pass


@dataclass(frozen=True)
class ChatTemplate:
    source: str
    bos: str
    eos: str
    sha256: str

    @classmethod
    def from_source(cls, source: str, bos: str = "", eos: str = "") -> "ChatTemplate":
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
    try:
        return _env(now).from_string(tpl.source).render(
            messages=messages, add_generation_prompt=add_generation_prompt,
            bos_token=tpl.bos, eos_token=tpl.eos, tools=None, enable_thinking=enable_thinking)
    except TemplateError:
        raise
    except jinja2.TemplateError as e:
        raise TemplateError(f"template render failed: {e}") from e


def parity_check(tpl: ChatTemplate, client, messages: list[dict], *, now: str) -> tuple[bool, str, str]:
    ours = render(tpl, messages, now=now)
    theirs = client.apply_template(messages)
    return ours == theirs, ours, theirs
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest harness/tests/test_templates.py -q`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add harness/templates.py harness/tests/test_templates.py harness/tests/conftest.py
git commit -m "harness: jinja2 chat-template rendering with GGUF source and parity check"
```

---

### Task 4: `client.py` (llama-server HTTP client)

**Files:**
- Create: `harness/client.py`, `harness/tests/test_client.py`

**Interfaces:**
- Produces:
  - `class ServerError(Exception)`.
  - `@dataclass Completion(text: str, finish_reason: str, prompt_n: int, predicted_n: int, timings: dict, raw: dict)`.
  - `class LlamaClient: __init__(url: str, timeout: float = 600)`; `health() -> bool`; `props() -> dict`; `apply_template(messages) -> str`; `tokenize(text) -> int`; `complete(prompt, *, id_slot, seed, n_predict, temperature, top_p=0.95, json_schema=None, cache_prompt=True, stop=None) -> Completion`; `_post(path, body) -> dict` and `_get(path) -> dict` (the seam tests monkeypatch).
  - `parse_completion(raw: dict) -> Completion` (module-level, pure): `finish_reason` is `"length"` when `raw.get("stop_type") == "limit"` or `raw.get("stopped_limit")` is true, else `"stop"`; `prompt_n = timings.prompt_n`, `predicted_n = timings.predicted_n` (0 when absent).

- [ ] **Step 1: Write the failing tests**

```python
# harness/tests/test_client.py
import json
import pytest
from harness import client as C


def test_parse_completion_stop_and_length():
    raw = {"content": "hi", "stop_type": "eos", "timings": {"prompt_n": 5, "predicted_n": 2, "predicted_per_second": 9.5}}
    c = C.parse_completion(raw)
    assert (c.text, c.finish_reason, c.prompt_n, c.predicted_n) == ("hi", "stop", 5, 2)
    assert c.timings["predicted_per_second"] == 9.5 and c.raw is raw
    assert C.parse_completion({"content": "x", "stop_type": "limit", "timings": {}}).finish_reason == "length"
    assert C.parse_completion({"content": "x", "stopped_limit": True}).finish_reason == "length"
    assert C.parse_completion({"content": "x"}).prompt_n == 0


def test_complete_builds_request_body(monkeypatch):
    cl = C.LlamaClient("http://127.0.0.1:1")
    seen = {}
    def fake_post(path, body):
        seen["path"], seen["body"] = path, body
        return {"content": "ok", "stop_type": "eos", "timings": {"prompt_n": 1, "predicted_n": 1}}
    monkeypatch.setattr(cl, "_post", fake_post)
    c = cl.complete("PROMPT", id_slot=3, seed=42, n_predict=300, temperature=0.7,
                    json_schema={"type": "object"}, stop=["<|im_end|>"])
    assert seen["path"] == "/completion" and c.text == "ok"
    b = seen["body"]
    assert b["prompt"] == "PROMPT" and b["id_slot"] == 3 and b["seed"] == 42
    assert b["n_predict"] == 300 and b["temperature"] == 0.7 and b["top_p"] == 0.95
    assert b["cache_prompt"] is True and b["json_schema"] == {"type": "object"} and b["stop"] == ["<|im_end|>"]


def test_complete_omits_json_schema_and_stop_when_absent(monkeypatch):
    cl = C.LlamaClient("http://127.0.0.1:1")
    seen = {}
    monkeypatch.setattr(cl, "_post", lambda p, b: seen.setdefault("body", b) or {"content": ""})
    cl.complete("P", id_slot=0, seed=1, n_predict=5, temperature=0.0)
    assert "json_schema" not in seen["body"] and "stop" not in seen["body"]


def test_tokenize_and_apply_template_and_health(monkeypatch):
    cl = C.LlamaClient("http://127.0.0.1:1")
    monkeypatch.setattr(cl, "_post", lambda p, b: {"tokens": [1, 2, 3]} if p == "/tokenize" else {"prompt": "RENDERED"})
    assert cl.tokenize("a b c") == 3
    assert cl.apply_template([{"role": "user", "content": "x"}]) == "RENDERED"
    monkeypatch.setattr(cl, "_get", lambda p: {"status": "ok"})
    assert cl.health() is True
    def boom(p):
        raise C.ServerError("down")
    monkeypatch.setattr(cl, "_get", boom)
    assert cl.health() is False


def test_post_wraps_network_errors_as_server_error():
    cl = C.LlamaClient("http://127.0.0.1:9", timeout=1)   # nothing listens on port 9
    with pytest.raises(C.ServerError):
        cl._post("/completion", {"prompt": "x"})
    with pytest.raises(C.ServerError):
        cl._get("/health")
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest harness/tests/test_client.py -q`
Expected: ModuleNotFoundError for `harness.client`.

- [ ] **Step 3: Implement**

```python
# harness/client.py
"""Minimal llama-server client. One instance per endpoint. No retries: a failed call is the caller's
problem to log, because a silent retry could hide a lost KV cache."""
from __future__ import annotations
import json
import urllib.error, urllib.request
from dataclasses import dataclass


class ServerError(Exception):
    pass


@dataclass
class Completion:
    text: str
    finish_reason: str
    prompt_n: int
    predicted_n: int
    timings: dict
    raw: dict


def parse_completion(raw: dict) -> Completion:
    timings = raw.get("timings") or {}
    limit = raw.get("stop_type") == "limit" or bool(raw.get("stopped_limit"))
    return Completion(
        text=raw.get("content", ""),
        finish_reason="length" if limit else "stop",
        prompt_n=int(timings.get("prompt_n") or 0),
        predicted_n=int(timings.get("predicted_n") or 0),
        timings=timings,
        raw=raw,
    )


class LlamaClient:
    def __init__(self, url: str, timeout: float = 600):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise ServerError(f"POST {path}: HTTP {e.code} {e.read()[:200]!r}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise ServerError(f"POST {path}: {e}") from e

    def _get(self, path: str) -> dict:
        try:
            with urllib.request.urlopen(self.url + path, timeout=min(self.timeout, 10)) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise ServerError(f"GET {path}: HTTP {e.code}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise ServerError(f"GET {path}: {e}") from e

    def health(self) -> bool:
        try:
            return self._get("/health").get("status") == "ok"
        except ServerError:
            return False

    def props(self) -> dict:
        return self._get("/props")

    def apply_template(self, messages: list[dict]) -> str:
        return self._post("/apply-template", {"messages": messages})["prompt"]

    def tokenize(self, text: str) -> int:
        return len(self._post("/tokenize", {"content": text, "add_special": False}).get("tokens", []))

    def complete(self, prompt: str, *, id_slot: int, seed: int, n_predict: int, temperature: float,
                 top_p: float = 0.95, json_schema: dict | None = None, cache_prompt: bool = True,
                 stop: list[str] | None = None) -> Completion:
        body = {"prompt": prompt, "id_slot": id_slot, "seed": seed, "n_predict": n_predict,
                "temperature": temperature, "top_p": top_p, "cache_prompt": cache_prompt}
        if json_schema is not None:
            body["json_schema"] = json_schema
        if stop:
            body["stop"] = stop
        return parse_completion(self._post("/completion", body))
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest harness/tests/test_client.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add harness/client.py harness/tests/test_client.py
git commit -m "harness: llama-server client with slot pinning and json_schema"
```

---

### Task 5: `dialogue.py` (the turn loop) with a `FakeClient`

**Files:**
- Create: `harness/dialogue.py`, `harness/tests/fakes.py`, `harness/tests/test_dialogue.py`

**Interfaces:**
- Consumes: `Transcript`, `SEEKER/MENTOR`, `ChatTemplate`, `render`, `Completion`, `ServerError`, `JsonlWriter`, `derive_seed`, `sha256_text`, `now_iso`.
- Produces:
  - `@dataclass AgentHandle(name: str, client, template: ChatTemplate, model_sha256: str, slot: int, alias: str = "")`.
  - `@dataclass GenSettings(temperature=0.7, top_p=0.95, n_predict=300, now="2026-09-08", enable_thinking=False)`.
  - `@dataclass DyadSpec(dyad_id, condition: dict, persona_text, persona_reminder, persona_mode, seed: int, n_turns: int)`; `DyadSpec.from_row(row: dict) -> DyadSpec`.
  - `class DialogueError(Exception)` with attributes `dyad_id, turn, agent`.
  - `expected_new_tokens(client, prompt: str, previous_prompt: str | None) -> int` (tokens of `prompt` minus tokens of the longest common string prefix with `previous_prompt`; whole prompt when there is no previous).
  - `class DialogueRunner: __init__(run_id, run_seed, seeker: AgentHandle, mentor: AgentHandle, settings: GenSettings, turns_log: JsonlWriter, clock=now_iso, cache_margin=64)`; `run(spec: DyadSpec, attempt: int) -> Transcript`.
  - Row schema written to `turns_log` per message: `run_id, dyad_id, attempt, turn, agent, model_sha256, persona_mode, prompt_sha256, prompt_chars, prompt_n, predicted_n, expected_new, cache_warning, finish_reason, text, seed, timings, adherence: None, ts`. On `ServerError`: same keys with `finish_reason="error"`, `text=""`, `error=str(e)`, then `DialogueError` is raised.
- `FakeClient(replies: list[str] | None = None, fail_on: int | None = None)`: records every `complete` call's kwargs in `.calls`; `tokenize` returns `len(text.split())`; `apply_template` renders the ChatML fixture; `complete` returns replies in order (cycling) with `timings.prompt_n` = number of whitespace tokens in the prompt that follow the longest common prefix with the previous prompt on that slot (simulating a working cache), `predicted_n = len(reply.split())`; raises `ServerError` on call number `fail_on` (1-based).

- [ ] **Step 1: Write the fake and the failing tests**

```python
# harness/tests/fakes.py
from harness.client import Completion, ServerError
from harness.templates import ChatTemplate, render
from harness.tests.conftest import CHATML


class FakeClient:
    def __init__(self, replies=None, fail_on=None):
        self.replies = replies or ["reply"]
        self.fail_on = fail_on
        self.calls = []
        self.tpl = ChatTemplate.from_source(CHATML)
        self._last_prompt = {}

    def health(self):
        return True

    def props(self):
        return {"model_path": "/fake/model.gguf", "total_slots": 4, "build_info": "fake", "model_alias": "fake",
                "default_generation_settings": {}}

    def apply_template(self, messages):
        return render(self.tpl, messages)

    def tokenize(self, text):
        return len(text.split())

    def complete(self, prompt, **kw):
        self.calls.append({"prompt": prompt, **kw})
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise ServerError("fake failure")
        slot = kw["id_slot"]
        prev = self._last_prompt.get(slot)
        common = 0
        if prev:
            while common < min(len(prev), len(prompt)) and prev[common] == prompt[common]:
                common += 1
        prompt_n = len(prompt[common:].split()) if prev else len(prompt.split())
        self._last_prompt[slot] = prompt
        reply = self.replies[(len(self.calls) - 1) % len(self.replies)]
        return Completion(text=reply, finish_reason="stop", prompt_n=prompt_n,
                          predicted_n=len(reply.split()),
                          timings={"prompt_n": prompt_n, "predicted_n": len(reply.split())}, raw={})
```

```python
# harness/tests/test_dialogue.py
import pytest
from harness import log
from harness.dialogue import AgentHandle, GenSettings, DyadSpec, DialogueRunner, DialogueError, expected_new_tokens
from harness.templates import ChatTemplate
from harness.transcript import SEEKER, MENTOR
from harness.tests.fakes import FakeClient
from harness.tests.conftest import CHATML


def make_runner(tmp_path, seeker_replies=None, mentor_replies=None, fail_on=None):
    tpl = ChatTemplate.from_source(CHATML)
    sc = FakeClient(seeker_replies or ["I need advice about the border."], fail_on=fail_on)
    mc = FakeClient(mentor_replies or ["Tell me more."])
    seeker = AgentHandle(SEEKER, sc, tpl, "seekerhash", slot=2, alias="s")
    mentor = AgentHandle(MENTOR, mc, tpl, "mentorhash", slot=2, alias="m")
    w = log.JsonlWriter(tmp_path / "turns.jsonl")
    return DialogueRunner("run1", 99, seeker, mentor, GenSettings(), w, clock=lambda: "T"), sc, mc, w


def spec(mode="reinforced", n_turns=3):
    return DyadSpec("d1", {"topic": "immigration_enforcement"}, "You are Dana.", "Note to self: Dana.", mode, 5, n_turns)


def test_dyad_spec_from_row():
    s = DyadSpec.from_row({"dyad_id": "x", "condition": {"topic": "t"}, "persona_text": "p",
                           "persona_reminder": "r", "persona_mode": "once", "seed": 1, "n_turns": 2})
    assert s.dyad_id == "x" and s.n_turns == 2 and s.persona_mode == "once"


def test_turn_order_and_row_count(tmp_path):
    runner, sc, mc, w = make_runner(tmp_path)
    t = runner.run(spec(n_turns=3), attempt=1)
    rows = log.read_jsonl(tmp_path / "turns.jsonl")
    assert [(r["turn"], r["agent"]) for r in rows] == [(1, SEEKER), (1, MENTOR), (2, SEEKER), (2, MENTOR), (3, SEEKER), (3, MENTOR)]
    assert t.n_messages == 6 and len(sc.calls) == 3 and len(mc.calls) == 3
    assert all(r["run_id"] == "run1" and r["attempt"] == 1 and r["adherence"] is None and r["ts"] == "T" for r in rows)


def test_every_request_is_pinned_and_cached(tmp_path):
    runner, sc, mc, _ = make_runner(tmp_path)
    runner.run(spec(), attempt=1)
    for c in sc.calls + mc.calls:
        assert c["id_slot"] == 2 and c["cache_prompt"] is True
        assert c["n_predict"] == 300 and c["temperature"] == 0.7 and c["top_p"] == 0.95


def test_seeker_opens_from_system_only_and_reminder_placement(tmp_path):
    runner, sc, mc, _ = make_runner(tmp_path)
    runner.run(spec("reinforced", n_turns=2), attempt=1)
    first = sc.calls[0]["prompt"]
    assert first.startswith("<|im_start|>system\nYou are Dana.<|im_end|>\n")
    assert "<|im_start|>user" not in first.split("<|im_start|>system\nNote to self")[0]
    assert first.endswith("<|im_start|>system\nNote to self: Dana.<|im_end|>\n<|im_start|>assistant\n")
    second = sc.calls[1]["prompt"]
    assert second.endswith("<|im_start|>system\nNote to self: Dana.<|im_end|>\n<|im_start|>assistant\n")
    assert second.count("Note to self") == 1
    for c in mc.calls:
        assert "<|im_start|>system" not in c["prompt"] and "Note to self" not in c["prompt"]


def test_once_mode_has_no_reminder(tmp_path):
    runner, sc, _, _ = make_runner(tmp_path)
    runner.run(spec("once", n_turns=2), attempt=1)
    assert all("Note to self" not in c["prompt"] for c in sc.calls)


def test_mentor_sees_seeker_lines_as_user(tmp_path):
    runner, sc, mc, _ = make_runner(tmp_path, seeker_replies=["SEEKER-LINE"], mentor_replies=["MENTOR-LINE"])
    runner.run(spec(n_turns=2), attempt=1)
    p = mc.calls[1]["prompt"]
    assert "<|im_start|>user\nSEEKER-LINE<|im_end|>" in p and "<|im_start|>assistant\nMENTOR-LINE<|im_end|>" in p


def test_seeds_distinct_and_logged(tmp_path):
    runner, sc, mc, _ = make_runner(tmp_path)
    runner.run(spec(n_turns=2), attempt=1)
    rows = log.read_jsonl(tmp_path / "turns.jsonl")
    seeds = [r["seed"] for r in rows]
    assert len(set(seeds)) == 4
    assert rows[0]["seed"] == log.derive_seed(99, "d1", 1, 1, SEEKER)
    assert [c["seed"] for c in sc.calls] == [rows[0]["seed"], rows[2]["seed"]]


def test_provenance_fields(tmp_path):
    runner, sc, _, _ = make_runner(tmp_path)
    runner.run(spec(n_turns=1), attempt=2)
    r = log.read_jsonl(tmp_path / "turns.jsonl")[0]
    assert r["model_sha256"] == "seekerhash" and r["persona_mode"] == "reinforced"
    assert r["prompt_sha256"] == log.sha256_text(sc.calls[0]["prompt"]) and r["prompt_chars"] == len(sc.calls[0]["prompt"])
    assert r["finish_reason"] == "stop" and r["attempt"] == 2 and r["predicted_n"] > 0


def test_cache_warning_is_false_when_cache_holds(tmp_path):
    runner, _, _, _ = make_runner(tmp_path)
    runner.run(spec("reinforced", n_turns=3), attempt=1)
    rows = log.read_jsonl(tmp_path / "turns.jsonl")
    assert all(r["cache_warning"] is False for r in rows)
    assert all(r["prompt_n"] <= r["expected_new"] + 64 for r in rows)


def test_cache_warning_true_when_server_reprefills_everything(tmp_path):
    runner, sc, mc, _ = make_runner(tmp_path)
    def cold_complete(prompt, **kw):
        c = FakeClient.complete(sc, prompt, **kw)
        c.prompt_n = len(prompt.split()) + 500   # pretend the cache was lost
        return c
    sc.complete = cold_complete
    runner.run(spec(n_turns=2), attempt=1)
    rows = [r for r in log.read_jsonl(tmp_path / "turns.jsonl") if r["agent"] == SEEKER]
    assert rows[1]["cache_warning"] is True


def test_expected_new_tokens():
    fc = FakeClient()
    assert expected_new_tokens(fc, "a b c d", None) == 4
    assert expected_new_tokens(fc, "a b c d e f", "a b c d") == 2
    assert expected_new_tokens(fc, "x y", "a b") == 2


def test_server_error_writes_error_row_and_raises(tmp_path):
    runner, sc, mc, _ = make_runner(tmp_path, fail_on=2)   # second seeker call fails
    with pytest.raises(DialogueError) as ei:
        runner.run(spec(n_turns=3), attempt=1)
    assert ei.value.turn == 2 and ei.value.agent == SEEKER and ei.value.dyad_id == "d1"
    rows = log.read_jsonl(tmp_path / "turns.jsonl")
    assert rows[-1]["finish_reason"] == "error" and rows[-1]["text"] == "" and "fake failure" in rows[-1]["error"]
    assert len(rows) == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest harness/tests/test_dialogue.py -q`
Expected: ModuleNotFoundError for `harness.dialogue`.

- [ ] **Step 3: Implement**

```python
# harness/dialogue.py
"""One dyad, end to end: seeker opens, agents alternate, every generation logged with provenance."""
from __future__ import annotations
from dataclasses import dataclass, field
from harness.client import ServerError
from harness.log import JsonlWriter, derive_seed, now_iso, sha256_text
from harness.templates import ChatTemplate, render
from harness.transcript import Transcript, SEEKER, MENTOR


@dataclass
class AgentHandle:
    name: str
    client: object
    template: ChatTemplate
    model_sha256: str
    slot: int
    alias: str = ""


@dataclass
class GenSettings:
    temperature: float = 0.7
    top_p: float = 0.95
    n_predict: int = 300
    now: str = "2026-09-08"
    enable_thinking: bool = False


@dataclass
class DyadSpec:
    dyad_id: str
    condition: dict
    persona_text: str
    persona_reminder: str
    persona_mode: str
    seed: int
    n_turns: int

    @classmethod
    def from_row(cls, row: dict) -> "DyadSpec":
        return cls(row["dyad_id"], dict(row.get("condition") or {}), row["persona_text"],
                   row.get("persona_reminder", ""), row.get("persona_mode", "reinforced"),
                   int(row.get("seed", 0)), int(row["n_turns"]))


class DialogueError(Exception):
    def __init__(self, dyad_id: str, turn: int, agent: str, cause: Exception):
        super().__init__(f"{dyad_id} turn {turn} {agent}: {cause}")
        self.dyad_id, self.turn, self.agent, self.cause = dyad_id, turn, agent, cause


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def expected_new_tokens(client, prompt: str, previous_prompt: str | None) -> int:
    """Tokens the server should have to prefill if its slot still holds previous_prompt."""
    if not previous_prompt:
        return client.tokenize(prompt)
    k = _common_prefix_len(prompt, previous_prompt)
    return client.tokenize(prompt) - client.tokenize(prompt[:k])


class DialogueRunner:
    def __init__(self, run_id: str, run_seed: int, seeker: AgentHandle, mentor: AgentHandle,
                 settings: GenSettings, turns_log: JsonlWriter, clock=now_iso, cache_margin: int = 64):
        self.run_id, self.run_seed = run_id, run_seed
        self.agents = {SEEKER: seeker, MENTOR: mentor}
        self.settings, self.turns_log, self.clock, self.cache_margin = settings, turns_log, clock, cache_margin

    def run(self, spec: DyadSpec, attempt: int) -> Transcript:
        transcript = Transcript(spec.dyad_id, spec.persona_text, spec.persona_reminder or None, spec.persona_mode)
        last_prompt: dict[str, str | None] = {SEEKER: None, MENTOR: None}
        for turn in range(1, spec.n_turns + 1):
            for agent in (SEEKER, MENTOR):
                text = self._generate(agent, transcript, spec, attempt, turn, last_prompt)
                transcript.append(turn, agent, text)
        return transcript

    def _generate(self, agent: str, transcript: Transcript, spec: DyadSpec, attempt: int, turn: int,
                  last_prompt: dict) -> str:
        h = self.agents[agent]
        s = self.settings
        prompt = render(h.template, transcript.view_for(agent), now=s.now, enable_thinking=s.enable_thinking)
        seed = derive_seed(self.run_seed, spec.dyad_id, attempt, turn, agent)
        row = {"run_id": self.run_id, "dyad_id": spec.dyad_id, "attempt": attempt, "turn": turn, "agent": agent,
               "model_sha256": h.model_sha256, "persona_mode": spec.persona_mode,
               "prompt_sha256": sha256_text(prompt), "prompt_chars": len(prompt), "seed": seed}
        try:
            expected = expected_new_tokens(h.client, prompt, last_prompt[agent])
            comp = h.client.complete(prompt, id_slot=h.slot, seed=seed, n_predict=s.n_predict,
                                     temperature=s.temperature, top_p=s.top_p)
        except ServerError as e:
            row.update({"prompt_n": None, "predicted_n": None, "expected_new": None, "cache_warning": None,
                        "finish_reason": "error", "text": "", "timings": {}, "error": str(e),
                        "adherence": None, "ts": self.clock()})
            self.turns_log.write(row)
            raise DialogueError(spec.dyad_id, turn, agent, e) from e
        last_prompt[agent] = prompt
        row.update({"prompt_n": comp.prompt_n, "predicted_n": comp.predicted_n, "expected_new": expected,
                    "cache_warning": comp.prompt_n > expected + self.cache_margin,
                    "finish_reason": comp.finish_reason, "text": comp.text, "timings": comp.timings,
                    "adherence": None, "ts": self.clock()})
        self.turns_log.write(row)
        return comp.text
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest harness/tests/test_dialogue.py -q`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add harness/dialogue.py harness/tests/fakes.py harness/tests/test_dialogue.py
git commit -m "harness: dialogue runner with slot pinning, persona modes and cache accounting"
```

---

### Task 6: `survey.py` and `instruments/batteries.json`

**Files:**
- Create: `harness/survey.py`, `instruments/batteries.json`, `harness/tests/test_survey.py`

**Interfaces:**
- Consumes: `AgentHandle`, `GenSettings`, `DyadSpec`, `Transcript`, `render`, `derive_seed`, `sha256_text`, `ServerError`.
- Produces:
  - `load_batteries(path) -> list[dict]` (validates each item has `id, battery, text, scale{min,max}`; raises `ValueError` otherwise; ids unique).
  - `answer_schema(item) -> dict` = `{"type":"object","properties":{"answer":{"type":"integer","minimum":min,"maximum":max}},"required":["answer"]}`.
  - `parse_answer(text: str, item: dict) -> int | None` (JSON parse, integer within scale, else None).
  - `class SurveyRunner: __init__(run_id, run_seed, mentor: AgentHandle, surveys_log: JsonlWriter, settings: GenSettings, clock=now_iso)`; `administer(spec: DyadSpec, attempt: int, phase: str, transcript: Transcript | None, items: list[dict]) -> list[dict]`. `phase` in `("pre","post")`; `post` requires a transcript. Row: `run_id, dyad_id, attempt, phase, item_id, battery, scale, answer, raw_text, prompt_sha256, prompt_n, seed, ts`, plus `error` on `ServerError` (row written, exception re-raised as `SurveyError(dyad_id, phase, item_id, cause)`).
  - Survey turn index for seeds: `0` for pre, `spec.n_turns + 1` for post; agent string `f"survey:{phase}:{item_id}"`.
  - Calls use `n_predict=32`, `temperature=0.0`, `json_schema=answer_schema(item)`, `id_slot=mentor.slot`.

- [ ] **Step 1: Write batteries.json and the failing tests**

```json
{
  "_note": "Placeholder wording for the de Jong (2024) categories. The US-adaptation task replaces text, not ids or scales.",
  "items": [
    {"id": "ideo_gender_racial_equality", "battery": "ideological", "scale": {"min": 1, "max": 5},
     "text": "On a scale from 1 (strongly disagree) to 5 (strongly agree): the government should do more to advance gender and racial equality. Answer with a single number."},
    {"id": "ideo_immigration", "battery": "ideological", "scale": {"min": 1, "max": 5},
     "text": "On a scale from 1 (strongly disagree) to 5 (strongly agree): immigration policy should be made more restrictive. Answer with a single number."},
    {"id": "ideo_redistribution", "battery": "ideological", "scale": {"min": 1, "max": 5},
     "text": "On a scale from 1 (strongly disagree) to 5 (strongly agree): the government should redistribute income from the better-off to the less well-off. Answer with a single number."},
    {"id": "ideo_multiculturalism", "battery": "ideological", "scale": {"min": 1, "max": 5},
     "text": "On a scale from 1 (strongly disagree) to 5 (strongly agree): multiculturalism makes the country a better place to live. Answer with a single number."},
    {"id": "ideo_gun_control", "battery": "ideological", "scale": {"min": 1, "max": 5},
     "text": "On a scale from 1 (strongly disagree) to 5 (strongly agree): gun laws should be made stricter. Answer with a single number."},
    {"id": "therm_dem_voters", "battery": "thermometer", "scale": {"min": 0, "max": 10},
     "text": "On a feeling thermometer from 0 (very cold) to 10 (very warm), how do you feel about Democratic voters? Answer with a single number."},
    {"id": "therm_dem_politicians", "battery": "thermometer", "scale": {"min": 0, "max": 10},
     "text": "On a feeling thermometer from 0 (very cold) to 10 (very warm), how do you feel about Democratic politicians? Answer with a single number."},
    {"id": "therm_rep_voters", "battery": "thermometer", "scale": {"min": 0, "max": 10},
     "text": "On a feeling thermometer from 0 (very cold) to 10 (very warm), how do you feel about Republican voters? Answer with a single number."},
    {"id": "therm_rep_politicians", "battery": "thermometer", "scale": {"min": 0, "max": 10},
     "text": "On a feeling thermometer from 0 (very cold) to 10 (very warm), how do you feel about Republican politicians? Answer with a single number."},
    {"id": "therm_independents", "battery": "thermometer", "scale": {"min": 0, "max": 10},
     "text": "On a feeling thermometer from 0 (very cold) to 10 (very warm), how do you feel about Independent voters? Answer with a single number."},
    {"id": "agree_democracy", "battery": "agreement", "scale": {"min": 1, "max": 5},
     "text": "On a scale from 1 (strongly disagree) to 5 (strongly agree): democracy is the best form of government even when it produces outcomes I dislike. Answer with a single number."},
    {"id": "agree_protest_rights", "battery": "agreement", "scale": {"min": 1, "max": 5},
     "text": "On a scale from 1 (strongly disagree) to 5 (strongly agree): people should be free to protest against the government, even in ways I find objectionable. Answer with a single number."},
    {"id": "agree_cross_partisan", "battery": "agreement", "scale": {"min": 1, "max": 5},
     "text": "On a scale from 1 (strongly disagree) to 5 (strongly agree): I am comfortable discussing politics with people who hold the opposite partisan view. Answer with a single number."}
  ]
}
```

```python
# harness/tests/test_survey.py
import json
from pathlib import Path
import pytest
from harness import log, survey
from harness.dialogue import AgentHandle, GenSettings, DyadSpec
from harness.survey import SurveyRunner, SurveyError
from harness.templates import ChatTemplate
from harness.transcript import Transcript, SEEKER, MENTOR
from harness.tests.fakes import FakeClient
from harness.tests.conftest import CHATML

REPO = Path(__file__).resolve().parents[2]


def items():
    return survey.load_batteries(REPO / "instruments" / "batteries.json")


def test_batteries_file_loads_13_unique_items():
    it = items()
    assert len(it) == 13 and len({i["id"] for i in it}) == 13
    assert {i["battery"] for i in it} == {"ideological", "thermometer", "agreement"}


def test_load_batteries_rejects_bad_item(tmp_path):
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"items": [{"id": "x", "battery": "b", "text": "t"}]}))
    with pytest.raises(ValueError):
        survey.load_batteries(p)


def test_answer_schema_and_parse():
    item = {"id": "i", "battery": "b", "text": "t", "scale": {"min": 1, "max": 5}}
    s = survey.answer_schema(item)
    assert s["properties"]["answer"] == {"type": "integer", "minimum": 1, "maximum": 5} and s["required"] == ["answer"]
    assert survey.parse_answer('{"answer": 4}', item) == 4
    assert survey.parse_answer('{"answer": 9}', item) is None
    assert survey.parse_answer('four', item) is None
    assert survey.parse_answer('{"answer": "3"}', item) is None


def make(tmp_path, replies=None, fail_on=None):
    tpl = ChatTemplate.from_source(CHATML)
    mc = FakeClient(replies or ['{"answer": 3}'], fail_on=fail_on)
    mentor = AgentHandle(MENTOR, mc, tpl, "mentorhash", slot=1)
    w = log.JsonlWriter(tmp_path / "surveys.jsonl")
    return SurveyRunner("run1", 99, mentor, w, GenSettings(), clock=lambda: "T"), mc


def spec():
    return DyadSpec("d1", {"topic": "t"}, "You are Dana.", "rem", "reinforced", 5, 2)


def test_pre_survey_fresh_context_per_item(tmp_path):
    runner, mc = make(tmp_path)
    rows = runner.administer(spec(), 1, "pre", None, items()[:3])
    assert len(rows) == 3 and len(mc.calls) == 3
    for c, it in zip(mc.calls, items()[:3]):
        assert c["prompt"] == f"<|im_start|>user\n{it['text']}<|im_end|>\n<|im_start|>assistant\n"
        assert c["json_schema"] == survey.answer_schema(it) and c["n_predict"] == 32 and c["temperature"] == 0.0
        assert c["id_slot"] == 1 and c["cache_prompt"] is True
    assert [r["answer"] for r in rows] == [3, 3, 3] and all(r["phase"] == "pre" for r in rows)


def test_post_survey_branches_off_mentor_view(tmp_path):
    runner, mc = make(tmp_path)
    t = Transcript("d1", "You are Dana.", "rem", "reinforced")
    t.append(1, SEEKER, "S1"); t.append(1, MENTOR, "M1")
    runner.administer(spec(), 1, "post", t, items()[:2])
    for c in mc.calls:
        assert c["prompt"].startswith("<|im_start|>user\nS1<|im_end|>\n<|im_start|>assistant\nM1<|im_end|>\n<|im_start|>user\n")
        assert "<|im_start|>system" not in c["prompt"]
    assert mc.calls[0]["prompt"] != mc.calls[1]["prompt"]


def test_post_requires_transcript(tmp_path):
    runner, _ = make(tmp_path)
    with pytest.raises(ValueError):
        runner.administer(spec(), 1, "post", None, items()[:1])
    with pytest.raises(ValueError):
        runner.administer(spec(), 1, "mid", None, items()[:1])


def test_rows_and_seeds(tmp_path):
    runner, mc = make(tmp_path, replies=['{"answer": 2}', 'nonsense'])
    rows = runner.administer(spec(), 1, "pre", None, items()[:2])
    logged = log.read_jsonl(tmp_path / "surveys.jsonl")
    assert logged == rows
    assert rows[0]["answer"] == 2 and rows[1]["answer"] is None and rows[1]["raw_text"] == "nonsense"
    assert rows[0]["seed"] == log.derive_seed(99, "d1", 1, 0, f"survey:pre:{items()[0]['id']}")
    assert rows[0]["seed"] != rows[1]["seed"]
    assert set(rows[0]) >= {"run_id", "dyad_id", "attempt", "phase", "item_id", "battery", "scale", "answer",
                            "raw_text", "prompt_sha256", "prompt_n", "seed", "ts"}
    post = runner.administer(spec(), 1, "post", Transcript("d1", "s", None, "once"), items()[:1])
    assert post[0]["seed"] == log.derive_seed(99, "d1", 1, 3, f"survey:post:{items()[0]['id']}")


def test_server_error_logs_and_raises(tmp_path):
    runner, mc = make(tmp_path, fail_on=2)
    with pytest.raises(SurveyError) as ei:
        runner.administer(spec(), 1, "pre", None, items()[:3])
    assert ei.value.item_id == items()[1]["id"] and ei.value.phase == "pre"
    rows = log.read_jsonl(tmp_path / "surveys.jsonl")
    assert len(rows) == 2 and rows[1]["answer"] is None and "fake failure" in rows[1]["error"]
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest harness/tests/test_survey.py -q`
Expected: ModuleNotFoundError for `harness.survey`.

- [ ] **Step 3: Implement**

```python
# harness/survey.py
"""Pre/post survey batteries for the mentor: one branch per item, numeric answer forced by JSON schema."""
from __future__ import annotations
import json
from pathlib import Path
from harness.client import ServerError
from harness.dialogue import AgentHandle, DyadSpec, GenSettings
from harness.log import JsonlWriter, derive_seed, now_iso, sha256_text
from harness.templates import render
from harness.transcript import Transcript, MENTOR

PHASES = ("pre", "post")
SURVEY_N_PREDICT = 32


class SurveyError(Exception):
    def __init__(self, dyad_id: str, phase: str, item_id: str, cause: Exception):
        super().__init__(f"{dyad_id} {phase} {item_id}: {cause}")
        self.dyad_id, self.phase, self.item_id, self.cause = dyad_id, phase, item_id, cause


def load_batteries(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    seen = set()
    for it in items:
        for k in ("id", "battery", "text", "scale"):
            if k not in it:
                raise ValueError(f"survey item missing {k!r}: {it}")
        if not isinstance(it["scale"], dict) or "min" not in it["scale"] or "max" not in it["scale"]:
            raise ValueError(f"survey item {it['id']} needs scale.min and scale.max")
        if it["id"] in seen:
            raise ValueError(f"duplicate survey item id {it['id']}")
        seen.add(it["id"])
    return items


def answer_schema(item: dict) -> dict:
    return {"type": "object",
            "properties": {"answer": {"type": "integer", "minimum": item["scale"]["min"], "maximum": item["scale"]["max"]}},
            "required": ["answer"]}


def parse_answer(text: str, item: dict) -> int | None:
    try:
        v = json.loads(text).get("answer")
    except (ValueError, AttributeError):
        return None
    if isinstance(v, bool) or not isinstance(v, int):
        return None
    if item["scale"]["min"] <= v <= item["scale"]["max"]:
        return v
    return None


class SurveyRunner:
    def __init__(self, run_id: str, run_seed: int, mentor: AgentHandle, surveys_log: JsonlWriter,
                 settings: GenSettings, clock=now_iso):
        self.run_id, self.run_seed, self.mentor = run_id, run_seed, mentor
        self.surveys_log, self.settings, self.clock = surveys_log, settings, clock

    def administer(self, spec: DyadSpec, attempt: int, phase: str, transcript: Transcript | None,
                   items: list[dict]) -> list[dict]:
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}")
        if phase == "post" and transcript is None:
            raise ValueError("post survey needs the dialogue transcript")
        base = transcript.view_for(MENTOR) if phase == "post" else []
        turn = 0 if phase == "pre" else spec.n_turns + 1
        rows = []
        for it in items:
            messages = base + [{"role": "user", "content": it["text"]}]
            prompt = render(self.mentor.template, messages, now=self.settings.now,
                            enable_thinking=self.settings.enable_thinking)
            seed = derive_seed(self.run_seed, spec.dyad_id, attempt, turn, f"survey:{phase}:{it['id']}")
            row = {"run_id": self.run_id, "dyad_id": spec.dyad_id, "attempt": attempt, "phase": phase,
                   "item_id": it["id"], "battery": it["battery"], "scale": it["scale"],
                   "prompt_sha256": sha256_text(prompt), "seed": seed}
            try:
                comp = self.mentor.client.complete(prompt, id_slot=self.mentor.slot, seed=seed,
                                                   n_predict=SURVEY_N_PREDICT, temperature=0.0,
                                                   json_schema=answer_schema(it))
            except ServerError as e:
                row.update({"answer": None, "raw_text": "", "prompt_n": None, "error": str(e), "ts": self.clock()})
                self.surveys_log.write(row)
                raise SurveyError(spec.dyad_id, phase, it["id"], e) from e
            row.update({"answer": parse_answer(comp.text, it), "raw_text": comp.text,
                        "prompt_n": comp.prompt_n, "ts": self.clock()})
            self.surveys_log.write(row)
            rows.append(row)
        return rows
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest harness/tests/test_survey.py -q`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add harness/survey.py instruments/batteries.json harness/tests/test_survey.py
git commit -m "harness: survey administration with branch-per-item and JSON-schema answers"
```

---

### Task 7: `scorer.py` (offline adherence judge)

**Files:**
- Create: `harness/scorer.py`, `harness/tests/test_scorer.py`

**Interfaces:**
- Consumes: `AgentHandle`, `GenSettings`, `render`, `read_jsonl`, `resume_index`, `derive_seed`, `sha256_text`, `RunPaths`, `ServerError`.
- Produces:
  - `METRICS = {"prompt_to_line": SEEKER, "line_to_line": SEEKER, "alignment": MENTOR}` (metric name -> agent it applies to).
  - `JUDGE_N_PREDICT = 160`.
  - `score_schema() -> dict` = object with `score` number 0..1 and `rationale` string, both required.
  - `build_judge_messages(metric, persona_text, topic, line, prior_own_lines: list[str], partner_line: str | None) -> list[dict]` (system + user).
  - `select_targets(turn_rows: list[dict], scope: str) -> list[tuple[dict, str]]`: `pilot` = every non-error row, both agents, all metrics for that agent; `main` = seeker rows only, turns divisible by 4 plus the highest turn, metrics for seeker. Raises `ValueError` on other scopes.
  - `latest_complete_attempts(status_rows) -> dict[dyad_id, attempt]` (only dyads whose latest attempt is `complete`).
  - `parse_score(text) -> tuple[float | None, str]`.
  - `class Scorer: __init__(run_id, run_seed, judge: AgentHandle, scores_log: JsonlWriter, settings: GenSettings, clock=now_iso)`; `score_run(paths: RunPaths, scope: str, manifest: dict) -> int` (number of rows written). Refuses (`ValueError`) when `judge.model_sha256` equals `manifest["seeker"]["model_sha256"]` or `manifest["mentor"]["model_sha256"]`. Skips `(dyad, attempt, turn, agent, metric)` already present in `scores.jsonl`. Reads persona and topic from `dyads.jsonl` (row with matching `dyad_id, attempt`). Row: `run_id, dyad_id, attempt, turn, agent, metric, score, rationale, raw_text, judge_sha256, judge_prompt_sha256, seed, ts` (`error` on failure; failures are logged and skipped, not raised).
  - Seeds: `derive_seed(run_seed, dyad_id, attempt, turn, f"judge:{agent}:{metric}")`.

- [ ] **Step 1: Write the failing tests**

```python
# harness/tests/test_scorer.py
import json
import pytest
from harness import log, scorer
from harness.dialogue import AgentHandle, GenSettings
from harness.scorer import Scorer
from harness.templates import ChatTemplate
from harness.transcript import SEEKER, MENTOR
from harness.tests.fakes import FakeClient
from harness.tests.conftest import CHATML


def test_score_schema():
    s = scorer.score_schema()
    assert s["properties"]["score"] == {"type": "number", "minimum": 0, "maximum": 1}
    assert set(s["required"]) == {"score", "rationale"}


def test_build_judge_messages_contents():
    m = scorer.build_judge_messages("line_to_line", "PERSONA", "immigration", "LINE", ["P1", "P2"], "PARTNER")
    assert m[0]["role"] == "system" and m[1]["role"] == "user"
    u = m[1]["content"]
    assert "PERSONA" in u and "LINE" in u and "P1" in u and "P2" in u and "PARTNER" in u and "immigration" in u
    a = scorer.build_judge_messages("alignment", "PERSONA", "t", "L", [], None)[1]["content"]
    assert "agree" in a.lower()
    with pytest.raises(ValueError):
        scorer.build_judge_messages("vibes", "p", "t", "l", [], None)


def rows():
    out = []
    for turn in range(1, 6):
        out.append({"dyad_id": "d", "attempt": 1, "turn": turn, "agent": SEEKER, "text": f"s{turn}", "finish_reason": "stop"})
        out.append({"dyad_id": "d", "attempt": 1, "turn": turn, "agent": MENTOR, "text": f"m{turn}", "finish_reason": "stop"})
    out.append({"dyad_id": "d", "attempt": 1, "turn": 6, "agent": SEEKER, "text": "", "finish_reason": "error"})
    return out


def test_select_targets_pilot_and_main():
    pilot = scorer.select_targets(rows(), "pilot")
    assert len(pilot) == 5 * 2 + 5 * 1          # seeker: 2 metrics, mentor: 1
    assert all(r["finish_reason"] != "error" for r, _ in pilot)
    main = scorer.select_targets(rows(), "main")
    turns = sorted({r["turn"] for r, _ in main})
    assert turns == [4, 5] and all(r["agent"] == SEEKER for r, _ in main)
    assert {m for _, m in main} == {"prompt_to_line", "line_to_line"}
    with pytest.raises(ValueError):
        scorer.select_targets(rows(), "all")


def test_latest_complete_attempts():
    st = [{"dyad_id": "a", "attempt": 1, "status": "failed"}, {"dyad_id": "a", "attempt": 2, "status": "complete"},
          {"dyad_id": "b", "attempt": 1, "status": "started"}]
    assert scorer.latest_complete_attempts(st) == {"a": 2}


def test_parse_score():
    assert scorer.parse_score('{"score": 0.8, "rationale": "fits"}') == (0.8, "fits")
    assert scorer.parse_score('{"score": 2, "rationale": "x"}') == (None, "x")
    assert scorer.parse_score('garbage') == (None, "")


def make_run(tmp_path):
    p = log.run_paths(tmp_path, "r1")
    log.JsonlWriter(p.dyads).write({"dyad_id": "d", "attempt": 1, "condition": {"topic": "immigration"},
                                    "persona_text": "PERSONA", "persona_reminder": "", "persona_mode": "once",
                                    "seed": 1, "n_turns": 2})
    st = log.JsonlWriter(p.status)
    st.write({"dyad_id": "d", "attempt": 1, "status": "started"}); st.write({"dyad_id": "d", "attempt": 1, "status": "complete"})
    tw = log.JsonlWriter(p.turns)
    for turn in (1, 2):
        tw.write({"run_id": "r1", "dyad_id": "d", "attempt": 1, "turn": turn, "agent": SEEKER, "text": f"s{turn}", "finish_reason": "stop"})
        tw.write({"run_id": "r1", "dyad_id": "d", "attempt": 1, "turn": turn, "agent": MENTOR, "text": f"m{turn}", "finish_reason": "stop"})
    manifest = {"seeker": {"model_sha256": "S"}, "mentor": {"model_sha256": "M"}}
    return p, manifest


def make_scorer(tmp_path, judge_hash="J", replies=None):
    jc = FakeClient(replies or ['{"score": 0.75, "rationale": "ok"}'])
    judge = AgentHandle("judge", jc, ChatTemplate.from_source(CHATML), judge_hash, slot=0)
    p, manifest = make_run(tmp_path)
    sc = Scorer("r1", 99, judge, log.JsonlWriter(p.scores), GenSettings(), clock=lambda: "T")
    return sc, jc, p, manifest


def test_score_run_pilot_writes_rows_and_skips_done(tmp_path):
    sc, jc, p, manifest = make_scorer(tmp_path)
    n = sc.score_run(p, "pilot", manifest)
    assert n == 2 * 2 + 2 * 1 and len(jc.calls) == n
    rows_ = log.read_jsonl(p.scores)
    assert all(r["score"] == 0.75 and r["judge_sha256"] == "J" and r["ts"] == "T" for r in rows_)
    r0 = rows_[0]
    assert set(r0) >= {"run_id", "dyad_id", "attempt", "turn", "agent", "metric", "score", "rationale", "raw_text",
                       "judge_sha256", "judge_prompt_sha256", "seed", "ts"}
    assert r0["seed"] == log.derive_seed(99, "d", 1, r0["turn"], f"judge:{r0['agent']}:{r0['metric']}")
    assert sc.score_run(p, "pilot", manifest) == 0            # idempotent
    for c in jc.calls:
        assert c["json_schema"] == scorer.score_schema() and c["temperature"] == 0.0 and c["n_predict"] == 160
        assert "PERSONA" in c["prompt"] and c["id_slot"] == 0


def test_score_run_refuses_judge_equal_to_dialogue_model(tmp_path):
    sc, _, p, manifest = make_scorer(tmp_path, judge_hash="M")
    with pytest.raises(ValueError):
        sc.score_run(p, "pilot", manifest)


def test_score_run_logs_errors_and_continues(tmp_path):
    sc, jc, p, manifest = make_scorer(tmp_path)
    jc.fail_on = 2
    n = sc.score_run(p, "pilot", manifest)
    rows_ = log.read_jsonl(p.scores)
    assert n == 5 and len(rows_) == 6 and sum(1 for r in rows_ if r.get("error")) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest harness/tests/test_scorer.py -q`
Expected: ModuleNotFoundError for `harness.scorer`.

- [ ] **Step 3: Implement**

```python
# harness/scorer.py
"""Offline adherence scorer: a judge model scores logged turns; results go to scores.jsonl."""
from __future__ import annotations
import json
from harness.client import ServerError
from harness.dialogue import AgentHandle, GenSettings
from harness.log import JsonlWriter, RunPaths, derive_seed, now_iso, read_jsonl, sha256_text
from harness.templates import render
from harness.transcript import SEEKER, MENTOR

METRICS = {"prompt_to_line": SEEKER, "line_to_line": SEEKER, "alignment": MENTOR}
SCOPES = ("pilot", "main")
JUDGE_N_PREDICT = 160

_SYSTEM = ("You are a careful annotator for a study of LLM-to-LLM political dialogue. You score one line at a "
           "time on a 0.0 to 1.0 scale and give a one-sentence rationale. Respond only with JSON.")

_TASKS = {
    "prompt_to_line": ("Does the LINE fit the PERSONA (backstory, values, stance anchors, openness)? "
                       "1.0 = fully in character, 0.0 = contradicts the persona."),
    "line_to_line": ("Is the LINE consistent with the speaker's own EARLIER LINES (positions, facts about themselves, tone)? "
                     "1.0 = fully consistent, 0.0 = contradicts them."),
    "alignment": ("How far does the LINE agree with the position the PERSONA holds on the TOPIC? "
                  "1.0 = fully agrees, 0.5 = neutral or balanced, 0.0 = opposes."),
}


def score_schema() -> dict:
    return {"type": "object",
            "properties": {"score": {"type": "number", "minimum": 0, "maximum": 1}, "rationale": {"type": "string"}},
            "required": ["score", "rationale"]}


def build_judge_messages(metric: str, persona_text: str, topic: str, line: str,
                         prior_own_lines: list[str], partner_line: str | None) -> list[dict]:
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}")
    parts = [f"TASK: {_TASKS[metric]}", f"TOPIC: {topic}", f"PERSONA:\n{persona_text}"]
    if metric == "line_to_line":
        earlier = "\n".join(f"- {l}" for l in prior_own_lines) or "- (none yet)"
        parts.append(f"EARLIER LINES by the same speaker:\n{earlier}")
    if partner_line is not None:
        parts.append(f"PARTNER'S PRECEDING LINE:\n{partner_line}")
    parts.append(f"LINE to score:\n{line}")
    parts.append('Return JSON: {"score": <0.0-1.0>, "rationale": "<one sentence>"}')
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def select_targets(turn_rows: list[dict], scope: str) -> list[tuple[dict, str]]:
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    rows = [r for r in turn_rows if r.get("finish_reason") != "error"]
    out = []
    if scope == "pilot":
        for r in rows:
            for metric, agent in METRICS.items():
                if r["agent"] == agent:
                    out.append((r, metric))
        return out
    seeker_rows = [r for r in rows if r["agent"] == SEEKER]
    by_dyad: dict[tuple, int] = {}
    for r in seeker_rows:
        key = (r["dyad_id"], r.get("attempt", 1))
        by_dyad[key] = max(by_dyad.get(key, 0), r["turn"])
    for r in seeker_rows:
        if r["turn"] % 4 == 0 or r["turn"] == by_dyad[(r["dyad_id"], r.get("attempt", 1))]:
            for metric, agent in METRICS.items():
                if agent == SEEKER:
                    out.append((r, metric))
    return out


def latest_complete_attempts(status_rows: list[dict]) -> dict[str, int]:
    latest: dict[str, dict] = {}
    for r in status_rows:
        cur = latest.get(r["dyad_id"])
        if cur is None or r["attempt"] >= cur["attempt"]:
            latest[r["dyad_id"]] = {"attempt": r["attempt"], "status": r["status"]}
    return {d: v["attempt"] for d, v in latest.items() if v["status"] == "complete"}


def parse_score(text: str) -> tuple[float | None, str]:
    try:
        d = json.loads(text)
        score = d.get("score")
        rationale = str(d.get("rationale", ""))
    except (ValueError, AttributeError):
        return None, ""
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
        return None, rationale
    return float(score), rationale


class Scorer:
    def __init__(self, run_id: str, run_seed: int, judge: AgentHandle, scores_log: JsonlWriter,
                 settings: GenSettings, clock=now_iso):
        self.run_id, self.run_seed, self.judge = run_id, run_seed, judge
        self.scores_log, self.settings, self.clock = scores_log, settings, clock

    def score_run(self, paths: RunPaths, scope: str, manifest: dict) -> int:
        for role in ("seeker", "mentor"):
            if manifest.get(role, {}).get("model_sha256") == self.judge.model_sha256:
                raise ValueError(f"judge model is the same as the {role} model; pick a third model")
        complete = latest_complete_attempts(read_jsonl(paths.status))
        dyads = {(d["dyad_id"], d["attempt"]): d for d in read_jsonl(paths.dyads)}
        done = {(s["dyad_id"], s["attempt"], s["turn"], s["agent"], s["metric"])
                for s in read_jsonl(paths.scores) if not s.get("error")}
        turns = [r for r in read_jsonl(paths.turns) if complete.get(r["dyad_id"]) == r.get("attempt", 1)]
        by_dyad: dict[tuple, list[dict]] = {}
        for r in turns:
            by_dyad.setdefault((r["dyad_id"], r.get("attempt", 1)), []).append(r)
        written = 0
        for row, metric in select_targets(turns, scope):
            key = (row["dyad_id"], row.get("attempt", 1), row["turn"], row["agent"], metric)
            if key in done:
                continue
            spec = dyads.get(key[:2], {})
            history = sorted(by_dyad[key[:2]], key=lambda r: (r["turn"], 0 if r["agent"] == SEEKER else 1))
            idx = history.index(row)
            prior_own = [r["text"] for r in history[:idx] if r["agent"] == row["agent"]]
            partner = next((r["text"] for r in reversed(history[:idx]) if r["agent"] != row["agent"]), None)
            messages = build_judge_messages(metric, spec.get("persona_text", ""), spec.get("condition", {}).get("topic", ""),
                                            row["text"], prior_own, partner)
            prompt = render(self.judge.template, messages, now=self.settings.now, enable_thinking=self.settings.enable_thinking)
            seed = derive_seed(self.run_seed, row["dyad_id"], key[1], row["turn"], f"judge:{row['agent']}:{metric}")
            out = {"run_id": self.run_id, "dyad_id": row["dyad_id"], "attempt": key[1], "turn": row["turn"],
                   "agent": row["agent"], "metric": metric, "judge_sha256": self.judge.model_sha256,
                   "judge_prompt_sha256": sha256_text(prompt), "seed": seed}
            try:
                comp = self.judge.client.complete(prompt, id_slot=self.judge.slot, seed=seed, n_predict=JUDGE_N_PREDICT,
                                                  temperature=0.0, json_schema=score_schema())
            except ServerError as e:
                out.update({"score": None, "rationale": "", "raw_text": "", "error": str(e), "ts": self.clock()})
                self.scores_log.write(out)
                continue
            score, rationale = parse_score(comp.text)
            out.update({"score": score, "rationale": rationale, "raw_text": comp.text, "ts": self.clock()})
            self.scores_log.write(out)
            written += 1
        return written
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest harness/tests/test_scorer.py -q`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add harness/scorer.py harness/tests/test_scorer.py
git commit -m "harness: offline adherence scorer with judge refusal and idempotent rows"
```

---

### Task 8: `run.py` (config, agents, check/run/survey/score CLI)

**Files:**
- Create: `harness/run.py`, `harness/config.example.json`, `harness/tests/test_run.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `DEFAULT_CONFIG: dict`; `load_config(path) -> dict` (deep-merges defaults; requires `seeker.url` and `mentor.url`).
  - `model_sha256_cached(path: str, cache_file: Path | None = None) -> str` (cache keyed by `path|size|mtime` in `~/.cache/llm-polarization/gguf-hashes.json`).
  - `build_agent(name: str, entry: dict, slot: int, cfg: dict, client_factory=LlamaClient) -> tuple[AgentHandle, dict]` (returns handle and the manifest entry `{url, alias, model_path, model_sha256, template_sha256, build_info, total_slots, default_generation_settings}`); GGUF path is `entry.get("gguf_path") or props["model_path"]`.
  - `check_agent(handle: AgentHandle, cfg: dict) -> list[tuple[str, bool, str]]` (health, parity, trailing-system acceptance for the seeker).
  - `plan_work(manifest_rows: list[dict], status_rows: list[dict]) -> list[tuple[DyadSpec, int]]`.
  - `run_dyad(worker_slot, spec, attempt, ctx) -> str` where `ctx` is a `RunContext` dataclass holding `run_id, run_seed, seeker, mentor, settings, items, logs (dict of JsonlWriter), clock`; writes `dyads.jsonl` row, `status started`, pre-survey, dialogue, post-survey, `status complete`; on `DialogueError`/`SurveyError` writes `status failed` with `reason` and returns `"failed"`.
  - `main(argv: list[str] | None = None) -> int` with subcommands `check`, `run`, `survey`, `score`.
- The `run` subcommand builds one `AgentHandle` pair per worker (same clients, different `slot`), uses `concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)`, and maps worker index to slot.
- `survey` subcommand: `--run-id --phase post` re-administers the post survey for complete dyads by rebuilding each transcript from `turns.jsonl` (same slot assignment: worker index); `--phase pre` runs pre items in a fresh context. Rows append to `surveys.jsonl` with the same attempt.

- [ ] **Step 1: Write config.example.json and the failing tests**

```json
{
  "data_dir": "data",
  "gguf_py_path": "/home/alex/.local/llamacpp/src/gguf-py",
  "batteries": "instruments/batteries.json",
  "run_seed": 20260908,
  "concurrency": null,
  "now": "2026-09-08",
  "generation": {"temperature": 0.7, "top_p": 0.95, "n_predict": 300, "timeout": 600, "enable_thinking": false},
  "seeker": {"url": "http://127.0.0.1:8201", "gguf_path": null},
  "mentor": {"url": "http://127.0.0.1:8202", "gguf_path": null},
  "judge":  {"url": "http://127.0.0.1:8099", "gguf_path": null}
}
```

```python
# harness/tests/test_run.py
import json
from pathlib import Path
import pytest
from harness import log, run as R
from harness.dialogue import AgentHandle, GenSettings, DyadSpec
from harness.templates import ChatTemplate
from harness.transcript import SEEKER, MENTOR
from harness.tests.fakes import FakeClient
from harness.tests.conftest import CHATML, REJECTS_TRAILING_SYSTEM

REPO = Path(__file__).resolve().parents[2]


def write_cfg(tmp_path, **over):
    cfg = {"data_dir": str(tmp_path / "data"), "gguf_py_path": None, "run_seed": 5, "now": "2026-09-08",
           "batteries": str(REPO / "instruments" / "batteries.json"),
           "seeker": {"url": "http://s"}, "mentor": {"url": "http://m"}, "judge": {"url": "http://j"}}
    cfg.update(over)
    p = tmp_path / "config.json"; p.write_text(json.dumps(cfg)); return p


def test_load_config_merges_defaults_and_requires_urls(tmp_path):
    cfg = R.load_config(write_cfg(tmp_path))
    assert cfg["generation"]["temperature"] == 0.7 and cfg["generation"]["n_predict"] == 300 and cfg["concurrency"] is None
    bad = tmp_path / "bad.json"; bad.write_text(json.dumps({"seeker": {"url": "x"}}))
    with pytest.raises(ValueError):
        R.load_config(bad)


def test_model_sha256_cached(tmp_path):
    f = tmp_path / "m.gguf"; f.write_bytes(b"abc")
    cache = tmp_path / "hashes.json"
    h1 = R.model_sha256_cached(str(f), cache)
    assert h1 == log.sha256_text("abc") and json.loads(cache.read_text())
    f.write_bytes(b"abcd")
    assert R.model_sha256_cached(str(f), cache) == log.sha256_text("abcd")


def fake_factory(url):
    return FakeClient(['{"answer": 3}'] if url in ("http://m", "http://j") else ["line"])


def test_build_agent_uses_props_model_path_and_template(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "read_template_from_gguf", lambda path, gguf_py_path=None: ChatTemplate.from_source(CHATML))
    monkeypatch.setattr(R, "model_sha256_cached", lambda path, cache_file=None: "HASH-" + path)
    cfg = R.load_config(write_cfg(tmp_path))
    handle, entry = R.build_agent(SEEKER, cfg["seeker"], 3, cfg, client_factory=fake_factory)
    assert handle.slot == 3 and handle.model_sha256 == "HASH-/fake/model.gguf" and handle.name == SEEKER
    assert entry["model_path"] == "/fake/model.gguf" and entry["total_slots"] == 4 and entry["template_sha256"] == handle.template.sha256


def test_check_agent_reports_parity_and_trailing_system(tmp_path):
    ok_tpl = ChatTemplate.from_source(CHATML)
    h = AgentHandle(SEEKER, FakeClient(), ok_tpl, "h", 0)
    results = R.check_agent(h, {"now": "2026-09-08"})
    assert all(ok for _, ok, _ in results) and {n for n, _, _ in results} >= {"health", "template_parity", "trailing_system"}
    bad = AgentHandle(SEEKER, FakeClient(), ChatTemplate.from_source(REJECTS_TRAILING_SYSTEM), "h", 0)
    results = dict((n, ok) for n, ok, _ in R.check_agent(bad, {"now": "2026-09-08"}))
    assert results["trailing_system"] is False and results["template_parity"] is False
    m = AgentHandle(MENTOR, FakeClient(), ChatTemplate.from_source(REJECTS_TRAILING_SYSTEM), "h", 0)
    assert "trailing_system" not in dict((n, ok) for n, ok, _ in R.check_agent(m, {"now": "2026-09-08"}))


def manifest_rows(n=3):
    return [{"dyad_id": f"d{i}", "condition": {"topic": "t"}, "persona_text": "P", "persona_reminder": "R",
             "persona_mode": "reinforced", "seed": i, "n_turns": 2} for i in range(n)]


def test_plan_work_resumes():
    status = [{"dyad_id": "d0", "attempt": 1, "status": "complete"}, {"dyad_id": "d1", "attempt": 1, "status": "failed"}]
    work = R.plan_work(manifest_rows(3), status)
    assert [(s.dyad_id, a) for s, a in work] == [("d1", 2), ("d2", 1)]


def make_ctx(tmp_path, seeker_client=None, mentor_client=None):
    tpl = ChatTemplate.from_source(CHATML)
    sc = seeker_client or FakeClient(["seeker line"])
    mc = mentor_client or FakeClient(['{"answer": 4}', "mentor line"])
    paths = log.run_paths(tmp_path, "r1")
    logs = {k: log.JsonlWriter(getattr(paths, k)) for k in ("dyads", "status", "turns", "surveys")}
    items = R.load_batteries(REPO / "instruments" / "batteries.json")[:2]
    ctx = R.RunContext("r1", 5, AgentHandle(SEEKER, sc, tpl, "S", 0), AgentHandle(MENTOR, mc, tpl, "M", 0),
                       GenSettings(), items, logs, clock=lambda: "T")
    return ctx, paths, sc, mc


def test_run_dyad_full_lifecycle(tmp_path):
    ctx, paths, sc, mc = make_ctx(tmp_path)
    spec = DyadSpec.from_row(manifest_rows(1)[0])
    assert R.run_dyad(2, spec, 1, ctx) == "complete"
    st = log.read_jsonl(paths.status)
    assert [s["status"] for s in st] == ["started", "complete"] and st[0]["attempt"] == 1
    d = log.read_jsonl(paths.dyads)[0]
    assert d["dyad_id"] == "d0" and d["attempt"] == 1 and d["persona_text"] == "P"
    turns = log.read_jsonl(paths.turns); surveys = log.read_jsonl(paths.surveys)
    assert len(turns) == 4 and len(surveys) == 4
    assert [s["phase"] for s in surveys] == ["pre", "pre", "post", "post"]
    assert all(c["id_slot"] == 2 for c in sc.calls + mc.calls)


def test_run_dyad_failure_marks_status(tmp_path):
    ctx, paths, sc, mc = make_ctx(tmp_path, seeker_client=FakeClient(["x"], fail_on=1))
    spec = DyadSpec.from_row(manifest_rows(1)[0])
    assert R.run_dyad(0, spec, 3, ctx) == "failed"
    st = log.read_jsonl(paths.status)
    assert st[-1]["status"] == "failed" and st[-1]["attempt"] == 3 and "fake failure" in st[-1]["reason"]


def test_main_run_and_score_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "read_template_from_gguf", lambda path, gguf_py_path=None: ChatTemplate.from_source(CHATML))
    monkeypatch.setattr(R, "model_sha256_cached", lambda path, cache_file=None: "HASH-" + path.split("/")[-1])
    clients = {}
    def factory(url):
        clients[url] = FakeClient(['{"answer": 2}', "line"] if url != "http://j" else ['{"score": 0.5, "rationale": "r"}'])
        clients[url].props = lambda: {"model_path": f"/{url[7:]}.gguf", "total_slots": 2, "build_info": "b", "model_alias": url[7:], "default_generation_settings": {}}
        return clients[url]
    monkeypatch.setattr(R, "LlamaClient", factory)
    cfg = write_cfg(tmp_path, concurrency=2)
    man = tmp_path / "dyads.jsonl"
    man.write_text("".join(json.dumps(r) + "\n" for r in manifest_rows(3)))
    assert R.main(["check", "--config", str(cfg)]) == 0
    assert R.main(["run", "--config", str(cfg), "--manifest", str(man), "--run-id", "r1"]) == 0
    paths = log.run_paths(tmp_path / "data", "r1")
    assert json.loads(paths.manifest.read_text())["seeker"]["model_sha256"] == "HASH-s.gguf"
    assert len(log.read_jsonl(paths.turns)) == 3 * 4
    assert sorted(s["status"] for s in log.read_jsonl(paths.status)).count("complete") == 3
    assert R.main(["run", "--config", str(cfg), "--manifest", str(man), "--run-id", "r1"]) == 0   # resume: nothing to do
    assert len(log.read_jsonl(paths.turns)) == 3 * 4
    assert R.main(["score", "--config", str(cfg), "--run-id", "r1", "--scope", "main"]) == 0
    scores = log.read_jsonl(paths.scores)
    assert len(scores) == 3 * 1 * 2 and all(s["score"] == 0.5 for s in scores)      # main: seeker, last turn (2), 2 metrics
    assert R.main(["survey", "--config", str(cfg), "--run-id", "r1", "--phase", "post"]) == 0
    assert len([s for s in log.read_jsonl(paths.surveys) if s["phase"] == "post"]) == 3 * 13 * 2
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest harness/tests/test_run.py -q`
Expected: ModuleNotFoundError for `harness.run`.

- [ ] **Step 3: Implement**

```python
# harness/run.py
"""CLI: check servers, run a dialogue manifest, re-administer surveys, score a run."""
from __future__ import annotations
import argparse, copy, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from harness import log
from harness.client import LlamaClient
from harness.dialogue import AgentHandle, DialogueError, DialogueRunner, DyadSpec, GenSettings
from harness.log import JsonlWriter, RunPaths, now_iso, read_jsonl, run_paths
from harness.scorer import Scorer
from harness.survey import SurveyError, SurveyRunner, load_batteries
from harness.templates import FIXTURE_MESSAGES, TemplateError, parity_check, read_template_from_gguf, render
from harness.transcript import MENTOR, SEEKER, Transcript

DEFAULT_CONFIG = {
    "data_dir": "data", "gguf_py_path": None, "batteries": "instruments/batteries.json", "run_seed": 0,
    "concurrency": None, "now": "2026-09-08",
    "generation": {"temperature": 0.7, "top_p": 0.95, "n_predict": 300, "timeout": 600, "enable_thinking": False},
    "seeker": {"url": None, "gguf_path": None}, "mentor": {"url": None, "gguf_path": None},
    "judge": {"url": None, "gguf_path": None},
}
HASH_CACHE = Path.home() / ".cache" / "llm-polarization" / "gguf-hashes.json"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path) -> dict:
    cfg = _merge(DEFAULT_CONFIG, json.loads(Path(path).read_text(encoding="utf-8")))
    for role in ("seeker", "mentor"):
        if not cfg[role].get("url"):
            raise ValueError(f"config needs {role}.url")
    return cfg


def model_sha256_cached(path: str, cache_file: Path | None = None) -> str:
    cache_file = cache_file or HASH_CACHE
    st = os.stat(path)
    key = f"{path}|{st.st_size}|{int(st.st_mtime)}"
    cache = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
        except ValueError:
            cache = {}
    if key not in cache:
        cache[key] = log.sha256_file(Path(path))
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache, indent=1))
    return cache[key]


def build_agent(name: str, entry: dict, slot: int, cfg: dict, client_factory=LlamaClient) -> tuple[AgentHandle, dict]:
    client = client_factory(entry["url"]) if client_factory is not LlamaClient else \
        LlamaClient(entry["url"], timeout=cfg["generation"]["timeout"])
    props = client.props()
    model_path = entry.get("gguf_path") or props.get("model_path")
    if not model_path:
        raise ValueError(f"{name}: no gguf_path in config and server reports no model_path")
    template = read_template_from_gguf(model_path, gguf_py_path=cfg.get("gguf_py_path"))
    sha = model_sha256_cached(model_path)
    handle = AgentHandle(name, client, template, sha, slot, alias=props.get("model_alias", ""))
    manifest_entry = {"url": entry["url"], "alias": props.get("model_alias", ""), "model_path": model_path,
                      "model_sha256": sha, "template_sha256": template.sha256, "build_info": props.get("build_info", ""),
                      "total_slots": props.get("total_slots"), "default_generation_settings": props.get("default_generation_settings", {})}
    return handle, manifest_entry


def check_agent(handle: AgentHandle, cfg: dict) -> list[tuple[str, bool, str]]:
    now = cfg.get("now", "2026-09-08")
    results = [("health", handle.client.health(), handle.client.url if hasattr(handle.client, "url") else "")]
    try:
        ok, ours, theirs = parity_check(handle.template, handle.client, FIXTURE_MESSAGES[:-1], now=now)
        results.append(("template_parity", ok, "" if ok else f"ours={ours[-120:]!r} theirs={theirs[-120:]!r}"))
    except TemplateError as e:
        results.append(("template_parity", False, str(e)))
    if handle.name == SEEKER:
        try:
            render(handle.template, FIXTURE_MESSAGES, now=now)
            results.append(("trailing_system", True, ""))
        except TemplateError as e:
            results.append(("trailing_system", False, str(e)))
    return results


def plan_work(manifest_rows: list[dict], status_rows: list[dict]) -> list[tuple[DyadSpec, int]]:
    idx = log.resume_index(status_rows)
    work = []
    for row in manifest_rows:
        spec = DyadSpec.from_row(row)
        attempt = log.next_attempt(idx, spec.dyad_id)
        if attempt is not None:
            work.append((spec, attempt))
    return work


@dataclass
class RunContext:
    run_id: str
    run_seed: int
    seeker: AgentHandle
    mentor: AgentHandle
    settings: GenSettings
    items: list[dict]
    logs: dict
    clock: object = now_iso


def _with_slot(h: AgentHandle, slot: int) -> AgentHandle:
    return AgentHandle(h.name, h.client, h.template, h.model_sha256, slot, h.alias)


def run_dyad(worker_slot: int, spec: DyadSpec, attempt: int, ctx: RunContext) -> str:
    seeker, mentor = _with_slot(ctx.seeker, worker_slot), _with_slot(ctx.mentor, worker_slot)
    ctx.logs["dyads"].write({"run_id": ctx.run_id, "dyad_id": spec.dyad_id, "attempt": attempt, "condition": spec.condition,
                             "persona_text": spec.persona_text, "persona_reminder": spec.persona_reminder,
                             "persona_mode": spec.persona_mode, "seed": spec.seed, "n_turns": spec.n_turns, "ts": ctx.clock()})
    ctx.logs["status"].write({"run_id": ctx.run_id, "dyad_id": spec.dyad_id, "attempt": attempt, "status": "started", "ts": ctx.clock()})
    dialogue = DialogueRunner(ctx.run_id, ctx.run_seed, seeker, mentor, ctx.settings, ctx.logs["turns"], clock=ctx.clock)
    surveys = SurveyRunner(ctx.run_id, ctx.run_seed, mentor, ctx.logs["surveys"], ctx.settings, clock=ctx.clock)
    try:
        surveys.administer(spec, attempt, "pre", None, ctx.items)
        transcript = dialogue.run(spec, attempt)
        surveys.administer(spec, attempt, "post", transcript, ctx.items)
    except (DialogueError, SurveyError) as e:
        ctx.logs["status"].write({"run_id": ctx.run_id, "dyad_id": spec.dyad_id, "attempt": attempt, "status": "failed",
                                  "reason": str(e), "ts": ctx.clock()})
        return "failed"
    ctx.logs["status"].write({"run_id": ctx.run_id, "dyad_id": spec.dyad_id, "attempt": attempt, "status": "complete", "ts": ctx.clock()})
    return "complete"


def _settings(cfg: dict) -> GenSettings:
    g = cfg["generation"]
    return GenSettings(g["temperature"], g["top_p"], g["n_predict"], cfg["now"], g["enable_thinking"])


def _agents(cfg: dict, roles=(SEEKER, MENTOR)) -> dict:
    out = {}
    for role in roles:
        out[role] = build_agent(role, cfg[role], 0, cfg)
    return out


def cmd_check(cfg: dict) -> int:
    ok_all = True
    for role, (handle, entry) in _agents(cfg).items():
        print(f"[{role}] {entry['alias']} {entry['model_path']} sha256={entry['model_sha256'][:12]} slots={entry['total_slots']}")
        for name, ok, detail in check_agent(handle, cfg):
            ok_all &= ok
            print(f"   {'ok ' if ok else 'FAIL'} {name} {detail}")
    return 0 if ok_all else 1


def _rebuild_transcript(dyad_row: dict, turn_rows: list[dict]) -> Transcript:
    t = Transcript(dyad_row["dyad_id"], dyad_row["persona_text"], dyad_row.get("persona_reminder") or None, dyad_row["persona_mode"])
    for r in sorted(turn_rows, key=lambda r: (r["turn"], 0 if r["agent"] == SEEKER else 1)):
        if r.get("finish_reason") != "error":
            t.append(r["turn"], r["agent"], r["text"])
    return t


def cmd_run(cfg: dict, manifest_path: str, run_id: str) -> int:
    if cmd_check(cfg) != 0:
        print("check failed; not running", file=sys.stderr)
        return 1
    agents = _agents(cfg)
    (seeker, s_entry), (mentor, m_entry) = agents[SEEKER], agents[MENTOR]
    paths = run_paths(cfg["data_dir"], run_id)
    log.write_manifest(paths, {"run_id": run_id, "started_at": now_iso(), "harness_commit": _git_commit(),
                               "config": cfg, "seeker": s_entry, "mentor": m_entry})
    concurrency = cfg["concurrency"] or min(int(s_entry["total_slots"] or 1), int(m_entry["total_slots"] or 1))
    rows = read_jsonl(Path(manifest_path))
    work = plan_work(rows, read_jsonl(paths.status))
    print(f"run {run_id}: {len(work)} of {len(rows)} dyads to run, concurrency {concurrency}")
    if not work:
        return 0
    logs = {k: JsonlWriter(getattr(paths, k)) for k in ("dyads", "status", "turns", "surveys")}
    ctx = RunContext(run_id, int(cfg["run_seed"]), seeker, mentor, _settings(cfg),
                     load_batteries(cfg["batteries"]), logs)
    free = list(range(concurrency))
    lock = threading.Lock()

    def job(spec, attempt):
        with lock:
            slot = free.pop()
        try:
            r = run_dyad(slot, spec, attempt, ctx)
            print(f"  {spec.dyad_id} attempt {attempt}: {r}", flush=True)
            return r
        finally:
            with lock:
                free.append(slot)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(lambda w: job(*w), work))
    failed = results.count("failed")
    print(f"done: {results.count('complete')} complete, {failed} failed")
    return 0


def cmd_survey(cfg: dict, run_id: str, phase: str) -> int:
    agents = _agents(cfg)
    mentor = agents[MENTOR][0]
    paths = run_paths(cfg["data_dir"], run_id)
    items = load_batteries(cfg["batteries"])
    complete = {d: v["attempt"] for d, v in log.resume_index(read_jsonl(paths.status)).items() if v["status"] == "complete"}
    dyads = {(d["dyad_id"], d["attempt"]): d for d in read_jsonl(paths.dyads)}
    turns = read_jsonl(paths.turns)
    runner = SurveyRunner(run_id, int(cfg["run_seed"]), _with_slot(mentor, 0), JsonlWriter(paths.surveys), _settings(cfg))
    n = 0
    for dyad_id, attempt in complete.items():
        spec = DyadSpec.from_row(dyads[(dyad_id, attempt)])
        transcript = None
        if phase == "post":
            transcript = _rebuild_transcript(dyads[(dyad_id, attempt)],
                                             [r for r in turns if r["dyad_id"] == dyad_id and r.get("attempt", 1) == attempt])
        try:
            n += len(runner.administer(spec, attempt, phase, transcript, items))
        except SurveyError as e:
            print(f"  {dyad_id}: {e}", file=sys.stderr)
    print(f"{phase} survey: {n} rows for {len(complete)} dyads")
    return 0


def cmd_score(cfg: dict, run_id: str, scope: str) -> int:
    if not cfg["judge"].get("url"):
        print("config needs judge.url", file=sys.stderr)
        return 1
    judge, _ = build_agent("judge", cfg["judge"], 0, cfg)
    paths = run_paths(cfg["data_dir"], run_id)
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    n = Scorer(run_id, int(cfg["run_seed"]), judge, JsonlWriter(paths.scores), _settings(cfg)).score_run(paths, scope, manifest)
    print(f"scored {n} new rows ({scope})")
    return 0


def _git_commit() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="harness", description="Dyad harness for the LLM polarization study")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "run", "survey", "score"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name != "check":
            p.add_argument("--run-id", required=True)
    sub.choices["run"].add_argument("--manifest", required=True)
    sub.choices["survey"].add_argument("--phase", choices=("pre", "post"), default="post")
    sub.choices["score"].add_argument("--scope", choices=("pilot", "main"), default="pilot")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.cmd == "check":
        return cmd_check(cfg)
    if a.cmd == "run":
        return cmd_run(cfg, a.manifest, a.run_id)
    if a.cmd == "survey":
        return cmd_survey(cfg, a.run_id, a.phase)
    return cmd_score(cfg, a.run_id, a.scope)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest harness/tests/test_run.py -q` then `python -m pytest harness/tests -q`
Expected: test_run 8 passed; whole suite green.

- [ ] **Step 5: Commit**

```bash
git add harness/run.py harness/config.example.json harness/tests/test_run.py
git commit -m "harness: CLI with check, run (resumable, slot-per-worker), survey and score"
```

---

### Task 9: Live test, docs, and spec addendum

**Files:**
- Create: `harness/tests/test_live.py`
- Modify: `harness/README.md` (replace the "not yet written" status with usage), `docs/superpowers/specs/2026-09-08-dyad-harness-design.md` (section 3: add `dyads.jsonl`), `README.md` (status line for the pipeline)

**Interfaces:**
- Consumes: `run.py` CLI, `build_agent`, `DialogueRunner`, `SurveyRunner`, `Scorer`.
- Live test contract: skipped unless `HARNESS_LIVE_URL` is set (a llama-server with `-np >= 2`, jinja enabled, a model whose GGUF is readable at the path `/props` reports); `GGUF_PY_PATH` must point at llama.cpp's `gguf-py`. Seeker and mentor use the same server on slots 0 and 1; the judge is the same server on slot 1 with a distinct fake hash so the refusal check does not fire.

- [ ] **Step 1: Write the live test (it should SKIP without the env var and PASS with it)**

```python
# harness/tests/test_live.py
"""Opt-in end-to-end test against a real llama-server. Set HARNESS_LIVE_URL and GGUF_PY_PATH.
On the desktop: ~/llm-serving/llama.cpp/build/bin/llama-server -m ~/llm-serving/gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -ngl 99 -fa on -np 2 -c 16384 -ctk q8_0 -ctv q8_0 --port 8090 --host 127.0.0.1"""
import os
import pytest
from harness import log, run as R
from harness.dialogue import AgentHandle, DialogueRunner, DyadSpec, GenSettings
from harness.scorer import Scorer
from harness.survey import SurveyRunner, load_batteries
from harness.transcript import SEEKER, MENTOR

URL = os.environ.get("HARNESS_LIVE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="HARNESS_LIVE_URL not set")


@pytest.fixture(scope="module")
def cfg():
    return R._merge(R.DEFAULT_CONFIG, {"seeker": {"url": URL}, "mentor": {"url": URL}, "judge": {"url": URL},
                                       "gguf_py_path": os.environ.get("GGUF_PY_PATH"), "run_seed": 1,
                                       "generation": {"n_predict": 40}})


def test_check_passes(cfg):
    for role in (SEEKER, MENTOR):
        handle, _ = R.build_agent(role, cfg[role], 0, cfg)
        results = R.check_agent(handle, cfg)
        assert all(ok for _, ok, _ in results), results


def test_two_turn_dialogue_survey_and_score(cfg, tmp_path):
    seeker, s_entry = R.build_agent(SEEKER, cfg[SEEKER], 0, cfg)
    mentor, m_entry = R.build_agent(MENTOR, cfg[MENTOR], 1, cfg)
    paths = log.run_paths(tmp_path, "live")
    logs = {k: log.JsonlWriter(getattr(paths, k)) for k in ("dyads", "status", "turns", "surveys", "scores")}
    settings = R._settings(cfg)
    spec = DyadSpec("live-1", {"topic": "immigration enforcement"},
                    "You are Dana, a 54-year-old rancher from Montana who wants practical advice. Open the conversation by "
                    "asking, in two sentences, for guidance about the National Guard being sent to the border.",
                    "Note to self: I am Dana, a rancher; I am worried but open-minded; keep asking for practical guidance.",
                    "reinforced", 7, 2)
    ctx = R.RunContext("live", 1, seeker, mentor, settings, load_batteries("instruments/batteries.json")[:2], logs)
    assert R.run_dyad(0, spec, 1, ctx) == "complete"
    turns = log.read_jsonl(paths.turns)
    assert [(r["turn"], r["agent"]) for r in turns] == [(1, SEEKER), (1, MENTOR), (2, SEEKER), (2, MENTOR)]
    assert all(r["text"].strip() for r in turns)
    assert all(not r["cache_warning"] for r in turns[2:]), [(r["prompt_n"], r["expected_new"]) for r in turns]
    surveys = log.read_jsonl(paths.surveys)
    assert len(surveys) == 4 and all(s["answer"] is not None for s in surveys), surveys
    judge = AgentHandle("judge", mentor.client, mentor.template, "judge-distinct", 1)
    manifest = {"seeker": s_entry, "mentor": m_entry}
    n = Scorer("live", 1, judge, logs["scores"], settings).score_run(paths, "pilot", manifest)
    assert n == 6
    scores = log.read_jsonl(paths.scores)
    assert all(s["score"] is not None and 0 <= s["score"] <= 1 for s in scores), scores
```

- [ ] **Step 2: Run without the server**

Run: `python -m pytest harness/tests/test_live.py -q`
Expected: 2 skipped.

- [ ] **Step 3: Run with a live server on the desktop's RTX 5080**

```bash
~/llm-serving/llama.cpp/build/bin/llama-server -m ~/llm-serving/gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -ngl 99 -fa on -np 2 -c 16384 -ctk q8_0 -ctv q8_0 --host 127.0.0.1 --port 8090 > /tmp/live-server.log 2>&1 &
until curl -s http://127.0.0.1:8090/health | grep -q ok; do sleep 2; done
cd ~/llm-polarization && HARNESS_LIVE_URL=http://127.0.0.1:8090 GGUF_PY_PATH=~/llm-serving/llama.cpp/gguf-py \
  python -m pytest harness/tests/test_live.py -q -s
```
Expected: 2 passed. Then stop the server (`pkill -f "port 8090"` from a different shell, or `kill %1`).

- [ ] **Step 4: Update docs**

Replace the first two paragraphs of `harness/README.md` with:

```markdown
# Dyad harness

`harness/` runs seeker/mentor dialogues against two llama-server endpoints, administers the mentor's
pre/post survey batteries, and scores seeker adherence offline. Design: `docs/superpowers/specs/2026-09-08-dyad-harness-design.md`.

## Use

```bash
pip install -r harness/requirements.txt            # jinja2 only; gguf-py comes from the llama.cpp checkout
cp harness/config.example.json config.json         # edit urls, gguf_py_path, run_seed
python -m harness.run check  --config config.json
python -m harness.run run    --config config.json --manifest dyads.jsonl --run-id pilot-2026-09-18
python -m harness.run score  --config config.json --run-id pilot-2026-09-18 --scope pilot
python -m harness.run survey --config config.json --run-id pilot-2026-09-18 --phase post   # re-administer
python -m pytest harness/tests -q                  # unit tests; HARNESS_LIVE_URL=... adds the live test
```

Servers are started outside the harness with the flags in `models/RUN_APPROACH.md`. Output lands in
`data/<run_id>/` as `manifest.json`, `dyads.jsonl`, `status.jsonl`, `turns.jsonl`, `surveys.jsonl`, `scores.jsonl`.
```

Keep the rest of the file (terms, four requirements, serving contract, persona-stability requirements).

In the spec's section 3, after the run manifest paragraph, add: "**`dyads.jsonl`**, one row per attempt: the DyadSpec (`dyad_id, attempt, condition, persona_text, persona_reminder, persona_mode, seed, n_turns, ts`). The scorer reads persona and topic from here." In the top-level `README.md`, change the status bullet "Pipeline: nothing written yet" to "Pipeline: `harness/` runs dialogues, surveys and scoring (unit-tested; live-tested on the desktop 5080). No pilot data yet."

- [ ] **Step 5: Run the whole suite and commit**

Run: `python -m pytest harness/tests -q`
Expected: all unit tests pass, live tests skipped (or passed if the env var is set).

```bash
git add harness/tests/test_live.py harness/README.md README.md docs/superpowers/specs/2026-09-08-dyad-harness-design.md
git commit -m "harness: live end-to-end test, usage docs, spec addendum for dyads.jsonl"
```

---

## Self-review

**Spec coverage.** Section 2 architecture: Tasks 1 to 8 create the eight modules. Section 3 data model: Task 1 (paths, manifest, status/resume), Task 5 (turns rows), Task 6 (surveys rows), Task 7 (scores rows), Task 8 (dyads rows, run manifest fields from `/props`). Section 4 engine: Task 2 (projection, modes), Task 5 (opening from system only, alternation, settings, seeds, cache accounting, error rows). Section 5 surveys: Task 6 (batteries file, pre/post branching, schema, item order). Section 6 scorer: Task 7 (three metrics, scopes, judge refusal, idempotence). Section 7 run control: Task 8 (config, check, run, resume, concurrency, survey/score subcommands; servers external). Section 8 testing: unit tests in every task, `FakeClient` in Task 5, live test in Task 9, parity check in Tasks 3 and 8. Ctrl-C handling: `ThreadPoolExecutor` lets in-flight jobs finish on KeyboardInterrupt; no extra code.

**Placeholders.** None: every step carries code. The batteries file is explicitly placeholder wording by design (spec section 5).

**Type consistency.** `AgentHandle(name, client, template, model_sha256, slot, alias)` is used identically in Tasks 5 to 9. `DyadSpec.from_row` in Task 5 is what Task 8's `plan_work` and Task 9 use. `GenSettings.now`/`enable_thinking` are read by `render` calls in Tasks 5, 6, 7. `derive_seed(run_seed, dyad_id, attempt, turn, agent)` signature is constant. `run_paths` field names (`dyads, status, turns, surveys, scores`) match the `logs` dict keys in Task 8 and Task 9.
