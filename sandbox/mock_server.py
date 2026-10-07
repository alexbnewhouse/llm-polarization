"""A fake llama-server and a minimal GGUF writer, so the sandbox, its tests and the end-to-end test run
without a GPU. Never for data: every reply is synthetic, deterministic in (seed, prompt), and says so.

It is faithful exactly where the harness looks:

- `/apply-template` renders with the harness's own `harness.templates.render`, from the same template and
  BOS/EOS strings written into the mock GGUF, so `check`'s template parity holds by construction.
- `/completion` keeps one prompt per slot and reports `prompt_n` as the tokens past the longest common
  prefix with that slot's previous prompt, the way llama-server's prompt cache does. That is what
  `check`'s cache-reuse probe and the dialogue's `cache_warning` measure.
- A request with a `json_schema` gets a reply that satisfies it, as llama-server's grammar would force:
  `{"answer": N}` on the survey item's scale, `{"score": x, "rationale": ...}` for the judge.
- A "token" is a whitespace-separated word, in `/tokenize`, `prompt_n`, `n_predict` and `n_ctx` alike,
  so the harness's own estimates (`harness.dialogue.expected_new_tokens`) agree with the server's count.

The mock writes its GGUF by hand (zero tensors, metadata only) so it never needs gguf-py; reading the file
back is the harness's job, exactly as it is for a real model.

    python -m sandbox.mock_server --dir workspace/mock --port 18201
"""
from __future__ import annotations
import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import random
import re
import struct
import sys
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from harness.templates import ChatTemplate, TemplateError, render

ROLES = ("seeker", "mentor", "judge")
BOS, EOS = "<s>", "</s>"
VOCAB = [BOS, EOS, "<|im_start|>", "<|im_end|>"]

# ChatML, byte for byte the fixture the harness's own tests use. It renders a system message wherever it
# appears, so the seeker's trailing persona reminder (`reinforced` mode) passes `check`'s trailing_system row.
CHATML = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)

MAX_BODY = 64 * 1024 * 1024
# Synthetic throughput for `timings` when there is no delay to measure: plausible numbers, not a benchmark.
PROMPT_TOKENS_PER_SECOND = 2000.0
PREDICTED_TOKENS_PER_SECOND = 100.0

# --------------------------------------------------------------------------------------------- GGUF

GGUF_VERSION = 3
GGUF_ALIGNMENT = 32
_GGUF_UINT32, _GGUF_STRING, _GGUF_ARRAY = 4, 8, 9


def _gguf_string(s: str) -> bytes:
    """A GGUF string: uint64 byte length, then the UTF-8 bytes (no terminator)."""
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _gguf_value(key: str, value) -> bytes:
    """The type tag and payload of one metadata value. Only the three types a mock model needs."""
    if isinstance(value, bool):      # bool is an int subclass; refuse it rather than write a UINT32 by accident
        raise ValueError(f"GGUF metadata {key!r}: bool is not supported")
    if isinstance(value, str):
        return struct.pack("<I", _GGUF_STRING) + _gguf_string(value)
    if isinstance(value, int):
        if not 0 <= value < 2 ** 32:
            raise ValueError(f"GGUF metadata {key!r}: {value} does not fit a uint32")
        return struct.pack("<II", _GGUF_UINT32, value)
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return struct.pack("<IIQ", _GGUF_ARRAY, _GGUF_STRING, len(value)) + b"".join(_gguf_string(v) for v in value)
    raise ValueError(f"GGUF metadata {key!r}: unsupported value type {type(value).__name__} "
                     "(str, int or list of str)")


def gguf_bytes(metadata: dict) -> bytes:
    """A complete GGUF v3 file with `metadata` and no tensors, little-endian, padded to the default
    32-byte alignment where tensor data would start."""
    parts = [b"GGUF", struct.pack("<IQQ", GGUF_VERSION, 0, len(metadata))]
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"GGUF metadata key must be a non-empty string, got {key!r}")
        parts.append(_gguf_string(key))
        parts.append(_gguf_value(key, value))
    data = b"".join(parts)
    return data + b"\0" * (-len(data) % GGUF_ALIGNMENT)


def write_gguf(path, metadata: dict) -> Path:
    """Write a minimal GGUF v3 (zero tensors) holding `metadata`: str -> STRING, int -> UINT32,
    list[str] -> ARRAY of STRING. The file is left untouched when it already holds these bytes, so its
    mtime -- and with it the harness's cached sha256 (`harness.run.model_sha256_cached`) -- stays put."""
    path = Path(path)
    data = gguf_bytes(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size == len(data) and path.read_bytes() == data:
        return path
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


def mock_metadata(role: str, template: str = CHATML) -> dict:
    """The metadata of a mock model for `role`. The name differs per role, so each role's file -- and its
    sha256, which is the harness's model identity -- differs too (the judge must not hash like the mentor)."""
    return {
        # Not a llama.cpp architecture on purpose: a real llama-server pointed at this file fails at once,
        # with a message naming it, instead of looking for tensors.
        "general.architecture": "sandbox-mock",
        "general.name": f"sandbox-mock-{role}",
        "general.description": "Synthetic GGUF written by sandbox/mock_server.py: metadata only, no weights. "
                               "Never for data.",
        "tokenizer.chat_template": template,
        "tokenizer.ggml.tokens": list(VOCAB),
        "tokenizer.ggml.bos_token_id": 0,
        "tokenizer.ggml.eos_token_id": 1,
    }


def write_mock_gguf(path, role: str, template: str = CHATML) -> Path:
    """Write the mock GGUF for one role: the chat template the mock server renders with, plus BOS/EOS."""
    if not isinstance(role, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", role):
        raise ValueError(f"role must be a lower-case slug, got {role!r}")
    return write_gguf(path, mock_metadata(role, template))


# --------------------------------------------------------------------------------------------- replies

# Sentence banks, flavoured by role. Readable, but every reply starts with a [mock-<role>] tag so a mock
# transcript can never be mistaken for a model's.
_BANKS = {
    "seeker": (
        (1, 1, ("I hear what you are saying, but I am not convinced yet.",
                "That is a fair point, and not one I had really considered.",
                "Honestly, that sounds a little too tidy to me.",
                "Okay, I can follow that part of the argument.",
                "I have heard that before, mostly from people who do not live where I do.",
                "That lines up with what I was already thinking, more or less.",
                "I am not sure that matches what I see around me every day.")),
        (1, 2, ("Where I come from, people argue about this at the dinner table and nobody gives an inch.",
                "My worry is that the people making these decisions never have to live with them.",
                "I want to be fair about it, but I do not want to pretend I have no opinion either.",
                "Some of my friends would say I am being stubborn about this, and maybe they are right.",
                "I read a lot about it last month and came away more confused than before.",
                "It feels like every side is talking past the other one.",
                "What bothers me most is how quickly people assume the worst about each other.")),
        (1, 1, ("What would you say to someone who sees it the other way?",
                "Can you walk me through why you think that is the better approach?",
                "Is there any evidence that would change your mind on this?",
                "How would you explain that to my neighbours?",
                "What am I missing here?",
                "Where do you think the real disagreement is?")),
    ),
    "mentor": (
        (1, 1, ("That is a reasonable thing to wonder about.",
                "Thanks for laying that out so clearly.",
                "I can see why that matters to you.",
                "It makes sense that this still feels unsettled.",
                "That is a common place to land on this question.")),
        (2, 3, ("One way to approach it is to separate the factual questions from the questions about values.",
                "It can help to ask which trade-offs you are most and least willing to accept.",
                "People who disagree with you often share more of your goals than it first appears.",
                "The evidence here is mixed, and reasonable people weigh it differently.",
                "You might look at how the policy has worked in places that have actually tried it.",
                "It is worth noticing which sources you trust and why you trust them.",
                "Stating the strongest version of the other side's view is often clarifying.")),
        (0, 1, ("What part of this matters most to you?",
                "Would it help to go through the strongest arguments on each side?",
                "Where would you like to take it from here?")),
    ),
    "judge": (
        (2, 2, ("This is a synthetic note from the sandbox mock judge.",
                "No assessment of the line was made.",
                "The text exists only to exercise the pipeline.",
                "A real judge model would answer in JSON here.")),
    ),
}


def _digest(seed, prompt: str) -> bytes:
    """The one source of randomness in a reply: sha256 of "seed|prompt", so the same request always gets
    the same reply and a new seed (every turn has one) gets a new one."""
    return hashlib.sha256(f"{seed}|{prompt}".encode("utf-8")).digest()


def free_text(role: str, digest: bytes) -> str:
    """A few sentences of synthetic, role-flavoured English: the seeker reacts and asks, the mentor
    acknowledges and advises, the judge says it is a mock."""
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    sentences = []
    for lo, hi, bank in _BANKS.get(role, _BANKS["mentor"]):
        sentences.extend(rng.sample(bank, rng.randint(lo, hi)))
    return f"[mock-{role}] " + " ".join(sentences)


def _unit(digest: bytes) -> float:
    """A number in [0, 1) from the digest."""
    return int.from_bytes(digest[8:16], "big") / 2 ** 64


def _number(spec: dict, key: str, default: float) -> float:
    """A finite numeric bound from a schema property, or the default."""
    v = spec.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return default
    return float(v)


def _value_for(schema, u: float):
    """A value satisfying a simple JSON schema, for schemas that are neither the survey's nor the judge's."""
    if not isinstance(schema, dict):
        return None
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "object" or (kind is None and isinstance(schema.get("properties"), dict)):
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        return {k: _value_for(v, u) for k, v in props.items()}
    if kind == "integer":
        lo = math.ceil(_number(schema, "minimum", 0))
        hi = math.floor(_number(schema, "maximum", max(lo, 10)))
        return lo + min(int(u * (max(hi, lo) - lo + 1)), max(hi, lo) - lo)
    if kind == "number":
        lo, hi = _number(schema, "minimum", 0.0), _number(schema, "maximum", 1.0)
        return round(lo + u * max(hi - lo, 0.0), 2)
    if kind == "string":
        return "mock"
    if kind == "boolean":
        return False
    if kind == "array":
        return []
    return None


def schema_text(schema: dict, digest: bytes) -> str:
    """The reply to a JSON-schema request. A survey item (`properties.answer`) gets an integer on its
    scale; the judge (`properties.score`) gets a score in [0, 1], two decimals, with a short rationale."""
    u = _unit(digest)
    props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    if isinstance(props.get("answer"), dict):
        spec = props["answer"]
        lo = math.ceil(_number(spec, "minimum", 1))
        hi = max(math.floor(_number(spec, "maximum", lo + 4)), lo)
        return json.dumps({"answer": lo + min(int(u * (hi - lo + 1)), hi - lo)})
    if isinstance(props.get("score"), dict):
        spec = props["score"]
        lo, hi = _number(spec, "minimum", 0.0), _number(spec, "maximum", 1.0)
        x = min(max(round(lo + u * max(hi - lo, 0.0), 2), lo), max(hi, lo))
        return json.dumps({"score": x, "rationale": f"mock rationale: a synthetic score of {x}, not a judgement."})
    return json.dumps(_value_for(schema, u))


def _common_prefix_len(a: str, b: str) -> int:
    """Length of the longest common leading substring, by binary search over slice comparisons: a
    100-turn prompt is a few hundred kilobytes and a character loop would dominate the request."""
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


# --------------------------------------------------------------------------------------------- server

class BadRequest(ValueError):
    """A request the mock refuses with HTTP 400, as llama-server would."""


class _Slot:
    """One server slot: the prompt its KV cache holds, and a lock, because a slot serves one request at a
    time (a second request for a busy slot waits, as on llama-server)."""
    def __init__(self, slot_id: int):
        self.id = slot_id
        self.prompt: str | None = None
        self.lock = threading.Lock()


class _HTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, mock: "MockLlamaServer"):
        self.mock = mock
        super().__init__(address, handler)


class _Handler(BaseHTTPRequestHandler):
    server_version = "sandbox-mock"

    def log_message(self, fmt, *args):     # quiet: a dyad run makes thousands of requests
        pass

    def _send(self, status: int, obj) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str, kind: str) -> None:
        # llama-server's error shape, so the harness's ServerError text reads the same as with a real server.
        self._send(status, {"error": {"code": status, "message": message, "type": kind}})

    def do_GET(self):
        mock = self.server.mock
        path = urlsplit(self.path).path
        if path == "/health":
            return self._send(200, {"status": "ok"})
        if path == "/props":
            return self._send(200, mock.props())
        return self._error(404, f"File Not Found: GET {path}", "not_found_error")

    def do_POST(self):
        mock = self.server.mock
        path = urlsplit(self.path).path
        routes = {"/apply-template": mock.apply_template, "/tokenize": mock.tokenize, "/completion": mock.complete}
        if path not in routes:
            return self._error(404, f"File Not Found: POST {path}", "not_found_error")
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length < 0 or length > MAX_BODY:
                raise BadRequest(f"body of {length} bytes is out of range")
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8")) if length else None
            except (UnicodeDecodeError, ValueError) as e:
                raise BadRequest(f"body is not JSON: {e}") from e
            if not isinstance(body, dict):
                raise BadRequest("body must be a JSON object")
            result = routes[path](body)
        except BadRequest as e:
            return self._error(400, str(e), "invalid_request_error")
        except Exception as e:  # noqa: BLE001 -- a mock bug must reach the harness as an HTTP error, not a hang-up
            return self._error(500, f"{type(e).__name__}: {e}", "server_error")
        return self._send(200, result)


class MockLlamaServer:
    """One fake llama-server endpoint (one role) on a daemon thread. `start()` returns its URL."""

    def __init__(self, role: str, gguf_path, *, host: str = "127.0.0.1", port: int = 0, slots: int = 8,
                 n_ctx: int = 131072, delay: float = 0.0, template: str = CHATML, now: str | None = None):
        """`delay` is seconds per generated word, so a live view of a mock run animates. `now` is the date
        `strftime_now` renders in `/apply-template`; None means today, which is what llama-server does."""
        if not isinstance(slots, int) or isinstance(slots, bool) or slots < 1:
            raise ValueError(f"slots must be an integer >= 1, got {slots!r}")
        if not isinstance(n_ctx, int) or isinstance(n_ctx, bool) or n_ctx < 1:
            raise ValueError(f"n_ctx must be an integer >= 1, got {n_ctx!r}")
        if not isinstance(delay, (int, float)) or isinstance(delay, bool) or not 0 <= delay < 60:
            raise ValueError(f"delay must be seconds per word in [0, 60), got {delay!r}")
        self.role, self.host, self.port = role, host, int(port)
        self.gguf_path = Path(gguf_path).resolve()
        self.slots, self.n_ctx, self.delay = slots, n_ctx, float(delay)
        self.template_source = template
        self.template = ChatTemplate.from_source(template, BOS, EOS)
        self.now = now
        self.alias = f"sandbox-mock-{role}"
        self.url = ""
        self._slots = [_Slot(i) for i in range(slots)]
        self._pick_lock = threading.Lock()
        self._httpd: _HTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle

    def start(self) -> str:
        """Bind and serve on a daemon thread; returns the base URL. A second call is a no-op."""
        if self._httpd is None:
            self._httpd = _HTTPServer((self.host, self.port), _Handler, self)
            self.port = self._httpd.server_address[1]
            self.url = f"http://{self.host}:{self.port}"
            self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.1},
                                            name=f"mock-{self.role}", daemon=True)
            self._thread.start()
        return self.url

    def stop(self) -> None:
        """Stop serving and close the socket, so the port is free again when this returns."""
        httpd, self._httpd = self._httpd, None
        if httpd is None:
            return
        httpd.shutdown()
        httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def running(self) -> bool:
        """True between start() and stop()."""
        return self._httpd is not None

    # -- endpoints (plain methods, so they are testable without a socket)

    def props(self) -> dict:
        """GET /props: what `harness.run.build_agent` records as the model's identity."""
        return {"model_path": str(self.gguf_path), "model_alias": self.alias, "total_slots": self.slots,
                "chat_template": self.template_source, "build_info": "sandbox-mock", "model_ftype": "mock",
                "bos_token": BOS, "eos_token": EOS,
                "default_generation_settings": {"n_ctx": self.n_ctx, "n_predict": -1, "temperature": 0.8,
                                                "top_p": 0.95, "seed": -1}}

    def apply_template(self, body: dict) -> dict:
        """POST /apply-template: the harness's own rendering, with add_generation_prompt, as llama-server."""
        messages = body.get("messages")
        if not isinstance(messages, list) or not all(
                isinstance(m, dict) and isinstance(m.get("role"), str) and isinstance(m.get("content"), str)
                for m in messages):
            raise BadRequest("messages must be a list of {role, content} objects with string values")
        now = self.now or _dt.date.today().isoformat()
        try:
            return {"prompt": render(self.template, messages, now=now)}
        except TemplateError as e:
            raise BadRequest(str(e)) from e

    def tokenize(self, body: dict) -> dict:
        """POST /tokenize: one token per whitespace-separated word (ids are a hash of the word, for show;
        only the count is used), plus BOS when add_special is set."""
        content = body.get("content")
        if not isinstance(content, str):
            raise BadRequest("content must be a string")
        tokens = [len(VOCAB) + zlib.crc32(w.encode("utf-8")) % 32000 for w in content.split()]
        if body.get("add_special"):
            tokens.insert(0, 0)
        return {"tokens": tokens}

    def _slot_for(self, id_slot, prompt: str) -> _Slot:
        """The requested slot; -1 (or none) picks the slot whose cache shares the longest prefix."""
        if id_slot is None or id_slot == -1:
            with self._pick_lock:
                return max(self._slots, key=lambda s: (_common_prefix_len(s.prompt or "", prompt), -s.id))
        if isinstance(id_slot, bool) or not isinstance(id_slot, int) or not 0 <= id_slot < self.slots:
            raise BadRequest(f"id_slot must be -1 or in [0, {self.slots}), got {id_slot!r}")
        return self._slots[id_slot]

    def complete(self, body: dict) -> dict:
        """POST /completion. The slot's cache decides prompt_n; the content is synthetic and deterministic
        in (seed, prompt); a JSON schema forces a schema-shaped reply; n_predict caps the reply in words."""
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise BadRequest("prompt must be a string (the mock does not take token arrays)")
        if body.get("stream"):
            raise BadRequest("the mock does not stream; send stream: false")
        n_predict = body.get("n_predict", -1)
        if isinstance(n_predict, bool) or not isinstance(n_predict, int):
            raise BadRequest(f"n_predict must be an integer, got {n_predict!r}")
        schema = body.get("json_schema")
        if isinstance(schema, str):
            try:
                schema = json.loads(schema)
            except ValueError as e:
                raise BadRequest(f"json_schema is not JSON: {e}") from e
        if schema is not None and not isinstance(schema, dict):
            raise BadRequest("json_schema must be an object")
        stop = body.get("stop") or []
        if not isinstance(stop, list) or not all(isinstance(s, str) for s in stop):
            raise BadRequest("stop must be a list of strings")
        seed = body.get("seed", -1)
        cache_prompt = body.get("cache_prompt", True) is not False

        n_prompt = len(prompt.split())
        if n_prompt >= self.n_ctx:
            raise BadRequest(f"the request exceeds the available context size ({n_prompt} tokens, "
                             f"n_ctx {self.n_ctx} per slot)")
        digest = _digest(seed, prompt)
        text = schema_text(schema, digest) if schema is not None else free_text(self.role, digest)
        stop_type, stopping_word = "eos", ""
        cuts = [(text.find(s), s) for s in stop if s and s in text]
        if cuts:
            at, stopping_word = min(cuts)
            text, stop_type = text[:at], "word"
        words = text.split()
        truncated = False
        cap = len(words)
        if n_predict >= 0:
            cap = min(cap, n_predict)
        room = self.n_ctx - n_prompt          # what is left of the slot's context for generated tokens
        if cap > room:
            cap, truncated = room, True
        if cap < len(words):
            text, stop_type, stopping_word = " ".join(words[:cap]), "limit", ""
        predicted_n = len(text.split())

        slot = self._slot_for(body.get("id_slot", -1), prompt)
        with slot.lock:
            prev = slot.prompt if cache_prompt else None
            if prev:
                prompt_n = len(prompt[_common_prefix_len(prev, prompt):].split())
            else:
                prompt_n = n_prompt
            # llama-server always re-evaluates at least the last prompt token, to get logits to sample from.
            prompt_n = max(prompt_n, 1) if n_prompt else 0
            slot.prompt = prompt
            if self.delay and predicted_n:
                time.sleep(self.delay * predicted_n)
        prompt_ms = 1000.0 * prompt_n / PROMPT_TOKENS_PER_SECOND
        gen_rate = (1.0 / self.delay) if self.delay else PREDICTED_TOKENS_PER_SECOND
        predicted_ms = 1000.0 * predicted_n / gen_rate
        return {
            "content": text, "id_slot": slot.id, "model": self.alias, "stop": True,
            "stop_type": stop_type, "stopping_word": stopping_word,
            "tokens_predicted": predicted_n,
            # llama-server's own accounting: tokens_evaluated is the whole prompt, tokens_cached what the
            # slot holds when the request ends (prompt plus reply); truncated means the slot ran out of context.
            "tokens_evaluated": n_prompt, "tokens_cached": n_prompt + predicted_n, "truncated": truncated,
            "timings": {"prompt_n": prompt_n, "prompt_ms": prompt_ms,
                        "prompt_per_second": PROMPT_TOKENS_PER_SECOND if prompt_n else 0.0,
                        "predicted_n": predicted_n, "predicted_ms": predicted_ms,
                        "predicted_per_second": gen_rate if predicted_n else 0.0},
        }


class MockBackend:
    """The three endpoints a study needs -- seeker, mentor, judge -- each with its own GGUF (so each has
    its own sha256, and `score` accepts the judge as a third model) and its own port. `base_port` 0 gives
    each an ephemeral port; otherwise seeker, mentor and judge get base_port, +1 and +2."""

    def __init__(self, directory, *, slots: int = 8, delay: float = 0.0, base_port: int = 0,
                 host: str = "127.0.0.1"):
        """`directory` holds the mock GGUFs (workspace/mock in the sandbox)."""
        if not isinstance(base_port, int) or isinstance(base_port, bool) or not 0 <= base_port <= 65533:
            raise ValueError(f"base_port must be 0 or a port that leaves room for +2, got {base_port!r}")
        self.directory = Path(directory)
        self.slots, self.delay, self.base_port, self.host = slots, delay, base_port, host
        self._servers: dict[str, MockLlamaServer] = {}
        self._lock = threading.RLock()

    def start(self) -> dict:
        """Write the GGUFs and start the three servers; returns info(). A second call is a no-op. If one
        server cannot bind, the others are stopped again and the OSError is raised."""
        with self._lock:
            if self._servers:
                return self.info()
            started: dict[str, MockLlamaServer] = {}
            try:
                for i, role in enumerate(ROLES):
                    gguf = write_mock_gguf(self.directory / f"sandbox-mock-{role}.gguf", role)
                    server = MockLlamaServer(role, gguf, host=self.host,
                                             port=self.base_port + i if self.base_port else 0,
                                             slots=self.slots, delay=self.delay)
                    server.start()
                    started[role] = server
            except BaseException:
                for server in started.values():
                    server.stop()
                raise
            self._servers = started
            return self.info()

    def stop(self) -> None:
        """Stop every server; the ports are free when this returns."""
        with self._lock:
            servers, self._servers = self._servers, {}
        for server in servers.values():
            server.stop()

    @property
    def running(self) -> bool:
        """True while the servers are up."""
        with self._lock:
            return bool(self._servers)

    def info(self) -> dict:
        """{"running", "seeker", "mentor", "judge", "slots", "delay"}; each role {"url", "gguf_path"} or None."""
        with self._lock:
            out: dict = {"running": bool(self._servers), "slots": self.slots, "delay": self.delay}
            for role in ROLES:
                s = self._servers.get(role)
                out[role] = {"url": s.url, "gguf_path": str(s.gguf_path)} if s else None
            return out


def config_snippet(info: dict) -> dict:
    """The seeker/mentor/judge block of a harness config that points at a running mock backend."""
    return {role: {"url": info[role]["url"], "gguf_path": info[role]["gguf_path"]}
            for role in ROLES if info.get(role)}


def main(argv: list[str] | None = None) -> int:
    """Serve the mock backend until Ctrl-C, printing the URLs, the GGUF paths and a config snippet."""
    ap = argparse.ArgumentParser(prog="python -m sandbox.mock_server",
                                 description="Fake llama-server endpoints (seeker, mentor, judge) for the sandbox. "
                                             "Synthetic replies: never for data.")
    ap.add_argument("--dir", default="workspace/mock", help="where the mock GGUFs are written")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18201, help="seeker port; mentor +1, judge +2; 0 = ephemeral")
    ap.add_argument("--slots", type=int, default=8, help="slots per server")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds per generated word")
    a = ap.parse_args(argv)
    try:
        backend = MockBackend(a.dir, slots=a.slots, delay=a.delay, base_port=a.port, host=a.host)
        info = backend.start()
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print("sandbox mock llama-server -- synthetic replies, never for data")
    for role in ROLES:
        print(f"  {role:<7} {info[role]['url']}  {info[role]['gguf_path']}")
    print("config snippet (merge into a harness config):")
    print(json.dumps(config_snippet(info), indent=2))
    print("Ctrl-C to stop", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        backend.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
