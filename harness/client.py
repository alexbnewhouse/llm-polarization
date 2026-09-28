"""Minimal llama-server client. One instance per endpoint. No retries: a failed call is the caller's
problem to log, because a silent retry could hide a lost KV cache."""
from __future__ import annotations
import json
import urllib.error, urllib.request
from dataclasses import dataclass


class ServerError(Exception):
    """A llama-server request failed: connection, timeout, or an HTTP error status."""
    pass


@dataclass
class Completion:
    """Result of a completion request: text, finish_reason, token counts, timings, and raw response.
    `truncated` / `tokens_evaluated` / `tokens_cached` are the server's own account of the slot's context
    and are None on builds that do not report them; `truncated` is the difference between a turn that hit
    the n_predict cap and a dyad that has run out of context."""
    text: str
    finish_reason: str
    prompt_n: int | None            # None when the server reported no timings: unknown, not zero
    predicted_n: int | None
    timings: dict
    raw: dict
    truncated: bool | None = None
    tokens_evaluated: int | None = None
    tokens_cached: int | None = None


def parse_completion(raw: dict) -> Completion:
    """Build a Completion from a llama-server /completion response. finish_reason is "length" when the
    n_predict cap stopped generation, else "stop"; "error" is set by the caller, never here."""
    timings = raw.get("timings") or {}
    # Newer llama-server reports stop_type; older builds reported stopped_limit. Accept either so a
    # server upgrade does not silently mislabel truncated turns as clean stops.
    limit = raw.get("stop_type") == "limit" or bool(raw.get("stopped_limit"))
    return Completion(
        text=raw.get("content", ""),
        finish_reason="length" if limit else "stop",
        prompt_n=None if timings.get("prompt_n") is None else int(timings["prompt_n"]),
        predicted_n=None if timings.get("predicted_n") is None else int(timings["predicted_n"]),
        timings=timings,
        raw=raw,
        truncated=raw.get("truncated"),
        tokens_evaluated=raw.get("tokens_evaluated"),
        tokens_cached=raw.get("tokens_cached"),
    )


class LlamaClient:
    """HTTP client for one llama-server URL. Every network or HTTP failure surfaces as ServerError."""

    def __init__(self, url: str, timeout: float = 600):
        """`timeout` is in seconds and applies to POSTs; GETs are capped at 10 s."""
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, body: dict) -> dict:
        """POST a JSON body to `path` and return the decoded JSON reply."""
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
        """GET `path` and return the decoded JSON reply, with the timeout capped at 10 s."""
        try:
            with urllib.request.urlopen(self.url + path, timeout=min(self.timeout, 10)) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise ServerError(f"GET {path}: HTTP {e.code}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise ServerError(f"GET {path}: {e}") from e

    def health(self) -> bool:
        """True when /health reports status ok; False on any other reply or on any error."""
        try:
            return self._get("/health").get("status") == "ok"
        except ServerError:
            return False

    def props(self) -> dict:
        """The server's /props: model path and alias, build, slots, chat template, sampler defaults."""
        return self._get("/props")

    def slots(self) -> list[dict]:
        """The server's /slots: one entry per slot with its `id` and `is_processing`. ServerError when the
        server was started with --no-slots."""
        return self._get("/slots")

    def apply_template(self, messages: list[dict]) -> str:
        """The server's own rendering of `messages` (/apply-template), which the parity check compares to."""
        return self._post("/apply-template", {"messages": messages})["prompt"]

    def tokenize(self, text: str) -> int:
        """How many tokens `text` is. Returns the count, not the token ids."""
        return len(self._post("/tokenize", {"content": text, "add_special": False}).get("tokens", []))

    def complete(self, prompt: str, *, id_slot: int, seed: int, n_predict: int, temperature: float,
                 top_p: float = 0.95, json_schema: dict | None = None, cache_prompt: bool = True,
                 stop: list[str] | None = None) -> Completion:
        """One /completion request on slot `id_slot`. `json_schema` is sent only when given and `stop` only
        when non-empty; `cache_prompt` lets the slot reuse its KV cache for the shared prefix."""
        body = {"prompt": prompt, "id_slot": id_slot, "seed": seed, "n_predict": n_predict,
                "temperature": temperature, "top_p": top_p, "cache_prompt": cache_prompt}
        if json_schema is not None:
            body["json_schema"] = json_schema
        if stop:
            body["stop"] = stop
        return parse_completion(self._post("/completion", body))
