"""Minimal llama-server client. One instance per endpoint. No retries: a failed call is the caller's
problem to log, because a silent retry could hide a lost KV cache."""
from __future__ import annotations
import json
import urllib.error, urllib.request
from dataclasses import dataclass


class ServerError(Exception):
    """Exception raised when the llama-server HTTP request fails or the server responds with an error."""
    pass


@dataclass
class Completion:
    """Result of a completion request: text, finish_reason, token counts, timings, and raw response."""
    text: str
    finish_reason: str
    prompt_n: int
    predicted_n: int
    timings: dict
    raw: dict


def parse_completion(raw: dict) -> Completion:
    """Extract completion result from llama-server response dict; determine finish_reason from stop_type or stopped_limit."""
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
    """HTTP client for llama-server at a given URL; manages timeouts and wraps network errors as ServerError."""

    def __init__(self, url: str, timeout: float = 600):
        """Initialize client with llama-server URL and optional timeout in seconds (default 600)."""
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, body: dict) -> dict:
        """Make POST request to path with JSON body; wrap any network/HTTP error as ServerError."""
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
        """Make GET request to path; wrap any network/HTTP error as ServerError; timeout capped at 10s."""
        try:
            with urllib.request.urlopen(self.url + path, timeout=min(self.timeout, 10)) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise ServerError(f"GET {path}: HTTP {e.code}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise ServerError(f"GET {path}: {e}") from e

    def health(self) -> bool:
        """Check if server is healthy; return True if /health endpoint returns status ok, False on any error."""
        try:
            return self._get("/health").get("status") == "ok"
        except ServerError:
            return False

    def props(self) -> dict:
        """Fetch server properties from /props endpoint and return as dict."""
        return self._get("/props")

    def apply_template(self, messages: list[dict]) -> str:
        """Apply llama-server chat template to messages list and return rendered prompt string."""
        return self._post("/apply-template", {"messages": messages})["prompt"]

    def tokenize(self, text: str) -> int:
        """Tokenize text and return token count."""
        return len(self._post("/tokenize", {"content": text, "add_special": False}).get("tokens", []))

    def complete(self, prompt: str, *, id_slot: int, seed: int, n_predict: int, temperature: float,
                 top_p: float = 0.95, json_schema: dict | None = None, cache_prompt: bool = True,
                 stop: list[str] | None = None) -> Completion:
        """Request text completion with slot pinning; omit json_schema and stop if None; return Completion object."""
        body = {"prompt": prompt, "id_slot": id_slot, "seed": seed, "n_predict": n_predict,
                "temperature": temperature, "top_p": top_p, "cache_prompt": cache_prompt}
        if json_schema is not None:
            body["json_schema"] = json_schema
        if stop:
            body["stop"] = stop
        return parse_completion(self._post("/completion", body))
