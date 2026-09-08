"""One dyad, end to end: seeker opens, agents alternate, every generation logged with provenance."""
from __future__ import annotations
from dataclasses import dataclass
from harness.client import ServerError
from harness.log import JsonlWriter, derive_seed, now_iso, sha256_text
from harness.templates import ChatTemplate, render
from harness.transcript import Transcript, SEEKER, MENTOR


@dataclass
class AgentHandle:
    """Bundles one agent's client, template and model identity so the runner can address it by name."""
    name: str
    client: object
    template: ChatTemplate
    model_sha256: str
    slot: int
    alias: str = ""


@dataclass
class GenSettings:
    """Generation parameters shared by every completion request in a run."""
    temperature: float = 0.7
    top_p: float = 0.95
    n_predict: int = 300
    now: str = "2026-09-08"
    enable_thinking: bool = False


@dataclass
class DyadSpec:
    """Describes one dyad to run: its condition, persona text, persona mode, seed and turn count."""
    dyad_id: str
    condition: dict
    persona_text: str
    persona_reminder: str
    persona_mode: str
    seed: int
    n_turns: int

    @classmethod
    def from_row(cls, row: dict) -> "DyadSpec":
        """Build a DyadSpec from a plain dict row, e.g. one read back from a dyads.jsonl file."""
        return cls(row["dyad_id"], dict(row.get("condition") or {}), row["persona_text"],
                   row.get("persona_reminder", ""), row.get("persona_mode", "reinforced"),
                   int(row.get("seed", 0)), int(row["n_turns"]))


class DialogueError(Exception):
    """Raised when a completion request fails mid-dyad; carries the dyad/turn/agent for the caller to log."""
    def __init__(self, dyad_id: str, turn: int, agent: str, cause: Exception):
        """Record the dyad_id, turn and agent at which the underlying cause occurred."""
        super().__init__(f"{dyad_id} turn {turn} {agent}: {cause}")
        self.dyad_id, self.turn, self.agent, self.cause = dyad_id, turn, agent, cause


def _common_prefix_len(a: str, b: str) -> int:
    """Return the length of the longest common leading substring of a and b."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def expected_new_tokens(client, prompt: str, previous_prompt: str | None) -> int:
    """Tokens the server should have to prefill if its slot still holds previous_prompt. The shared prefix
    is measured in characters, which can end mid-token, so this is an estimate -- that is what the
    64-token cache_margin absorbs. It only sets the cache_warning flag; it never changes what is sent."""
    if not previous_prompt:
        return client.tokenize(prompt)
    k = _common_prefix_len(prompt, previous_prompt)
    return client.tokenize(prompt) - client.tokenize(prompt[:k])


class DialogueRunner:
    """Drives one dyad through its turns, generating, logging and appending each message in order."""
    def __init__(self, run_id: str, run_seed: int, seeker: AgentHandle, mentor: AgentHandle,
                 settings: GenSettings, turns_log: JsonlWriter, clock=now_iso, cache_margin: int = 64):
        """Wire up the agents, shared settings and turn log this runner will use for every dyad it runs."""
        self.run_id, self.run_seed = run_id, run_seed
        self.agents = {SEEKER: seeker, MENTOR: mentor}
        self.settings, self.turns_log, self.clock, self.cache_margin = settings, turns_log, clock, cache_margin

    def run(self, spec: DyadSpec, attempt: int) -> Transcript:
        """Run one dyad from an empty transcript through spec.n_turns of alternating seeker/mentor turns."""
        transcript = Transcript(spec.dyad_id, spec.persona_text, spec.persona_reminder or None, spec.persona_mode)
        last_prompt: dict[str, str | None] = {SEEKER: None, MENTOR: None}
        for turn in range(1, spec.n_turns + 1):
            for agent in (SEEKER, MENTOR):
                text = self._generate(agent, transcript, spec, attempt, turn, last_prompt)
                transcript.append(turn, agent, text)
        return transcript

    def _generate(self, agent: str, transcript: Transcript, spec: DyadSpec, attempt: int, turn: int,
                  last_prompt: dict) -> str:
        """Render one agent's prompt, request a completion, log the provenance row, and return the reply
        text -- and record this prompt in last_prompt[agent] (the caller's dict, written here) so the next
        turn can tell how much of it the slot should still hold."""
        h = self.agents[agent]
        s = self.settings
        prompt = render(h.template, transcript.view_for(agent), now=s.now, enable_thinking=s.enable_thinking)
        seed = derive_seed(self.run_seed, spec.seed, spec.dyad_id, attempt, turn, agent)
        row = {"run_id": self.run_id, "dyad_id": spec.dyad_id, "attempt": attempt, "turn": turn, "agent": agent,
               "model_sha256": h.model_sha256, "persona_mode": spec.persona_mode,
               "id_slot": h.slot, "temperature": s.temperature, "top_p": s.top_p, "n_predict": s.n_predict,
               "prompt_sha256": sha256_text(prompt), "prompt_chars": len(prompt), "seed": seed}
        try:
            expected = expected_new_tokens(h.client, prompt, last_prompt[agent])
            comp = h.client.complete(prompt, id_slot=h.slot, seed=seed, n_predict=s.n_predict,
                                     temperature=s.temperature, top_p=s.top_p, cache_prompt=True)
        except ServerError as e:
            row.update({"prompt_n": None, "predicted_n": None, "expected_new": None, "cache_warning": None,
                        "truncated": None, "tokens_evaluated": None, "tokens_cached": None,
                        "finish_reason": "error", "text": "", "timings": {}, "error": str(e),
                        "adherence": None, "ts": self.clock()})
            self.turns_log.write(row)
            raise DialogueError(spec.dyad_id, turn, agent, e) from e
        last_prompt[agent] = prompt
        row.update({"prompt_n": comp.prompt_n, "predicted_n": comp.predicted_n, "expected_new": expected,
                    "cache_warning": comp.prompt_n > expected + self.cache_margin,
                    # The server's own context accounting: `truncated` true means this slot ran out of
                    # context, which finish_reason "length" (the n_predict cap) does not distinguish.
                    "truncated": comp.truncated, "tokens_evaluated": comp.tokens_evaluated,
                    "tokens_cached": comp.tokens_cached,
                    "finish_reason": comp.finish_reason, "text": comp.text, "timings": comp.timings,
                    "adherence": None, "ts": self.clock()})
        self.turns_log.write(row)
        return comp.text
