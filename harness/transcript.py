"""Canonical transcript of one dyad and the egocentric projection each agent generates from."""
from __future__ import annotations
from dataclasses import dataclass, field

SEEKER = "seeker"
MENTOR = "mentor"
AGENTS = (SEEKER, MENTOR)
PERSONA_MODES = ("once", "reinforced")


def partner_of(agent: str) -> str:
    """Return the other agent in the dyad."""
    return MENTOR if agent == SEEKER else SEEKER


@dataclass
class Message:
    """A single turn message from one agent in the transcript."""
    turn: int
    agent: str
    text: str


@dataclass
class Transcript:
    """Canonical transcript of a dyad conversation with egocentric projection views."""
    dyad_id: str
    seeker_system: str
    reminder: str | None
    persona_mode: str
    messages: list[Message] = field(default_factory=list)

    def __post_init__(self):
        if self.persona_mode not in PERSONA_MODES:
            raise ValueError(f"persona_mode must be one of {PERSONA_MODES}, got {self.persona_mode!r}")

    def append(self, turn: int, agent: str, text: str) -> Message:
        """Add a message to the transcript and return it."""
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
        """Return all text lines spoken by the given agent."""
        return [m.text for m in self.messages if m.agent == agent]

    def last_line_of(self, agent: str) -> str | None:
        """Return the most recent text line from the given agent, or None if they have not spoken."""
        lines = self.lines_of(agent)
        return lines[-1] if lines else None

    @property
    def n_messages(self) -> int:
        """Total number of messages in the transcript."""
        return len(self.messages)
