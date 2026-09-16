"""Free text to a numeric scale: the salvage path behind the survey's JSON-schema answers.

The survey forces `{"answer": <int>}` with a JSON schema (harness/survey.py), so on a healthy server
the reply parses on the first branch here and the method is `json`. The other methods exist for the
replies that path cannot produce a number from: a reply cut off by `n_predict`, a server that ignored
the grammar, a model that answered `"3"` instead of `3`, or a bare `4` from a re-administered item.
Every salvaged answer is labelled with how it was found, so the analysis can report the salvage rate
and decide whether to keep those rows. The parser never guesses: two in-range candidates with nothing
to choose between them is `ambiguous`, not the first one.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass

# How the number was found, or why it was not. `value` is None exactly for the last three.
METHODS = ("json", "labelled", "bare", "ambiguous", "out_of_range", "none")

_NUMBER = r"-?\d+(?:\.\d+)?"
# A number token: not preceded by a word character or a dot, so "4-5" is 4 and 5 (not 4 and -5) and
# "2.5" is one token, while "-1" at the start of a reply keeps its sign.
_TOKEN = re.compile(rf"(?<![\w.])({_NUMBER})")
_THINK_CLOSED = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN = re.compile(r"<think>.*\Z", re.DOTALL)
# "4/5", "4 out of 5", "7/10": the denominator has to be the top of the scale.
_FRACTION = re.compile(rf"({_NUMBER})\s*(?:/|out of)\s*(\d+)\b")
# "3 or 4", "between 3 and 4", "3-4": two candidates the model declined to choose between.
_PAIR = re.compile(rf"({_NUMBER})\s*(?:or|and|to|-|–|—)\s*({_NUMBER})")
# A label word, then at most 40 digit-free characters, then the number.
_LABEL = re.compile(
    r"\b(?:answer|rating|rate|rated|score|response|choose|chose|pick|picked|select|say|go with|give)\b"
    rf"[^0-9\n]{{0,40}}?({_NUMBER})", re.IGNORECASE)
_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
          "eight": 8, "nine": 9, "ten": 10}


@dataclass(frozen=True)
class Parsed:
    """A parsed scale answer: the integer, or None, and the method that found it (one of METHODS)."""
    value: int | None
    method: str


def _as_int(token: str) -> int | None:
    """The token as an int when it is integral ("4", "4.0", "-1"); None for "4.5"."""
    f = float(token)
    return int(f) if f == int(f) else None


def _strip_scale_restatement(text: str, lo: int, hi: int) -> str:
    """Remove "1 to 5", "0-10", "1 (strongly disagree) to 5 (strongly agree)": a model restating the
    scale before answering must not contribute two more candidates."""
    pat = re.compile(rf"(?<![\w.]){lo}\b\s*(?:\([^)]*\)\s*)?(?:to|through|-|–|—)\s*{hi}\b(?:\s*\([^)]*\))?")
    return pat.sub(" ", text)


def _json_answer(text: str) -> tuple[int | None, bool]:
    """(answer, found): the JSON-schema path, exactly as strict as the schema. `found` is True when the
    reply is a JSON object whose `answer` is an int (bools excluded); the caller range-checks it."""
    try:
        v = json.loads(text).get("answer")
    except (ValueError, AttributeError):
        return None, False
    if isinstance(v, bool) or not isinstance(v, int):
        return None, False
    return v, True


def parse_scale_answer(text: str, lo: int, hi: int) -> Parsed:
    """Parse one survey reply into an integer on [lo, hi]. The JSON-schema path first (`json`); then
    free-text salvage: a fraction over the scale top or a labelled number (`labelled`), else the single
    in-range number in the reply (`bare`). Two in-range candidates with nothing to choose between them
    is `ambiguous`; a number that is only ever outside the scale is `out_of_range`; no number at all is
    `none`. Text inside a <think> block, closed or not, is never an answer."""
    if lo > hi:
        raise ValueError(f"scale min {lo} is above max {hi}")
    in_range = lambda v: lo <= v <= hi  # noqa: E731

    v, found = _json_answer(text)
    if found:
        return Parsed(v, "json") if in_range(v) else Parsed(None, "out_of_range")

    body = _THINK_OPEN.sub("", _THINK_CLOSED.sub("", text))
    body = _strip_scale_restatement(body, lo, hi)

    m = _FRACTION.search(body)
    if m and int(m.group(2)) == hi:
        v = _as_int(m.group(1))
        if v is not None:
            return Parsed(v, "labelled") if in_range(v) else Parsed(None, "out_of_range")

    for m in _PAIR.finditer(body):
        a, b = _as_int(m.group(1)), _as_int(m.group(2))
        if a is not None and b is not None and a != b and in_range(a) and in_range(b):
            return Parsed(None, "ambiguous")

    m = _LABEL.search(body)
    if m:
        v = _as_int(m.group(1))
        if v is not None:
            return Parsed(v, "labelled") if in_range(v) else Parsed(None, "out_of_range")

    tokens = _TOKEN.findall(body)
    ints = [i for i in (_as_int(t) for t in tokens) if i is not None]
    candidates = sorted({i for i in ints if in_range(i)})
    if len(candidates) == 1:
        return Parsed(candidates[0], "bare")
    if len(candidates) > 1:
        return Parsed(None, "ambiguous")
    if ints:
        return Parsed(None, "out_of_range")
    if tokens:
        # Only non-integral numbers ("4.5"): a point between two scale values.
        return Parsed(None, "ambiguous")
    word = re.sub(r"[^a-z]", "", body.lower())
    if word in _WORDS:
        v = _WORDS[word]
        return Parsed(v, "bare") if in_range(v) else Parsed(None, "out_of_range")
    return Parsed(None, "none")
