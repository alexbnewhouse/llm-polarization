"""The free-text-to-scale parser, run against 50 hand-written model responses and the red-team's four
adversarial ones (parser_cases.jsonl)."""
import json
from pathlib import Path
import pytest
from harness import parser
from harness.parser import Parsed, parse_scale_answer

CASES_PATH = Path(__file__).with_name("parser_cases.jsonl")
FAILURE_METHODS = ("ambiguous", "out_of_range", "none")


def load_cases() -> list[dict]:
    return [json.loads(line) for line in CASES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_fixture_holds_fifty_hand_written_responses_and_the_red_teams_four():
    cases = load_cases()
    assert len(cases) == 54 and sum("red-team L1" in c["note"] for c in cases) == 4
    assert len({c["text"] for c in cases}) == 54, "every hand-written response is distinct"


def test_fixture_uses_only_declared_methods_and_null_iff_failure():
    for c in load_cases():
        assert c["method"] in parser.METHODS, c
        assert (c["value"] is None) == (c["method"] in FAILURE_METHODS), c


@pytest.mark.parametrize("case", load_cases(), ids=lambda c: repr(c["text"][:36]))
def test_hand_written_response(case):
    got = parse_scale_answer(case["text"], case["lo"], case["hi"])
    assert isinstance(got, Parsed)
    assert (got.value, got.method) == (case["value"], case["method"]), case["note"]


def test_json_method_means_the_schema_was_honoured():
    # Only a JSON object with an integer `answer` inside the scale is `json`; everything else that
    # yields a number is salvage and must say so, because the analysis reports the salvage rate.
    assert parse_scale_answer('{"answer": 2}', 1, 5) == Parsed(2, "json")
    assert parse_scale_answer('{"answer": "2"}', 1, 5).method != "json"
    assert parse_scale_answer('2', 1, 5).method != "json"


def test_scale_bounds_are_respected_not_assumed():
    assert parse_scale_answer("5", 1, 5) == Parsed(5, "bare")
    assert parse_scale_answer("5", 0, 4) == Parsed(None, "out_of_range")
    assert parse_scale_answer("5 or 6", 0, 10) == Parsed(None, "ambiguous")
    assert parse_scale_answer("5 or 6", 1, 5) == Parsed(5, "bare"), "6 is out of range, so 5 is the only candidate"
    with pytest.raises(ValueError):
        parse_scale_answer("3", 5, 1)


def test_unclosed_thinking_block_is_not_an_answer():
    assert parse_scale_answer("<think>I would say 4", 1, 5) == Parsed(None, "none")


def test_the_answer_marker_wins_and_reasoning_is_never_the_answer():
    # Red-team L1, beyond the four fixture cases: an answer marker pointing at two numbers is still
    # ambiguous, the last marker wins, and a harmony reply with no final channel has no answer.
    assert parse_scale_answer("Answer: 3 or 4", 1, 5) == Parsed(None, "ambiguous")
    assert parse_scale_answer("Answer: 3. Final answer: 4", 1, 5) == Parsed(4, "labelled")
    assert parse_scale_answer("<|channel|>analysis<|message|>Rating 5<|end|>", 1, 5) == Parsed(None, "none")
    assert parse_scale_answer("I'd say 3.\n<think>maybe 5</think>", 1, 5) == Parsed(3, "labelled")
    assert parse_scale_answer("Answer: 9", 1, 5) == Parsed(None, "out_of_range")
