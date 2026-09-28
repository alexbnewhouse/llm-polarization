"""The arms' real chat templates, read out of their GGUFs (gap audit F13): the unit tests use toy mimics
(conftest.py), and only the files on the box hold the real ones. Live-only, like test_live.py: set

    HARNESS_TEMPLATE_GGUFS=qwen3=/models/qwen3.6.gguf,gpt-oss=/models/gpt-oss-20b.gguf,olmo=/models/olmo-3.gguf
    GGUF_PY_PATH=<llama.cpp>/gguf-py

and each family named there is checked; one not named is skipped. Any other name (a seeker candidate:
seeker=/models/<candidate>.gguf) is checked the same way. What must hold: the mentor's shape renders, the
prompt depends on the pinned `now`, never the wall clock, and, for a model that may be the seeker, the
reminder is rendered after the last history message with the persona kept (the reinforced delivery).

gpt-oss's template keeps only the first system message (llama.cpp models/templates/
openai-gpt-oss-120b.jinja, "Extract developer message"), so its reminder check is an expected failure: it
could not be a reinforced seeker. It is a mentor arm, and the mentor gets no system message."""
import os
import pytest
from harness import run as R
from harness.dialogue import AgentHandle
from harness.templates import date_only_diff, read_template_from_gguf, render
from harness.transcript import MENTOR, SEEKER, Transcript

FAMILIES = ("qwen3", "gpt-oss", "olmo")
NOW = "2026-09-08"
# Models whose template drops a later system message, so they can never be the seeker.
NOT_A_SEEKER = {"gpt-oss": "the template keeps only the first system message"}


def template_ggufs(value: str) -> dict[str, str]:
    """HARNESS_TEMPLATE_GGUFS as {name: path}: comma-separated name=path pairs."""
    out = {}
    for pair in filter(None, (p.strip() for p in (value or "").split(","))):
        name, sep, path = pair.partition("=")
        if not sep or not name.strip() or not path.strip():
            raise ValueError(f"HARNESS_TEMPLATE_GGUFS: {pair!r} is not name=path")
        out[name.strip()] = path.strip()
    return out


def test_template_ggufs_reads_name_path_pairs():
    assert template_ggufs("") == {}
    assert template_ggufs("qwen3=/a.gguf, olmo=/m/b.gguf") == {"qwen3": "/a.gguf", "olmo": "/m/b.gguf"}
    with pytest.raises(ValueError):
        template_ggufs("qwen3")


def _names() -> list[str]:
    try:
        extra = [n for n in template_ggufs(os.environ.get("HARNESS_TEMPLATE_GGUFS", "")) if n not in FAMILIES]
    except ValueError:
        extra = []
    return list(FAMILIES) + extra


@pytest.fixture(params=_names())
def tpl(request):
    ggufs = template_ggufs(os.environ.get("HARNESS_TEMPLATE_GGUFS", ""))
    if request.param not in ggufs:
        pytest.skip(f"set HARNESS_TEMPLATE_GGUFS with {request.param}=<gguf> to check its real template")
    if request.param in NOT_A_SEEKER and "reminder" in request.function.__name__:
        request.applymarker(pytest.mark.xfail(reason=NOT_A_SEEKER[request.param], strict=False))
    return read_template_from_gguf(ggufs[request.param], gguf_py_path=os.environ.get("GGUF_PY_PATH"))


def _dialogue(mode: str = "reinforced") -> Transcript:
    t = Transcript("d", "You are Dana, a rancher in Montana. PERSONA-MARKER.", "REMINDER-MARKER: I am Dana.", mode)
    for turn in (1, 2, 3):
        t.append(turn, SEEKER, f"Seeker line {turn}.")
        t.append(turn, MENTOR, f"Mentor line {turn}.")
    return t


def test_the_reminder_is_rendered_last_and_the_persona_kept(tpl):
    row = R._trailing_system_row(AgentHandle(SEEKER, None, tpl, "", 0), NOW, False)
    assert row[1] is True, row[2]
    prompt = render(tpl, _dialogue().view_for(SEEKER), now=NOW)
    assert "PERSONA-MARKER" in prompt and prompt.index("PERSONA-MARKER") < prompt.index("Seeker line 1.")
    assert prompt.rindex("REMINDER-MARKER") > prompt.rindex("Mentor line 3.")
    assert "REMINDER-MARKER" not in render(tpl, _dialogue("once").view_for(SEEKER), now=NOW)


def test_the_mentor_shape_renders_without_the_persona(tpl):
    prompt = render(tpl, _dialogue().view_for(MENTOR), now=NOW)
    assert "Mentor line 3." in prompt and "PERSONA-MARKER" not in prompt and "REMINDER-MARKER" not in prompt


def test_the_prompt_follows_the_pinned_date_not_the_wall_clock(tpl):
    messages = _dialogue().view_for(SEEKER)
    first = render(tpl, messages, now=NOW)
    assert render(tpl, messages, now=NOW) == first                      # deterministic
    later = render(tpl, messages, now="2027-01-15")
    if "strftime_now" not in tpl.source:
        assert later == first
    else:
        # the date is printed, it is the pinned one, and nothing but the date moves with it
        assert later != first and date_only_diff(first, later, max_segments=tpl.source.count("strftime_now"))
