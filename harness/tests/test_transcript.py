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
