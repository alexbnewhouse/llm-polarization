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
