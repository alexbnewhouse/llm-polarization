"""Chat-template rendering with jinja2, the check fixtures, the parity check, and GGUF-reading errors."""
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
    def apply_template(self, messages, chat_template_kwargs=None):
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


def test_user_first_fixture_has_no_system_message():
    # The mentor's only shape. Parity proved on the system-first fixture proves nothing about it.
    assert templates.FIXTURE_MESSAGES_USER_FIRST
    assert all(m["role"] != "system" for m in templates.FIXTURE_MESSAGES_USER_FIRST)


def test_missing_gguf_dependency_becomes_a_template_error(tmp_path, monkeypatch):
    # gguf-py is not pip-installed and needs numpy; a bare ModuleNotFoundError traceback out of the first
    # command the researcher runs is not an error message.
    import sys
    monkeypatch.setattr(sys, "path", list(sys.path))          # restored, so the fake package cannot leak
    monkeypatch.delitem(sys.modules, "gguf", raising=False)
    pkg = tmp_path / "gguf"; pkg.mkdir()
    (pkg / "__init__.py").write_text("import numpy_is_not_installed_here\n")
    with pytest.raises(TemplateError) as ei:
        templates._gguf_module(str(tmp_path))
    assert "numpy_is_not_installed_here" in str(ei.value) and str(tmp_path) in str(ei.value)


class _Server:
    """A fake /apply-template that renders `source` its own way: on `day`, honouring chat_template_kwargs
    or not, and dropping a leading BOS or not, as llama-server may."""
    def __init__(self, source, bos="", day="2026-09-08", kwargs=True, strip_bos=False):
        self.tpl, self.day = ChatTemplate.from_source(source, bos=bos), day
        self.kwargs, self.strip_bos, self.seen = kwargs, strip_bos, []

    def apply_template(self, messages, chat_template_kwargs=None):
        self.seen.append(chat_template_kwargs)
        kw = chat_template_kwargs if self.kwargs and chat_template_kwargs else {}
        out = render(self.tpl, messages, now=self.day, **kw)
        return out[len(self.tpl.bos):] if self.strip_bos and out.startswith(self.tpl.bos) else out


def test_a_dated_template_is_compared_on_the_servers_date_and_only_the_date_may_differ(dated):
    # Red-team H4: llama-server prints its wall-clock date and takes no override, so a pinned `now` failed
    # parity on every other day. The comparison renders ours with the server's date; the prompts keep `now`.
    import datetime
    today = datetime.date.today().strftime("%Y-%m-%d")
    tpl = ChatTemplate.from_source(dated)
    ok, ours, theirs, notes = templates.parity_detail(tpl, _Server(dated, day=today), MSGS, now="2000-01-01")
    assert ok and notes == [f"date_adjusted {today}"] and "2000-01-01" in ours and today in theirs
    other = dated.replace("Knowledge cutoff: 2024-06", "Knowledge cutoff: 2025-01")
    ok, _, _, notes = templates.parity_detail(tpl, _Server(other, day=today), MSGS, now="2000-01-01")
    assert not ok and notes == []
    assert templates.date_only_diff("date 2026-09-08 end", "date 2026-10-18 end")
    assert not templates.date_only_diff("a 2026-09-08, then more text b", "a 2026-09-28, then more text c")
    assert not templates.date_only_diff("x" * 40, "y" * 40)


def test_a_leading_bos_the_server_drops_is_ignored_and_said(chatml):
    # Red-team L5: llama.cpp strips the leading BOS from the rendered chat prompt when the vocab adds one.
    source = "{{ bos_token }}" + chatml
    tpl = ChatTemplate.from_source(source, bos="<s>")
    ok, ours, theirs, notes = templates.parity_detail(tpl, _Server(source, bos="<s>", strip_bos=True), MSGS,
                                                      now="2026-09-08")
    assert ok and notes == ["bos_stripped"] and ours == "<s>" + theirs
    assert templates.parity_detail(tpl, _Server(source, bos="<s>"), MSGS, now="2026-09-08")[3] == []
    trimmed = _Server(source, bos="<s>")
    trimmed.apply_template = lambda m, chat_template_kwargs=None: render(trimmed.tpl, m)[4:]
    assert not templates.parity_detail(tpl, trimmed, MSGS, now="2026-09-08")[0]


def test_both_sides_render_with_the_configs_enable_thinking(chatml):
    # Red-team L4: parity rendered ours with enable_thinking False and sent the server nothing, so with a
    # hybrid-thinking template the gate checked another rendering than the run uses.
    source = chatml + ("{%- if add_generation_prompt and not enable_thinking %}"
                       "{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}")
    tpl = ChatTemplate.from_source(source)
    srv = _Server(source)
    ok, ours, _, _ = templates.parity_detail(tpl, srv, MSGS, now="2026-09-08", enable_thinking=True)
    assert ok and srv.seen[-1] == {"enable_thinking": True} and not ours.endswith("</think>\n\n")
    deaf = _Server(source, kwargs=False)                     # a server that ignores the kwarg
    assert not templates.parity_detail(tpl, deaf, MSGS, now="2026-09-08", enable_thinking=True)[0]
    assert templates.parity_detail(tpl, deaf, MSGS, now="2026-09-08", enable_thinking=False)[0]
