"""FakeClient: an in-process stand-in for LlamaClient that the unit tests drive instead of a server."""
from harness.client import Completion, ServerError
from harness.templates import ChatTemplate, render
from harness.tests.conftest import CHATML


class FakeClient:
    """Replies from a fixed list, in turn, and records every completion call in `calls`. Tokens are
    whitespace-separated words, and each slot remembers its last prompt, so prompt_n counts only the words
    after the prefix shared with that prompt: the prefix caching the dialogue tests audit. `fail_on=n`
    makes the n-th completion call raise ServerError."""

    def __init__(self, replies=None, fail_on=None):
        self.replies = replies or ["reply"]
        self.fail_on = fail_on
        self.calls = []
        self.tpl = ChatTemplate.from_source(CHATML)
        self._last_prompt = {}

    def health(self):
        return True

    def props(self):
        return {"model_path": "/fake/model.gguf", "total_slots": 4, "build_info": "fake", "model_alias": "fake",
                "default_generation_settings": {}}

    def slots(self):
        return [{"id": i, "is_processing": False} for i in range(self.props().get("total_slots") or 0)]

    def apply_template(self, messages, chat_template_kwargs=None):
        self.template_kwargs = chat_template_kwargs
        return render(self.tpl, messages, **({"enable_thinking": chat_template_kwargs["enable_thinking"]}
                                               if chat_template_kwargs else {}))

    def adds_bos(self):
        return False

    def tokenize(self, text):
        return len(text.split())

    def complete(self, prompt, **kw):
        self.calls.append({"prompt": prompt, **kw})
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise ServerError("fake failure")
        slot = kw["id_slot"]
        prev = self._last_prompt.get(slot)
        common = 0
        if prev:
            while common < min(len(prev), len(prompt)) and prev[common] == prompt[common]:
                common += 1
        prompt_n = len(prompt[common:].split()) if prev else len(prompt.split())
        cached_n = len(prompt[:common].split())     # words in the matched leading prefix: 0 with no prev
        self._last_prompt[slot] = prompt
        reply = self.replies[(len(self.calls) - 1) % len(self.replies)]
        return Completion(text=reply, finish_reason="stop", prompt_n=prompt_n,
                          predicted_n=len(reply.split()),
                          timings={"prompt_n": prompt_n, "predicted_n": len(reply.split())}, raw={},
                          truncated=False, tokens_evaluated=prompt_n, tokens_cached=cached_n)
