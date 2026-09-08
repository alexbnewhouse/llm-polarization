import pytest

CHATML = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)

# Mirrors the part of Olmo-3's template that broke llama.cpp's parser: it serializes `tools`.
OLMO_LIKE = (
    "{%- if tools is not none -%}<functions>{{ tools | tojson }}</functions>{%- endif -%}"
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)

# Mirrors gpt-oss's use of strftime_now in its system preamble.
DATED = (
    "<|start|>system<|message|>Knowledge cutoff: 2024-06\\nCurrent date: {{ strftime_now('%Y-%m-%d') }}<|end|>"
    "{%- for message in messages %}<|start|>{{ message['role'] }}<|message|>{{ message['content'] }}<|end|>{%- endfor %}"
    "{%- if add_generation_prompt %}<|start|>assistant{%- endif %}"
)

REJECTS_TRAILING_SYSTEM = (
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' and not loop.first %}{{ raise_exception('system must be first') }}{%- endif %}"
    "{{- message['role'] + ': ' + message['content'] + '\\n' }}{%- endfor %}"
)


@pytest.fixture
def chatml():
    return CHATML


@pytest.fixture
def olmo_like():
    return OLMO_LIKE


@pytest.fixture
def dated():
    return DATED


@pytest.fixture
def rejects_trailing_system():
    return REJECTS_TRAILING_SYSTEM
