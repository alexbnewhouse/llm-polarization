import json
from pathlib import Path
import pytest
from harness import log, survey
from harness.dialogue import AgentHandle, GenSettings, DyadSpec
from harness.survey import SurveyRunner, SurveyError
from harness.templates import ChatTemplate
from harness.transcript import Transcript, SEEKER, MENTOR
from harness.tests.fakes import FakeClient
from harness.tests.conftest import CHATML

REPO = Path(__file__).resolve().parents[2]


def items():
    return survey.load_batteries(REPO / "instruments" / "batteries.json")


def test_batteries_file_loads_13_unique_items():
    it = items()
    assert len(it) == 13 and len({i["id"] for i in it}) == 13
    assert {i["battery"] for i in it} == {"ideological", "thermometer", "agreement"}


def test_batteries_file_declares_a_version_and_whether_it_is_adapted():
    # The wording is a placeholder until the US-adaptation task lands; a run that cannot say which
    # wording it administered cannot be interpreted afterwards.
    data = json.loads((REPO / "instruments" / "batteries.json").read_text(encoding="utf-8"))
    assert isinstance(data["version"], str) and data["version"]
    assert data["adapted"] is False


def test_load_batteries_rejects_bad_item(tmp_path):
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"items": [{"id": "x", "battery": "b", "text": "t"}]}))
    with pytest.raises(ValueError):
        survey.load_batteries(p)


def test_answer_schema_and_parse():
    item = {"id": "i", "battery": "b", "text": "t", "scale": {"min": 1, "max": 5}}
    s = survey.answer_schema(item)
    assert s["properties"]["answer"] == {"type": "integer", "minimum": 1, "maximum": 5} and s["required"] == ["answer"]
    assert survey.parse_answer('{"answer": 4}', item) == 4
    assert survey.parse_answer('{"answer": 9}', item) is None
    assert survey.parse_answer('four', item) is None
    assert survey.parse_answer('{"answer": "3"}', item) is None


def make(tmp_path, replies=None, fail_on=None):
    tpl = ChatTemplate.from_source(CHATML)
    mc = FakeClient(replies or ['{"answer": 3}'], fail_on=fail_on)
    mentor = AgentHandle(MENTOR, mc, tpl, "mentorhash", slot=1)
    w = log.JsonlWriter(tmp_path / "surveys.jsonl")
    return SurveyRunner("run1", 99, mentor, w, GenSettings(), clock=lambda: "T",
                        batteries_sha256="BATT"), mc


def spec():
    return DyadSpec("d1", {"topic": "t"}, "You are Dana.", "rem", "reinforced", 5, 2)


def test_pre_survey_fresh_context_per_item(tmp_path):
    runner, mc = make(tmp_path)
    rows = runner.administer(spec(), 1, "pre", None, items()[:3])
    assert len(rows) == 3 and len(mc.calls) == 3
    for c, it in zip(mc.calls, items()[:3]):
        assert c["prompt"] == f"<|im_start|>user\n{it['text']}<|im_end|>\n<|im_start|>assistant\n"
        assert c["json_schema"] == survey.answer_schema(it) and c["n_predict"] == 32 and c["temperature"] == 0.0
        assert c["id_slot"] == 1 and c["cache_prompt"] is True
    assert [r["answer"] for r in rows] == [3, 3, 3] and all(r["phase"] == "pre" for r in rows)


def test_post_survey_branches_off_mentor_view(tmp_path):
    runner, mc = make(tmp_path)
    t = Transcript("d1", "You are Dana.", "rem", "reinforced")
    t.append(1, SEEKER, "S1"); t.append(1, MENTOR, "M1")
    runner.administer(spec(), 1, "post", t, items()[:2])
    for c in mc.calls:
        assert c["prompt"].startswith("<|im_start|>user\nS1<|im_end|>\n<|im_start|>assistant\nM1<|im_end|>\n<|im_start|>user\n")
        assert "<|im_start|>system" not in c["prompt"]
    assert mc.calls[0]["prompt"] != mc.calls[1]["prompt"]


def test_post_requires_transcript(tmp_path):
    runner, _ = make(tmp_path)
    with pytest.raises(ValueError):
        runner.administer(spec(), 1, "post", None, items()[:1])
    with pytest.raises(ValueError):
        runner.administer(spec(), 1, "mid", None, items()[:1])


def test_rows_and_seeds(tmp_path):
    runner, mc = make(tmp_path, replies=['{"answer": 2}', 'nonsense'])
    rows = runner.administer(spec(), 1, "pre", None, items()[:2])
    logged = log.read_jsonl(tmp_path / "surveys.jsonl")
    assert logged == rows
    assert rows[0]["answer"] == 2 and rows[1]["answer"] is None and rows[1]["raw_text"] == "nonsense"
    assert rows[0]["seed"] == log.derive_seed(99, 5, "d1", 1, 0, f"survey:pre:{items()[0]['id']}")
    assert rows[0]["seed"] != rows[1]["seed"]
    assert set(rows[0]) >= {"run_id", "dyad_id", "attempt", "phase", "item_id", "battery", "scale", "answer",
                            "raw_text", "prompt_sha256", "prompt_n", "seed", "ts"}
    post = runner.administer(spec(), 1, "post", Transcript("d1", "s", None, "once"), items()[:1])
    assert post[0]["seed"] == log.derive_seed(99, 5, "d1", 1, 3, f"survey:post:{items()[0]['id']}")


def test_rows_carry_model_instrument_and_slot_identity(tmp_path):
    runner, mc = make(tmp_path)
    rows = runner.administer(spec(), 1, "pre", None, items()[:1])
    r = rows[0]
    assert r["model_sha256"] == "mentorhash" and r["template_sha256"] == ChatTemplate.from_source(CHATML).sha256
    assert r["batteries_sha256"] == "BATT" and r["id_slot"] == 1 and r["turn"] == 0
    assert r["temperature"] == 0.0 and r["n_predict"] == 32 and r["prompt_chars"] == len(mc.calls[0]["prompt"])
    assert r["origin"] == "run"


def test_origin_marks_a_readministered_pass(tmp_path):
    runner, _ = make(tmp_path)
    rows = runner.administer(spec(), 1, "pre", None, items()[:1], origin="readministered")
    assert rows[0]["origin"] == "readministered"
    # The key (run_id, dyad_id, attempt, phase, item_id) is identical to the in-run row; origin is what
    # tells the two passes apart in analysis.
    assert rows[0]["seed"] == log.derive_seed(99, 5, "d1", 1, 0, f"survey:pre:{items()[0]['id']}")
    with pytest.raises(ValueError):
        runner.administer(spec(), 1, "pre", None, items()[:1], origin="again")


def test_server_error_logs_and_raises(tmp_path):
    runner, mc = make(tmp_path, fail_on=2)
    with pytest.raises(SurveyError) as ei:
        runner.administer(spec(), 1, "pre", None, items()[:3])
    assert ei.value.item_id == items()[1]["id"] and ei.value.phase == "pre"
    rows = log.read_jsonl(tmp_path / "surveys.jsonl")
    assert len(rows) == 2 and rows[1]["answer"] is None and "fake failure" in rows[1]["error"]
