"""No model weights, GPU or network required. Use a fake vLLM-style client."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from controlled_answer import (
    AnswerModelSettings,
    EvidenceItem,
    GenerationContractError,
    GenerationRequest,
    ModelUnavailableError,
    Qwen3DPOAdapter,
    StructuredAnswer,
    create_answer_model,
)


class FakeClient:
    def __init__(self, content: str | None = None, *, error=None, finish_reason="stop"):
        self.called_with = None
        self.content = content
        self.error = error
        self.finish_reason = finish_reason
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.called_with = kwargs
        if self.error is not None:
            raise self.error
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason=self.finish_reason,
            message=SimpleNamespace(content=self.content),
        )])


def make_request():
    return GenerationRequest(
        original_query="如何启用功能？",
        atomic_requirements=("说明步骤",),
        final_evidence=(EvidenceItem("E1", "菜单中可以开启功能。"),),
    )


def success_json():
    return json.dumps({
        "abstain": False, "abstain_reason": None,
        "answer_blocks": [{"heading": None, "text": "在菜单中开启。", "evidence_ids": ["E1"]}],
    }, ensure_ascii=False)


def make_adapter(fake, mode="structured_outputs"):
    return Qwen3DPOAdapter(AnswerModelSettings(
        variant="dpo", model_name="qwen3-8b-dpo",
        base_url="http://127.0.0.1:8000/v1", api_key="EMPTY", output_mode=mode,
    ), client=fake)


def test_default_selection_is_dpo_not_sft(monkeypatch):
    monkeypatch.delenv("ANSWER_MODEL_VARIANT", raising=False)
    monkeypatch.delenv("QWEN3_DPO_MODEL_NAME", raising=False)
    model = create_answer_model(client=FakeClient(success_json()))
    assert isinstance(model, Qwen3DPOAdapter)
    assert model.settings.model_name == "qwen3-8b-dpo"


def test_sft_is_manual_baseline_only(monkeypatch):
    monkeypatch.setenv("QWEN3_SFT_MODEL_NAME", "qwen3-8b-sft-checkpoint")
    model = create_answer_model(variant="sft", client=FakeClient(success_json()))
    assert model.settings.variant == "sft"
    assert model.settings.model_name == "qwen3-8b-sft-checkpoint"


def test_config_rejects_unknown_variant():
    with pytest.raises(ValueError, match="ANSWER_MODEL_VARIANT"):
        AnswerModelSettings.from_env({"ANSWER_MODEL_VARIANT": "orpo"})


def test_explicit_dpo_rejects_sft_configuration():
    cfg = AnswerModelSettings("sft", "sft", "http://localhost/v1", "EMPTY")
    with pytest.raises(ValueError, match="DPO"):
        Qwen3DPOAdapter(cfg)


def test_good_answer_uses_sealed_evidence_only():
    fake = FakeClient(success_json())
    answer = make_adapter(fake).generate(make_request())
    assert isinstance(answer, StructuredAnswer)
    assert answer.answer_blocks[0].evidence_ids == ["E1"]
    assert fake.called_with["model"] == "qwen3-8b-dpo"
    assert fake.called_with["stream"] is False
    assert "structured_outputs" in fake.called_with["extra_body"]
    payload = json.loads(fake.called_with["messages"][1]["content"])
    assert payload["allowed_evidence_ids"] == ["E1"]
    assert payload["final_evidence"] == [{"evidence_id": "E1", "text": "菜单中可以开启功能。"}]


def test_legacy_guided_json_mode():
    fake = FakeClient(success_json())
    make_adapter(fake, mode="guided_json").generate(make_request())
    assert "guided_json" in fake.called_with["extra_body"]
    assert "structured_outputs" not in fake.called_with["extra_body"]


def test_valid_abstention():
    fake = FakeClient('{"abstain":true,"abstain_reason":"证据不足","answer_blocks":[]}')
    answer = make_adapter(fake).generate(make_request())
    assert answer.abstain is True


@pytest.mark.parametrize("content", [
    "not json", "",
    '{"abstain":true,"abstain_reason":"缺资料","answer_blocks":[{"text":"不应回答","evidence_ids":["E1"]}]}',
    '{"abstain":false,"abstain_reason":null,"answer_blocks":[]}',
    '{"abstain":false,"abstain_reason":null,"answer_blocks":[{"text":"答","evidence_ids":["E1","E1"]}]}',
    '{"abstain":false,"abstain_reason":null,"answer_blocks":[{"text":"答","evidence_ids":["E0"]}]}',
    '{"abstain":false,"abstain_reason":null,"answer_blocks":[{"text":"答","evidence_ids":["E1"],"fake_page":9}]}',
    '{"abstain":"false","abstain_reason":null,"answer_blocks":[{"text":"答","evidence_ids":["E1"]}]}',
    None,
])
def test_invalid_answers_fail_closed(content):
    with pytest.raises(GenerationContractError):
        make_adapter(FakeClient(content)).generate(make_request())


def test_fabricated_but_well_formed_citation_fails():
    fake = FakeClient('{"abstain":false,"abstain_reason":null,"answer_blocks":[{"text":"答","evidence_ids":["E99"]}]}')
    with pytest.raises(GenerationContractError, match="outside final_evidence"):
        make_adapter(fake).generate(make_request())


def test_truncation_fails_closed():
    with pytest.raises(GenerationContractError):
        make_adapter(FakeClient(success_json(), finish_reason="length")).generate(make_request())


def test_unavailable_dpo_never_falls_back_to_sft():
    fake = FakeClient(error=ConnectionError("connection refused"))
    with pytest.raises(ModelUnavailableError, match="no automatic SFT fallback"):
        make_adapter(fake).generate(make_request())
    assert fake.called_with["model"] == "qwen3-8b-dpo"


@pytest.mark.parametrize("ev", [(), (EvidenceItem("E1", "a"), EvidenceItem("E1", "b"))])
def test_empty_or_duplicate_final_evidence_rejected_before_call(ev):
    with pytest.raises(ValueError):
        GenerationRequest("q", (), ev)


def test_invalid_input_evidence_id():
    with pytest.raises(ValueError, match="Evidence ID"):
        EvidenceItem("../../etc/passwd", "content")


def test_response_extra_fields_rejected():
    with pytest.raises(ValidationError):
        StructuredAnswer.model_validate_json('{"abstain":true,"abstain_reason":"no","answer_blocks":[],"extra":1}')


def test_final_evidence_tuple_immutable():
    request = GenerationRequest("q", [], [EvidenceItem("E1", "ok")])
    assert isinstance(request.final_evidence, tuple)
    assert isinstance(request.atomic_requirements, tuple)


def test_no_automatic_sft_name_on_dpo_variant():
    settings = AnswerModelSettings.from_env({
        "ANSWER_MODEL_VARIANT": "dpo",
        "QWEN3_SFT_MODEL_NAME": "my-sft",
    })
    assert settings.model_name == "qwen3-8b-dpo"
