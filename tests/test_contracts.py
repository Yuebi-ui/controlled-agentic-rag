import json

import pytest
from pydantic import ValidationError

from controlled_rag.contracts import AnswerBlock, Evidence, StructuredAnswer


def test_valid_answer():
    good = StructuredAnswer(abstain=False, abstain_reason=None,
                            answer_blocks=[AnswerBlock(text="有证据支持", evidence_ids=["E1"])])
    assert good.abstain is False


@pytest.mark.parametrize("payload", [
    {"abstain": True, "abstain_reason": "不足", "answer_blocks": [{"text": "仍然回答", "evidence_ids": ["E1"]}]},
    {"abstain": True, "abstain_reason": None, "answer_blocks": []},
    {"abstain": False, "abstain_reason": None, "answer_blocks": []},
    {"abstain": False, "abstain_reason": None, "answer_blocks": [{"text": "x", "evidence_ids": ["E1", "E1"]}]},
    {"abstain": False, "abstain_reason": None, "answer_blocks": [{"text": "x", "evidence_ids": ["E1"], "injected": True}]},
])
def test_invalid_answer_contract(payload):
    with pytest.raises(ValidationError):
        StructuredAnswer.model_validate_json(json.dumps(payload, ensure_ascii=False))


def test_missing_metadata_and_page_rejected():
    with pytest.raises(ValidationError):
        Evidence.model_validate({"evidence_key": "p1"})


def test_abstaining_with_no_blocks():
    result = StructuredAnswer.model_validate_json(
        '{"abstain":true,"abstain_reason":"无法确认","answer_blocks":[]}')
    assert result.abstain
