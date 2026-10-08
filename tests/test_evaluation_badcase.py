import pytest
from controlled_rag.evaluation.badcase import evaluate_badcases
from controlled_rag.evaluation.schemas import BadCaseRecord


def _row(**overrides):
    d={'id':'test','category':'fake_citation','severity':'critical',
       'expected_status':'failed','actual_status':'failed',
       'actual_reason_code':'MODEL_OUTPUT_OR_CITATION_INVALID',
       'actual_retrieval_rounds':1,'forbidden_citation_ids':['E99']}
    d.update(overrides)
    return BadCaseRecord(**d)


def test_badcase_pass_and_failure():
    report=evaluate_badcases([_row(), _row(id='bad', actual_status='answered',
                              observed_citation_ids=['E99'], actual_retrieval_rounds=3)])
    assert report['critical_failed_count']==1
    assert report['sample_count']==2
    assert report['pass_rate']==.5
    assert set(report['failed_cases'][0]['violations'])=={'wrong_final_status','too_many_retrieval_rounds','forbidden_citation'}


def test_badcase_empty_fails():
    with pytest.raises(ValueError):
        evaluate_badcases([])
