import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from controlled_answer import AnswerModelSettings, Qwen3DPOAdapter
from controlled_rag.adapters.qwen3_dpo import PipelineQwen3AnswerModel
from controlled_rag.demo import build_demo_pipeline, demo_request
from controlled_rag.evaluation.tracing import LabeledPipelineCase, TraceRecorder, collect_case, save_records
from controlled_rag.contracts import IntegrityError
from controlled_rag.evaluation.runner import evaluate_suite


class FakeOpenAIClient:
    def __init__(self, content=None, error=None):
        self.calls=[]
        self.content=content
        self.error=error
        self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason='stop', message=SimpleNamespace(content=self.content))])


def _model(fake):
    return PipelineQwen3AnswerModel(Qwen3DPOAdapter(AnswerModelSettings(
        variant='dpo', model_name='qwen3-8b-dpo',
        base_url='http://localhost/v1', api_key='EMPTY'), client=fake))


def test_dpo_bridge_runs_pipeline_and_retains_guard():
    content=json.dumps({'abstain':False,'abstain_reason':None,
                        'answer_blocks':[{'text':'虚构说明','evidence_ids':['E1']}]})
    fake=FakeOpenAIClient(content)
    pipeline=build_demo_pipeline(_model(fake))
    outcome=pipeline.run('如何开启儿童锁', demo_request())
    assert outcome.final_status=='answered'
    assert outcome.citations[0].evidence_id=='E1'
    assert len(fake.calls)==1 and fake.calls[0]['model']=='qwen3-8b-dpo'
    assert fake.calls[0]['stream'] is False


def test_fabricated_dpo_citation_rejected_at_adapter():
    fake=FakeOpenAIClient('{"abstain":false,"abstain_reason":null,"answer_blocks":['
                           '{"text":"不合法","evidence_ids":["E99"]}]}')
    response=build_demo_pipeline(_model(fake)).run('如何开启儿童锁',demo_request())
    assert response.final_status=='failed'
    assert response.citations==[]
    assert response.reason_code=='MODEL_OUTPUT_OR_CITATION_INVALID'


def test_dpo_endpoint_unavailable_never_silently_falls_back():
    fake=FakeOpenAIClient(error=ConnectionError('no checkpoint'))
    result=build_demo_pipeline(_model(fake)).run('如何开启儿童锁',demo_request())
    assert result.final_status=='failed'
    assert result.reason_code=='GENERATOR_UNAVAILABLE'
    assert len(fake.calls)==1
    assert fake.calls[0]['model']=='qwen3-8b-dpo'


def test_trace_emits_reranker_candidates_and_round2_lifecycle():
    case=LabeledPipelineCase(id='test',query='如何开启儿童锁，以及儿童锁影响哪些车门？',
                             gold_answerable=True,gold_parent_ids=['parent-lock-on','parent-lock-scope'])
    result,records=collect_case(build_demo_pipeline(),demo_request('test'),case)
    assert result.route1=='partial'
    assert result.route2=='sufficient'
    assert records['round2']['requirement_count']==2
    assert records['round2']['final_covered']==2
    assert records['round2']['merged_evidence_keys']
    assert set(records['reranker_round1']['before'])==set(records['reranker_round1']['after'])
    assert records['citation']['citations'][0]['semantic_supported'] is None


def test_trace_cases_write_metric_compatible_files(tmp_path):
    cases=[LabeledPipelineCase(id='a',query='如何开启儿童锁',
                               gold_answerable=True,gold_parent_ids=['parent-lock-on']),
           LabeledPipelineCase(id='b',query='手册里有没有火箭发动机？',gold_answerable=False)]
    pipeline=build_demo_pipeline()
    traces=[collect_case(pipeline,demo_request(case.id),case)[1] for case in cases]
    counts=save_records(tmp_path,traces)
    assert counts['round2']==2
    assert evaluate_suite('round2',tmp_path/'round2.jsonl')['metrics']['sample_count']==2


def test_trace_after_empty_query_has_result_but_no_fake_retrieval():
    trace=TraceRecorder()
    result=build_demo_pipeline().run('   ',demo_request(),observer=trace)
    assert result.retrieval_rounds==0
    assert list(trace.events)==['result']


def test_nonfinite_reranker_scores_fail_closed():
    pipeline=build_demo_pipeline()
    pipeline.scorer.score_many=lambda query, evidences: [float('nan')]*len(evidences)
    result=pipeline.run('如何开启儿童锁',demo_request())
    assert result.final_status=='failed'
    assert result.reason_code=='INTEGRITY_OR_CONTROL_FAILURE'


def test_permission_denied_prevents_dpo_endpoint_call():
    fake=FakeOpenAIClient('{}')
    pipeline=build_demo_pipeline(_model(fake))
    request=demo_request().model_copy(update={'allowed_kb_ids':frozenset()})
    result=pipeline.run('如何开启儿童锁',request)
    assert result.final_status=='failed'
    assert not fake.calls
