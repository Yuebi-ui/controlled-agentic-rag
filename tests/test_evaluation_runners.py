import asyncio
from pathlib import Path

import httpx
import pytest

from controlled_answer import AnswerModelSettings, create_answer_model, Qwen3BaseAdapter
from controlled_rag.evaluation.generation import evaluate_generation
from controlled_rag.evaluation.runner import evaluate_suite
from controlled_rag.evaluation.safety import evaluate_safety
from controlled_rag.evaluation.schemas import SafetyRecord
from controlled_rag.evaluation.tracing import TraceRecorder


def test_base_variant_selected_explicitly_and_default_remains_dpo():
    base=create_answer_model(variant='base')
    assert isinstance(base,Qwen3BaseAdapter)
    assert base.settings.model_name=='qwen3-8b-base'
    assert create_answer_model().settings.variant=='dpo'


def test_injection_needs_reviewed_attack_outcome():
    with pytest.raises(ValueError):
        SafetyRecord(id='i',attack_type='prompt_injection',expected_status='answered',
                     actual_status='answered',actual_reason_code='ANSWERED')
    bad=SafetyRecord(id='i',attack_type='prompt_injection',expected_status='answered',
                     actual_status='answered',actual_reason_code='ANSWERED',attack_succeeded=True)
    assert evaluate_safety([bad])['failed_case_ids']==['i']


def test_live_generation_collector_unavailable_endpoint_does_not_create_results():
    from controlled_answer import ModelUnavailableError
    from eval.generation.run_live import collect

    class Down:
        def generate(self,request):
            raise ModelUnavailableError('not running')

    cases=[{'id':'a','original_query':'q','atomic_requirements':['q'],
            'final_evidence':[{'evidence_id':'E1','text':'reference'}],
            'gold_answerable':True}]
    with pytest.raises(ModelUnavailableError):
        collect(Down(), cases, 'dpo')


def test_live_generation_collector_no_semantic_ground_truth_assumed():
    from eval.generation.run_live import collect

    class Good:
        def generate(self,request):
            return type('Answer',(),{'abstain':False})()

    cases=[{'id':'a','original_query':'q','atomic_requirements':['q'],
            'final_evidence':[{'evidence_id':'E1','text':'reference'}],
            'gold_answerable':True}]
    pred=collect(Good(),cases,'dpo')
    assert pred[0]['correctness'] is None
    assert pred[0]['faithfulness'] is None
    assert evaluate_generation([__import__('controlled_rag.evaluation.schemas',fromlist=['GenerationRecord']).GenerationRecord.model_validate(pred[0])])['models']['dpo']['faithfulness_rate_annotated'] is None


def test_http_benchmark_uses_one_monotonic_clock_with_mock_transport():
    from scripts.http_bench import benchmark
    def handler(request):
        return httpx.Response(200, json={'final_status':'answered'})
    trace=asyncio.run(benchmark('http://127.0.0.1:9000',5,concurrency=2,
                                transport=httpx.MockTransport(handler)))
    assert len(trace)==5
    assert all(0<=row['start_ms']<=row['end_ms'] for row in trace)
    assert all(row['http_status']==200 for row in trace)
    assert len({row['id'] for row in trace})==5


def test_executable_synthetic_security_suite_behaves_as_expected():
    from eval.safety.run_control_probes import run_probes
    from controlled_rag.evaluation.schemas import SafetyRecord
    rows=[SafetyRecord.model_validate(x) for x in run_probes()]
    result=evaluate_safety(rows)
    assert result['safety_pass_rate']==1
    assert set(result['by_attack']) >= {'unauthorized_kb','stale_version','fake_citation','no_answer'}
    assert 'prompt_injection' not in result['by_attack']
