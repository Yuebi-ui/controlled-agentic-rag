import math

import pytest
from pydantic import ValidationError

from controlled_rag.evaluation.ablation import evaluate_ablation
from controlled_rag.evaluation.benchmark import evaluate_benchmark
from controlled_rag.evaluation.citation import evaluate_citation
from controlled_rag.evaluation.dataset import audit_splits
from controlled_rag.evaluation.generation import evaluate_generation
from controlled_rag.evaluation.metrics import (at_k, confusion_stats, expected_calibration_error,
                                               multiclass_brier, nearest_rank)
from controlled_rag.evaluation.regression import evaluate_release_gate
from controlled_rag.evaluation.retrieval import evaluate_reranker, evaluate_retrieval
from controlled_rag.evaluation.round2 import evaluate_round2
from controlled_rag.evaluation.router import evaluate_router
from controlled_rag.evaluation.safety import evaluate_safety
from controlled_rag.evaluation.schemas import (AblationRecord, BenchmarkRecord, CitationRecord,
                                                GenerationRecord, RetrievalRecord, RerankerRecord,
                                                Round2Record, RouterRecord, SafetyRecord)


def test_at_k_known_result_and_empty_ranking():
    found = at_k(['noise', 'gold'], {'gold'}, 2)
    assert found['recall'] == 1
    assert found['hit'] == 1
    assert found['mrr'] == 0.5
    assert found['ndcg'] == pytest.approx(1/math.log2(3))
    assert at_k([], {'gold'}, 10) == {'recall': 0, 'hit': 0, 'mrr': 0, 'ndcg': 0}


@pytest.mark.parametrize('ranked', [[], ['a', 'a', 'b'], ['z']])
def test_recall_bounded_and_duplicates_do_not_score_twice(ranked):
    outcome = at_k(ranked, {'a', 'b'}, 10)
    assert 0 <= outcome['recall'] <= 1
    assert 0 <= outcome['ndcg'] <= 1


@pytest.mark.parametrize('k,gold', [(0, {'a'}), (-2, {'a'}), (10, set())])
def test_invalid_at_k_rejected(k, gold):
    with pytest.raises(ValueError):
        at_k(['a'], gold, k)


def test_retrieval_parent_vs_image_denominators():
    records = [RetrievalRecord(id='1', gold_parent_ids=['p1','p2'], ranked_parent_ids=['p1'],
                               gold_image_ids=['i1'], ranked_image_ids=['nope', 'i1']),
               RetrievalRecord(id='2', gold_parent_ids=['p3'], ranked_parent_ids=[])]
    result = evaluate_retrieval(records)
    assert result['parent_recall_at_10'] == pytest.approx(.25)
    assert result['image_recall_at_1'] == 0
    assert result['image_recall_at_5'] == 1
    assert result['image_labeled_count'] == 1


def test_retrieval_without_images_is_unmeasurable_not_zero():
    val = evaluate_retrieval([RetrievalRecord(id='1', gold_parent_ids=['p'])])
    assert val['image_recall_at_10'] is None
    assert val['image_labeled_count'] == 0


def test_retrieval_requires_nonempty_targets_and_unique_ids():
    with pytest.raises(ValidationError):
        RetrievalRecord(id='r')
    with pytest.raises(ValidationError):
        RetrievalRecord(id='r', gold_parent_ids=['p'], ranked_parent_ids=['p', 'p'])


def test_reranker_improvements_and_pairwise_tie_policy():
    samples = [RerankerRecord(id='r', positive_ids=['p'], before=['x','p'], after=['p','x'],
                              after_scores={'p': .8, 'x': .8})]
    out = evaluate_reranker(samples)
    assert out['delta_mrr_at_5'] == .5
    assert out['pair_count'] == 1
    assert out['pairwise_accuracy'] == .5


def test_reranker_no_pair_scored_uses_null():
    val = evaluate_reranker([RerankerRecord(id='a', positive_ids=['p'], before=['p'], after=['p'])])
    assert val['pair_count'] == 0 and val['pairwise_accuracy'] is None


def test_reranker_reject_candidate_set_change_and_missing_scores():
    with pytest.raises(ValidationError):
        RerankerRecord(id='a', positive_ids=['p'], before=['a'], after=['b'])
    with pytest.raises(ValidationError):
        RerankerRecord(id='a', positive_ids=['p'], before=['a'], after=['a'], after_scores={'other':1})


def test_router_macro_f1_and_false_sufficient():
    rows = [RouterRecord(id='a', gold_state='sufficient', predicted_state='sufficient'),
            RouterRecord(id='b', gold_state='partial', predicted_state='sufficient'),
            RouterRecord(id='c', gold_state='failed', predicted_state='failed')]
    val = evaluate_router(rows)
    assert val['false_sufficient_rate'] == .5
    assert val['false_sufficient_count'] == 1
    assert val['confusion']['partial']['sufficient'] == 1
    assert val['macro_f1'] < 1
    assert val['ece_toplabel_10bin'] is None
    assert val['brier_multiclass_unscaled'] is None


def test_router_calibration_reference_case():
    prob = (.8, .1, .1)
    assert multiclass_brier([prob], ['sufficient']) == pytest.approx(.06)
    assert expected_calibration_error([prob], ['sufficient']) == pytest.approx(.2)
    row = RouterRecord(id='rt', gold_state='sufficient', predicted_state='sufficient',
                       probabilities={'sufficient':.8, 'partial':.1, 'failed':.1})
    assert evaluate_router([row])['brier_multiclass_unscaled'] == pytest.approx(.06)


@pytest.mark.parametrize('prob', [
    {'sufficient':.6, 'partial':.3, 'failed':.2},
    {'sufficient':.6, 'partial':-.1, 'failed':.5},
    {'sufficient':.9, 'partial':.1},
])
def test_invalid_router_calibration_rejected(prob):
    with pytest.raises(ValidationError):
        RouterRecord(id='bad', gold_state='failed', predicted_state='failed', probabilities=prob)


def test_router_raw_vs_guarded_false_sufficient_distinguished():
    rows = [RouterRecord(id='a', gold_state='partial', predicted_state='partial', raw_model_state='sufficient'),
            RouterRecord(id='b', gold_state='failed', predicted_state='failed', raw_model_state='failed')]
    out = evaluate_router(rows)
    assert out['false_sufficient_rate'] == 0
    assert out['raw_false_sufficient_rate'] == .5


def _round2(**kw):
    options = {'id':'r', 'round1_state':'partial', 'round2_state':'sufficient', 'retrieval_rounds':2,
               'final_status':'answered', 'gold_answerable':True, 'requirement_count':2,
               'round1_covered':1, 'final_covered':2,
               'mandatory_round1_keys':['p1'], 'merged_evidence_keys':['p1','p2'],
               'final_evidence_keys':['p2']}
    options.update(kw)
    return Round2Record(**options)


def test_round2_rescue_uses_answerable_triggered_denominator_and_pool_preservation():
    out = evaluate_round2([_round2(), _round2(id='fail', gold_answerable=False,
          round1_state='failed', round2_state='failed', final_status='abstained',
          mandatory_round1_keys=[], round1_covered=0, final_covered=0)])
    assert out['rescue_rate_answerable_triggered'] == 1
    assert out['rescue_rate_all_triggered'] == .5
    assert out['mean_coverage_gain_triggered'] == .25
    # Mandatory evidence is retained in MERGED pool; top-K can exclude it.
    assert out['violations']['mandatory_round1_evidence_lost'] == 0


def test_round2_unsafe_answer_violation():
    out = evaluate_round2([_round2(round2_state='failed', final_status='answered',
                                      gold_answerable=False, merged_evidence_keys=[] ,final_evidence_keys=[])])
    assert out['violations']['unsafe_answer_without_sufficient_final_gate'] == 1
    assert out['violations']['mandatory_round1_evidence_lost'] == 1
    assert out['violations']['answered_unanswerable_gold'] == 1
    assert not out['all_control_invariants_hold']


def test_round2_no_eligible_rescues_returns_null():
    out = evaluate_round2([_round2(id='n', gold_answerable=False, final_status='abstained',
                                    round2_state='failed')])
    assert out['rescue_rate_answerable_triggered'] is None


@pytest.mark.parametrize('override', [
    {'retrieval_rounds':1, 'round2_state':'partial'},
    {'requirement_count':1, 'round1_covered':2},
    {'round1_state':None},
])
def test_round2_invalid_state_records(override):
    with pytest.raises(ValidationError):
        _round2(**override)


def test_generation_missing_annotations_never_imputed():
    result = evaluate_generation([GenerationRecord(id='g',model_variant='dpo', gold_answerable=True,
                                                   schema_valid=True,predicted_abstain=False)])
    model = result['models']['dpo']
    assert model['correctness_rate_annotated'] is None
    assert model['faithfulness_rate_annotated'] is None
    assert model['json_validity_rate'] == 1
    assert model['abstention_f1'] is None


def test_generation_abstention_confusion_and_coverage():
    rows = [GenerationRecord(id='a',model_variant='sft', gold_answerable=False,
                             schema_valid=True,predicted_abstain=True),
            GenerationRecord(id='b',model_variant='sft', gold_answerable=True,
                             schema_valid=True,predicted_abstain=True),
            GenerationRecord(id='c',model_variant='sft', gold_answerable=False,
                             schema_valid=True,predicted_abstain=False,
                             correctness=False, faithfulness=False)]
    v = evaluate_generation(rows)['models']['sft']
    assert v['abstention_precision'] == .5
    assert v['abstention_recall'] == .5
    assert v['abstention_f1'] == .5
    assert v['unsafe_answers_on_unanswerable'] == 1


def test_generation_duplicate_variant_id_fails():
    x = GenerationRecord(id='z',model_variant='dpo', gold_answerable=True,
                         schema_valid=True,predicted_abstain=False)
    with pytest.raises(ValueError, match='duplicate'):
        evaluate_generation([x,x])


def test_generation_bad_schema_state_rejected():
    with pytest.raises(ValidationError):
        GenerationRecord(id='x', model_variant='dpo', gold_answerable=True, schema_valid=False,
                         predicted_abstain=True)


def _citation(**extra):
    data = {'id':'q', 'allowed_evidence_ids':['E1'],
            'canonical_evidence':{'E1':{'document_id':'doc','source_uri':'demo://x','source_pages':[1]}},
            'citations':[{'evidence_id':'E1','document_id':'doc','source_uri':'demo://x','source_pages':[1]}]}
    data.update(extra)
    return CitationRecord.model_validate(data)


def test_citation_id_validity_separate_from_semantic_support():
    value = evaluate_citation([_citation()])
    assert value['citation_validity_rate'] == 1
    assert value['semantic_support_rate_annotated'] is None
    annotated = _citation(citations=[{'evidence_id':'E1','document_id':'doc',
                                      'source_uri':'demo://x','source_pages':[1], 'semantic_supported':False}])
    value = evaluate_citation([annotated])
    assert value['citation_validity_rate'] == 1
    assert value['semantic_support_rate_annotated'] == 0


def test_citation_forged_id_or_metadata_does_not_count_valid():
    value = evaluate_citation([_citation(citations=[{'evidence_id':'E99','document_id':'doc',
                   'source_uri':'demo://x','source_pages':[1]},
                  {'evidence_id':'E1','document_id':'doc','source_uri':'demo://x', 'source_pages':[999]}])])
    assert value['valid_citation_count'] == 0
    assert value['citation_validity_rate'] == 0


def test_citation_no_citations_is_null_not_perfect():
    result = evaluate_citation([_citation(citations=[])])
    assert result['citation_validity_rate'] is None


def test_safety_failure_reasons_and_generator_leak():
    ok = SafetyRecord(id='a',attack_type='auth',expected_status='failed',
                      actual_status='failed',expected_reason_codes=['INTEGRITY_OR_CONTROL_FAILURE'],
                      actual_reason_code='INTEGRITY_OR_CONTROL_FAILURE',must_not_generate=True)
    bad = ok.model_copy(update={'id':'b', 'generation_called':True})
    out = evaluate_safety([ok,bad])
    assert out['safety_pass_rate'] == .5
    assert out['failed_case_ids'] == ['b']


def test_nearest_rank_edge_cases():
    assert nearest_rank([100, 200, 300, 400], .95) == 400
    assert nearest_rank([100, 200], .50) == 100
    assert nearest_rank([], .95) is None
    with pytest.raises(ValueError):
        nearest_rank([1], 1.1)


def test_benchmark_with_non2xx_and_gpu_samples():
    rows = [BenchmarkRecord(id='a', scenario='C1', concurrency=2, start_ms=0.0,
                            end_ms=100.,http_status=200),
            BenchmarkRecord(id='b', scenario='C1', concurrency=2, start_ms=0.0,
                            end_ms=200.,http_status=503,gpu_memory_mb=888.0)]
    value = evaluate_benchmark(rows)['overall']
    assert value['p95_latency_ms'] == 200
    assert value['success_rate'] == .5
    assert value['throughput_requests_per_second'] == 10
    assert value['gpu_peak_mb_observed'] == 888.0


def test_benchmark_zero_window_does_not_divide_by_zero():
    row = BenchmarkRecord(id='r',scenario='C1',concurrency=1,start_ms=1.0,end_ms=1.0,http_status=200)
    assert evaluate_benchmark([row])['overall']['throughput_requests_per_second'] is None


def test_ablation_paired_only_deterministic_seed():
    rows = [AblationRecord(id='x',variant='baseline',metrics={'m':.5}),
            AblationRecord(id='y',variant='baseline',metrics={'m':.1}),
            AblationRecord(id='x',variant='new',metrics={'m':.9}),
            AblationRecord(id='z',variant='new',metrics={'m':.8})]
    value = evaluate_ablation(rows,bootstraps=100,seed=9)
    comp = value['variant_comparisons']['new']
    assert comp['paired_sample_count'] == 1
    assert comp['dropped_unpaired_variant_cases'] == 1
    assert comp['metrics']['m']['paired_mean_delta'] == .4
    assert value == evaluate_ablation(rows,bootstraps=100,seed=9)


def test_ablation_unpaired_fails_closed():
    rows = [AblationRecord(id='x',variant='baseline',metrics={'m':1.}),
            AblationRecord(id='y',variant='new',metrics={'m':.5})]
    with pytest.raises(ValueError):
        evaluate_ablation(rows)


def test_release_gate_missing_metric_rejected_and_delta_checked():
    candidate={'a':{'score':.9}}
    rules={'absolute':{'a.score':{'min':.8},'no.metric':{'min':0}},
           'delta':{'a.score':{'min':-.05}}}
    gate=evaluate_release_gate(candidate, rules, baseline={'a':{'score':1.0}})
    assert not gate['passed'] and gate['check_count'] == 3
    assert len(gate['failed_checks']) == 2
    assert gate['failed_checks'][0]['reason'] == 'missing_or_invalid_metric'


def test_release_gate_absolute_and_relative_success():
    gate=evaluate_release_gate({'m':.8}, {'absolute':{'m':{'min':.75}},
                                              'delta':{'m':{'min':-.05}}}, {'m':.82})
    assert gate['passed']


def test_release_gate_no_baseline_for_delta_is_error():
    with pytest.raises(ValueError):
        evaluate_release_gate({'m':.8}, {'delta':{'m':{'max':0}}})


def test_audit_document_and_hash_leak_detection():
    rows=[{'id':'a','split':'train','document_id':'d1','text_hash':'h'},
          {'id':'b','split':'test','document_id':'d1','text_hash':'h'}]
    report=audit_splits(rows)
    assert not report['passed']
    assert report['cross_split_document_leaks'] == ['d1']
    assert report['cross_split_exact_hash_leaks'] == ['h']


def test_empty_metrics_error():
    for evaluator in (evaluate_retrieval,evaluate_router,evaluate_citation,evaluate_benchmark,
                      evaluate_generation,evaluate_safety,evaluate_round2):
        with pytest.raises(ValueError):
            evaluator([])


def test_release_gate_invalid_threshold_type_rejected():
    with pytest.raises(ValueError, match='invalid min'):
        evaluate_release_gate({'m':1.0}, {'absolute':{'m':{'min':'1'}}})


def test_explicit_empty_merged_pool_flags_retention_lost():
    result=evaluate_round2([_round2(merged_evidence_keys=[],final_evidence_keys=['p1'])])
    assert result['violations']['mandatory_round1_evidence_lost']==1


def test_dataset_split_duplicate_id_and_invalid_row_are_reported_not_crash():
    data=[{'id':'a','split':'train','document_id':'d1'},
          {'id':'a','split':'train','document_id':'d1'},
          {'id':{'not':'a-string'},'split':'test','document_id':'d2'}]
    report=audit_splits(data)
    assert report['duplicate_ids_within_split']==['a']
    assert report['invalid_rows']
    assert not report['passed']
