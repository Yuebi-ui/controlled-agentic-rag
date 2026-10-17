"""End-to-end synthetic workflow and real config-toggle ablation invariants."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from controlled_rag.demo import build_demo_pipeline, demo_request, DemoGenerator
from controlled_rag.evaluation.ablation import evaluate_ablation
from controlled_rag.evaluation.ablation_runner import Matrix, run_matrix
from controlled_rag.evaluation.schemas import AblationRecord, WorkflowRecord
from controlled_rag.evaluation.workflow import evaluate_workflow
from controlled_rag.evaluation.workflow_runner import (
    WorkflowCase, load_cases, run_workflow_cases, save_workflow_report,
)
from controlled_rag.orchestrator import PipelineConfig

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / 'eval/data/agent_workflow_synthetic.jsonl'
MATRIX = ROOT / 'eval/data/ablation_matrix_synthetic.json'


def test_workflow_runs_actual_synthetic_pipeline():
    rows, metrics = run_workflow_cases(build_demo_pipeline(), demo_request, load_cases(CASES))
    assert len(rows) == 3
    assert [r.predicted_round1_state for r in rows] == ['sufficient', 'partial', 'failed']
    assert [r.actual_round2 for r in rows] == [False, True, True]
    result = evaluate_workflow(rows)
    assert result['route1']['classification']['accuracy'] == 1.0
    assert result['round2_trigger']['tp'] == 2
    assert result['rescue']['rescued_count'] == 1
    assert result['control_invariants']['all_hold']
    assert metrics['demo_lock_both']['parent_candidate_recall_at_10'] == 1.0
    assert 'parent_candidate_recall_at_10' not in metrics['demo_no_evidence']


def test_one_round_ablation_is_fail_closed():
    gen = DemoGenerator()
    pipeline = build_demo_pipeline(generator=gen)
    pipeline.config = replace(pipeline.config, max_retrieval_rounds=1)
    rows, metrics = run_workflow_cases(pipeline, demo_request, load_cases(CASES))
    assert [r.actual_round2 for r in rows] == [False, False, False]
    assert rows[1].final_status == 'abstained'
    assert rows[1].final_gate_state == 'partial'
    assert rows[1].sealed_final_present is False
    assert rows[1].retrieval_rounds == 1
    assert gen.calls == 1  # only Round1 sufficient case generates
    assert metrics['demo_lock_both']['parent_candidate_recall_at_10'] == 0.5
    assert evaluate_workflow(rows)['control_invariants']['all_hold']


def test_matrix_executes_and_writes_matched_outputs(tmp_path):
    outcome = run_matrix(cases_path=CASES, matrix_path=MATRIX, output=tmp_path,
                         mode='synthetic', seed=99, bootstraps=50)
    diff = outcome['comparison']['variant_comparisons']['without_round2']
    assert diff['paired_sample_count'] == 3
    value = diff['metrics']['answerability_decision_correct']
    assert value['paired_metric_sample_count'] == 3
    assert value['paired_mean_delta'] == pytest.approx(-1 / 3)
    recall = diff['metrics']['parent_candidate_recall_at_10']
    assert recall['paired_metric_sample_count'] == 2
    assert recall['omitted_shared_cases_missing_metric'] == 1
    assert recall['paired_mean_delta'] == pytest.approx(-0.25)
    assert (tmp_path / 'variants/baseline/workflow_cases.jsonl').exists()
    assert (tmp_path / 'variants/without_round2/workflow_report.json').exists()
    assert (tmp_path / 'ablation_cases.jsonl').exists()
    assert json.loads((tmp_path / 'ablation_report.json').read_text())['provenance']['source_type'] == 'synthetic'


def test_ablation_repeated_runs_have_same_numeric_metrics(tmp_path):
    a = run_matrix(cases_path=CASES, matrix_path=MATRIX, output=tmp_path/'one',
                   mode='synthetic', bootstraps=50, seed=91)
    b = run_matrix(cases_path=CASES, matrix_path=MATRIX, output=tmp_path/'two',
                   mode='synthetic', bootstraps=50, seed=91)
    assert a['comparison'] == b['comparison']


def test_sparse_per_metric_pairing_does_not_discard_all_metrics():
    rows = [
        AblationRecord(id='a', variant='baseline', metrics={'x': 1.0, 'y': 0.0}),
        AblationRecord(id='b', variant='baseline', metrics={'x': 0.0}),
        AblationRecord(id='a', variant='without', metrics={'x': 0.0, 'y': 1.0}),
        AblationRecord(id='b', variant='without', metrics={'x': 1.0}),
    ]
    out = evaluate_ablation(rows, bootstraps=0)
    metrics = out['variant_comparisons']['without']['metrics']
    assert metrics['x']['paired_metric_sample_count'] == 2
    assert metrics['y']['paired_metric_sample_count'] == 1
    assert metrics['y']['omitted_shared_cases_missing_metric'] == 1
    assert metrics['y']['paired_mean_delta'] == 1.0
    assert metrics['y']['bootstrap_95pct_ci'] is None


def test_workflow_detects_gate_violations_and_gold_unanswerable():
    record = WorkflowRecord(
        id='bad', gold_answerable=False, gold_round1_state='failed',
        predicted_round1_state='failed', expected_round2=True,
        actual_round2=True, retrieval_rounds=2,
        final_gate_state='failed', final_status='answered',
        reason_code='ANSWERED', sealed_final_present=True,
        mandatory_round1_keys=['p1'], merged_evidence_keys=[], elapsed_ms=1,
    )
    out = evaluate_workflow([record])
    c = out['control_invariants']['violation_counts']
    assert c['answered_without_sufficient_gate'] == 1
    assert c['sealed_without_sufficient_gate'] == 1
    assert c['mandatory_round1_evidence_lost'] == 1
    assert c['answered_gold_unanswerable'] == 1
    assert out['answerability_decision']['accuracy'] == 0.0


def test_workflow_no_labels_for_predicted_route_is_explicit():
    record = WorkflowRecord(
        id='err', gold_answerable=False, gold_round1_state='failed',
        predicted_round1_state=None, expected_round2=True,
        actual_round2=False, retrieval_rounds=0,
        final_gate_state=None, final_status='failed', reason_code='INTEGRITY',
        sealed_final_present=False, elapsed_ms=0,
    )
    out = evaluate_workflow([record])
    assert out['route1']['classification'] is None
    assert out['route1']['missing_predictions'] == 1
    assert out['round2_trigger']['fn'] == 1
    assert out['round2_trigger']['precision'] is None
    assert out['answerability_decision']['accuracy'] == 0.0


def test_duplicate_and_bad_gold_labels_fail(tmp_path):
    lines = CASES.read_text().splitlines()
    file = tmp_path / 'dup.jsonl'
    file.write_text(lines[0] + '\n' + lines[0] + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='duplicate'):
        load_cases(file)
    with pytest.raises(ValidationError):
        WorkflowCase(id='x', query='abc', gold_answerable=True,
                     gold_round1_state='sufficient', expected_round2=True)
    with pytest.raises(ValidationError):
        WorkflowRecord(id='x', gold_answerable=True, gold_round1_state='partial',
                       expected_round2=True, actual_round2=False, retrieval_rounds=2,
                       final_status='failed', reason_code='FAIL', sealed_final_present=False,
                       elapsed_ms=1)


@pytest.mark.parametrize('rounds', [0, 3, -1])
def test_round_limit_validation(rounds):
    with pytest.raises(ValueError, match='max_retrieval_rounds'):
        PipelineConfig(max_retrieval_rounds=rounds)


def test_invalid_matrix_and_real_factory_requirements(tmp_path):
    with pytest.raises(ValueError, match='requires a variant-aware'):
        run_matrix(cases_path=CASES, matrix_path=MATRIX,
                   output=tmp_path, mode='real')
    with pytest.raises(ValueError, match='cannot use a real factory'):
        run_matrix(cases_path=CASES, matrix_path=MATRIX,
                   output=tmp_path, mode='synthetic', factory='x:y')
    with pytest.raises(ValidationError):
        Matrix.model_validate({'baseline': 'a', 'variants': [
            {'name': 'a', 'settings': {'max_retrieval_rounds': 2}},
            {'name': 'a', 'settings': {'max_retrieval_rounds': 1}},
        ]})
    with pytest.raises(ValidationError):
        Matrix.model_validate({'baseline': 'a', 'variants': [
            {'name': 'a', 'settings': {'max_retrieval_rounds': 2}},
            {'name': '../escape', 'settings': {'max_retrieval_rounds': 1}},
        ]})
    with pytest.raises(ValidationError):
        Matrix.model_validate({'baseline': 'a', 'variants': [
            {'name': 'a', 'settings': {'max_retrieval_rounds': 2}},
            {'name': 'b', 'settings': {'max_retrieval_rounds': 2}},
        ]})


def test_saved_workflow_report_stays_metadata_only(tmp_path):
    rows, _ = run_workflow_cases(build_demo_pipeline(), demo_request, load_cases(CASES))
    save_workflow_report(tmp_path, rows, 'synthetic', CASES)
    raw = (tmp_path/'workflow_cases.jsonl').read_text()
    assert '儿童锁' not in raw and '虚构演示手册' not in raw
    parsed = [json.loads(line) for line in raw.splitlines()]
    assert all(not {'query', 'answer', 'answer_blocks', 'text_surrogate'} & set(row)
               for row in parsed)


def test_variant_factory_signature_and_config_application(tmp_path, monkeypatch):
    """Exercise plugin loading without making a real-model performance claim."""
    import sys
    from types import ModuleType

    stub = ModuleType('local_ablation_factory_stub')

    def factory(name, settings):
        pipeline = build_demo_pipeline()
        pipeline.config = replace(pipeline.config,
                                  max_retrieval_rounds=settings['max_retrieval_rounds'])
        return pipeline, demo_request

    stub.build = factory
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    report = run_matrix(cases_path=CASES, matrix_path=MATRIX,
                        output=tmp_path / 'matched', mode='real',
                        factory=stub.__name__ + ':build', bootstraps=0)
    assert report['provenance']['source_type'] == 'real'
    assert report['comparison']['variant_comparisons']['without_round2']['paired_sample_count'] == 3

    def wrong_factory(name, settings):
        return build_demo_pipeline(), demo_request  # wrong when variant wants 1 round

    stub.build = wrong_factory
    with pytest.raises(ValueError, match='did not apply'):
        run_matrix(cases_path=CASES, matrix_path=MATRIX,
                   output=tmp_path / 'mismatched', mode='real',
                   factory=stub.__name__ + ':build', bootstraps=0)
