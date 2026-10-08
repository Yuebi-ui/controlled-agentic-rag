import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from controlled_rag.evaluation.__main__ import main
from controlled_rag.evaluation.runner import (check_gate, combine_reports, evaluate_suite,
                                              load_jsonl, read_json)
from controlled_rag.evaluation.schemas import RouterRecord


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('suite', ['retrieval','reranker','router','round2','generation',
                                  'citation','safety','benchmark','ablation','badcase','split-audit'])
def test_all_synthetic_fixtures_parse_and_evaluate(suite):
    result=evaluate_suite(suite,ROOT/'eval'/'data'/f'{suite}_synthetic.jsonl')
    assert result['suite']==suite
    assert len(result['provenance']['input_sha256'])==64
    assert result['provenance']['source_type']=='synthetic'
    assert result['metrics']


def test_jsonl_invalid_and_duplicates_include_line_number(tmp_path):
    path=tmp_path/'data.jsonl'
    path.write_text('''{"id":"a","gold_state":"partial","predicted_state":"partial"}\n{"id":"a","gold_state":"failed","predicted_state":"failed"}\n''')
    with pytest.raises(ValueError,match='duplicate sample ID'):
        load_jsonl(path,RouterRecord,'router')
    path.write_text('''{"id":"a","gold_state":"not_a_route","predicted_state":"partial"}\n''')
    with pytest.raises(ValueError,match=':1: invalid'):
        load_jsonl(path,RouterRecord,'router')


def test_cli_roundtrip_output(tmp_path):
    outfile=tmp_path/'metrics.json'
    rc=main(['--suite','retrieval','--input',str(ROOT/'eval/data/retrieval_synthetic.jsonl'),
             '--output',str(outfile)])
    assert rc==0
    assert read_json(outfile)['metrics']['parent_recall_at_10'] >= 0


def test_release_cli_exits_nonzero_when_gate_fails(tmp_path):
    report=tmp_path/'candidate.json'
    rules=tmp_path/'rules.json'
    output=tmp_path/'gate.json'
    report.write_text(json.dumps({'metrics':{'router':{'false_sufficient_rate':.2}}}))
    rules.write_text(json.dumps({'absolute':{'router.false_sufficient_rate':{'max':.05}}}))
    rc=main(['--suite','release','--input',str(report),'--output',str(output),
             '--rules',str(rules)])
    assert rc==1
    assert not read_json(output)['passed']


def test_combine_reports_rejects_duplicate_suite():
    a=evaluate_suite('router',ROOT/'eval/data/router_synthetic.jsonl')
    with pytest.raises(ValueError,match='duplicate suite'):
        combine_reports([a,a])


def test_synthetic_release_gate_refuses_missing_components():
    candidate={'metrics':{'router':{'false_sufficient_rate':0}}}
    rules=read_json(ROOT/'eval/regression/synthetic_control_gate.json')
    assert not check_gate(candidate,rules)['passed']
