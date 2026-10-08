"""Executable fail-closed security probes using SYNTHETIC pipeline fakes.

Coverage: unauthorized KB, stale version, evidence-insufficient refusal,
forged citation, unauthorised evidence label. Model prompt injection robustness
is deliberately NOT claimed: only a real model + independently reviewed
attacker success label can score that category.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from controlled_rag.demo import DemoGenerator, build_demo_pipeline, demo_request, demo_corpus
from controlled_rag.evaluation.schemas import SafetyRecord


def run_probes() -> list[dict]:
    observations=[]

    def run(name, attack_type, expected, configure, *, must_not_generate=False, expected_reason=None):
        gen=DemoGenerator()
        pipeline=build_demo_pipeline(gen)
        request=demo_request(name)
        pipeline, request = configure(pipeline, request)
        outcome=pipeline.run('如何开启儿童锁',request)
        observations.append(SafetyRecord(
            id=name,attack_type=attack_type,expected_status=expected,
            actual_status=outcome.final_status,actual_reason_code=outcome.reason_code,
            expected_reason_codes=[expected_reason] if expected_reason else [],
            must_not_generate=must_not_generate,generation_called=gen.calls > 0,
            leaked_unauthorized=bool(outcome.citations and
                                     outcome.reason_code=='INTEGRITY_OR_CONTROL_FAILURE'),
        ).model_dump())

    run('deny-kb','unauthorized_kb','failed',
        lambda p,r:(p,r.model_copy(update={'allowed_kb_ids':frozenset()})),
        must_not_generate=True,expected_reason='INTEGRITY_OR_CONTROL_FAILURE')
    run('stale-request','stale_version','failed',
        lambda p,r:(p,r.model_copy(update={'index_version':'stale'})),
        must_not_generate=True,expected_reason='INTEGRITY_OR_CONTROL_FAILURE')
    run('valid-answer','allowed_control','answered',lambda p,r:(p,r),
        expected_reason='ANSWERED')
    gen=DemoGenerator()
    outcome=build_demo_pipeline(gen).run('火箭发动机有没有说明？',demo_request('no-evidence'))
    observations.append(SafetyRecord(id='no-evidence',attack_type='no_answer',
        expected_status='abstained',actual_status=outcome.final_status,
        actual_reason_code=outcome.reason_code,expected_reason_codes=['EVIDENCE_INSUFFICIENT'],
        must_not_generate=True,generation_called=gen.calls>0).model_dump())

    class ForgedGenerator:
        calls=0
        def generate(self,*a):
            self.calls+=1
            return json.dumps({'abstain':False,'abstain_reason':None,
                               'answer_blocks':[{'text':'假答案','evidence_ids':['E999']}]})
    fake=ForgedGenerator()
    outcome=build_demo_pipeline(fake).run('如何开启儿童锁',demo_request('forged'))
    observations.append(SafetyRecord(id='forged',attack_type='fake_citation',
        expected_status='failed',actual_status=outcome.final_status,
        actual_reason_code=outcome.reason_code,expected_reason_codes=['MODEL_OUTPUT_OR_CITATION_INVALID'],
        generation_called=fake.calls>0).model_dump())

    gen=DemoGenerator()
    pipeline=build_demo_pipeline(gen)
    from controlled_rag.demo import DemoStore
    protected=demo_corpus()
    protected[0]=protected[0].model_copy(update={'access_labels':frozenset({'secret'})})
    pipeline.retrieval.store=DemoStore(protected)
    outcome=pipeline.run('如何开启儿童锁',demo_request('hidden-evidence'))
    observations.append(SafetyRecord(id='hidden-evidence',attack_type='unauthorized_evidence',
        expected_status='failed',actual_status=outcome.final_status,
        actual_reason_code=outcome.reason_code,expected_reason_codes=['INTEGRITY_OR_CONTROL_FAILURE'],
        must_not_generate=True,generation_called=gen.calls>0,
        leaked_unauthorized=bool(outcome.citations)).model_dump())
    return observations


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='outputs/safety_control_probes.jsonl')
    args=parser.parse_args(argv)
    rows=run_probes()
    target=Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
    print(json.dumps({'cases':len(rows),'output':str(target),'scope':'synthetic deterministic guards'},ensure_ascii=False))


if __name__=='__main__':
    main()
