"""Paired ablation execution of actual Pipeline variants on identical cases.

Each variant is built independently. For real experiments the external factory
must implement requested component substitutions and supply the same frozen
corpus/index version. This runner never creates or assumes such infrastructure.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..demo import build_demo_pipeline, demo_request
from .ablation import evaluate_ablation
from .runner import write_json
from .schemas import AblationRecord
from .workflow_runner import (load_cases, load_factory, run_workflow_cases,
                              save_workflow_report)


class Variant(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    name: str = Field(pattern=r'^[A-Za-z0-9_-]+$', min_length=1)
    settings: dict[str, int | float | bool | str]


class Matrix(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    baseline: str
    variants: list[Variant] = Field(min_length=2)

    @model_validator(mode='after')
    def check(self):
        names = [v.name for v in self.variants]
        if len(set(names)) != len(names) or self.baseline not in names:
            raise ValueError('matrix requires unique variants including baseline')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', self.baseline):
            raise ValueError('unsafe baseline name')
        if len({json.dumps(v.settings, sort_keys=True) for v in self.variants}) < 2:
            raise ValueError('at least two distinct variant configurations required')
        return self


def read_matrix(path: str | Path) -> Matrix:
    return Matrix.model_validate_json(Path(path).read_text(encoding='utf-8'))


def run_matrix(*, cases_path: str | Path, matrix_path: str | Path,
               output: str | Path, mode: str,
               factory: str | None = None, bootstraps: int = 1000,
               seed: int = 2026) -> dict:
    if mode not in ('synthetic', 'real'):
        raise ValueError('mode must be synthetic or real')
    if bootstraps < 0:
        raise ValueError('bootstraps must not be negative')
    if mode == 'real' and not factory:
        raise ValueError('real mode requires a variant-aware --factory')
    if mode == 'synthetic' and factory:
        raise ValueError('synthetic mode cannot use a real factory')
    matrix = read_matrix(matrix_path)
    cases = load_cases(cases_path)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    variant_factory = load_factory(factory) if factory else None
    ablation_records: list[AblationRecord] = []
    summary = {}
    for variant in matrix.variants:
        if mode == 'synthetic':
            if set(variant.settings) != {'max_retrieval_rounds'}:
                raise ValueError('synthetic matrix supports max_retrieval_rounds only')
            pipeline, request_factory = build_demo_pipeline(), demo_request
            pipeline.config = replace(
                pipeline.config, max_retrieval_rounds=variant.settings['max_retrieval_rounds'])
        else:
            pipeline, request_factory = variant_factory(variant.name, dict(variant.settings))
            # Fail loudly when a declared control toggle was not applied.
            if ('max_retrieval_rounds' in variant.settings and
                    pipeline.config.max_retrieval_rounds != variant.settings['max_retrieval_rounds']):
                raise ValueError(f'{variant.name}: factory did not apply max_retrieval_rounds')
        records, case_metrics = run_workflow_cases(pipeline, request_factory, cases)
        if set(case_metrics) != {c.id for c in cases}:
            raise ValueError('variant result missing one or more case IDs')
        report = save_workflow_report(destination / 'variants' / variant.name, records,
                                      mode, cases_path,
                                      context={'variant': variant.name,
                                               'settings': variant.settings})
        summary[variant.name] = {
            'settings': variant.settings,
            'sample_count': len(records),
            'workflow_metrics': report['metrics'],
        }
        for case in cases:
            ablation_records.append(AblationRecord(
                id=case.id, variant=variant.name, metrics=case_metrics[case.id]))
    with (destination / 'ablation_cases.jsonl').open('w', encoding='utf-8') as f:
        for row in ablation_records:
            f.write(row.model_dump_json() + '\n')
    comparison = evaluate_ablation(ablation_records, matrix.baseline,
                                   bootstraps=bootstraps, seed=seed)
    result = {
        'provenance': {
            'source_type': mode,
            'cases_sha256': hashlib.sha256(Path(cases_path).read_bytes()).hexdigest(),
            'matrix_sha256': hashlib.sha256(Path(matrix_path).read_bytes()).hexdigest(),
            'factory': factory if mode == 'real' else 'controlled_rag.demo:build_demo_pipeline',
            'synthetic_notice': ('Demo retrievers, scorer, router and generator only; '
                                 'NOT real retrieval, DPO or GPU performance scores.') if mode == 'synthetic' else None,
            'gold_labels': 'provided externally; runner does not audit labels',
        },
        'variants': summary,
        'comparison': comparison,
    }
    write_json(destination / 'ablation_report.json', result)
    return result
