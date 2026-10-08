"""Pure evaluation entrypoint. Models, annotations and backend traces are inputs."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Type

from pydantic import ValidationError

from .ablation import evaluate_ablation
from .benchmark import evaluate_benchmark
from .badcase import evaluate_badcases
from .citation import evaluate_citation
from .dataset import audit_splits
from .generation import evaluate_generation
from .regression import evaluate_release_gate
from .retrieval import evaluate_reranker, evaluate_retrieval
from .round2 import evaluate_round2
from .router import evaluate_router
from .safety import evaluate_safety
from .schemas import (AblationRecord, BadCaseRecord, BenchmarkRecord, CitationRecord, GenerationRecord,
                      RetrievalRecord, RerankerRecord, Round2Record, RouterRecord,
                      SafetyRecord)

SUITES: dict[str, tuple[Type | None, object]] = {
    'retrieval': (RetrievalRecord, evaluate_retrieval),
    'reranker': (RerankerRecord, evaluate_reranker),
    'router': (RouterRecord, evaluate_router),
    'round2': (Round2Record, evaluate_round2),
    'generation': (GenerationRecord, evaluate_generation),
    'citation': (CitationRecord, evaluate_citation),
    'safety': (SafetyRecord, evaluate_safety),
    'benchmark': (BenchmarkRecord, evaluate_benchmark),
    'ablation': (AblationRecord, evaluate_ablation),
    'badcase': (BadCaseRecord, evaluate_badcases),
    'split-audit': (None, audit_splits),
}


def load_jsonl(path: str | Path, schema: Type | None, suite: str) -> list:
    path = Path(path)
    output = []
    seen = set()
    with path.open(encoding='utf-8-sig') as file:
        for line_no, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise ValueError('JSONL must contain objects')
                record = schema.model_validate(data) if schema else data
            except (ValueError, TypeError, ValidationError) as exc:
                raise ValueError(f'{path}:{line_no}: invalid {suite} record: {exc}') from exc
            if suite not in ('generation', 'ablation', 'split-audit'):
                uid = record.id
                if uid in seen:
                    raise ValueError(f'{path}:{line_no}: duplicate sample ID {uid}')
                seen.add(uid)
            output.append(record)
    if not output:
        raise ValueError(f'no JSONL records at {path}')
    return output


def evaluate_suite(suite: str, input_path: str | Path, *, provenance: str = 'synthetic',
                   model_version: str | None = None, index_version: str | None = None,
                   ablation_baseline: str = 'baseline') -> dict:
    """provenance is caller-supplied metadata, not a fact verified by the script."""
    if suite not in SUITES:
        raise ValueError(f'unknown suite: {suite}')
    if provenance not in ('synthetic', 'real_unverified', 'real_audited'):
        raise ValueError('unsupported provenance label')
    path = Path(input_path)
    data = load_jsonl(path, SUITES[suite][0], suite)
    if suite == 'ablation':
        metrics = evaluate_ablation(data, baseline=ablation_baseline)
    else:
        metrics = SUITES[suite][1](data)
    return {
        'suite': suite,
        'provenance': {
            'source_type': provenance,
            'input_path': str(path),
            'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'run_time_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'model_version': model_version,
            'index_version': index_version,
            'semantic_labels_externally_verified': False if provenance != 'real_audited' else 'declared_by_operator',
            'warning': 'Synthetic fixtures are not model or production performance results.'
                       if provenance == 'synthetic' else 'Trace provenance and labels require independent audit.',
        },
        'metrics': metrics,
    }


def write_json(path: str | Path, report: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_json(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('JSON object expected')
    return data


def combine_reports(reports: list[dict]) -> dict:
    if not reports:
        raise ValueError('no reports to combine')
    combined = {'provenance': 'combined component reports, see each subreport', 'metrics': {}, 'source_reports': {}}
    for report in reports:
        name = report['suite']
        if name in combined['metrics']:
            raise ValueError(f'duplicate suite: {name}')
        combined['metrics'][name] = report['metrics']
        combined['source_reports'][name] = report['provenance']
    return combined


def check_gate(report: dict, rules: dict, baseline: dict | None = None) -> dict:
    if 'metrics' not in report:
        raise ValueError('release gate input requires metrics')
    return evaluate_release_gate(report['metrics'], rules,
                                 baseline['metrics'] if baseline is not None else None)
