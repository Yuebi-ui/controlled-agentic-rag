"""Fail-closed release checks: missing target metrics or absent baselines fail.

Rule configuration is explicit. A CI green badge only means these declared
rules passed for the supplied input reports, NOT production readiness.
"""
from __future__ import annotations

import math
from collections.abc import Mapping


def metric_at(report: Mapping, path: str):
    node = report
    for part in path.split('.'):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _number(report, path):
    value = metric_at(report, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def evaluate_release_gate(candidate: dict, rules: dict, baseline: dict | None = None) -> dict:
    """rules = {absolute: {path: {min/max}}, delta: {path: {min/max}}}.

    Delta means candidate - baseline. For a maximum acceptable drop of .02,
    use {"delta": {"metric": {"min": -0.02}}}.
    """
    if not isinstance(rules, dict) or not rules or not set(rules).issubset({'absolute', 'delta'}):
        raise ValueError('release gate requires absolute/delta rule objects')
    checks = []
    for category, constraints in rules.items():
        if not isinstance(constraints, dict) or not constraints:
            raise ValueError('each gate category requires rules')
        if category == 'delta' and baseline is None:
            raise ValueError('delta rules require a baseline report')
        for path, bounds in constraints.items():
            if not isinstance(bounds, dict) or not bounds or not set(bounds).issubset({'min', 'max'}):
                raise ValueError(f'invalid release bound {path}')
            for name, limit in bounds.items():
                if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit):
                    raise ValueError(f'invalid {name} release threshold for {path}')
            new = _number(candidate, path)
            old = _number(baseline, path) if category == 'delta' else None
            # Missing or invalid candidate/baseline is always a hard failure.
            val = (new - old if new is not None and old is not None else None) if category == 'delta' else new
            passed = (val is not None and
                      ('min' not in bounds or val >= bounds['min']) and
                      ('max' not in bounds or val <= bounds['max']))
            checks.append({'category': category, 'metric': path, 'observed': val,
                           'bounds': bounds, 'passed': bool(passed),
                           'reason': 'ok' if passed else ('missing_or_invalid_metric' if val is None else 'threshold_violation')})
    return {'passed': bool(checks) and all(c['passed'] for c in checks),
            'check_count': len(checks), 'failed_checks': [c for c in checks if not c['passed']],
            'checks': checks}
