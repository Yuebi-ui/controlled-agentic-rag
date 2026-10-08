"""Latency / throughput aggregation from timestamped benchmark request traces.

Throughput assumes timestamps share the SAME monotonic clock, and the input
is a complete window including failed requests; the runner must guarantee it.
"""
from __future__ import annotations

from .metrics import avg, divide, nearest_rank
from .schemas import BenchmarkRecord


def _summarize(rows: list[BenchmarkRecord]) -> dict:
    if not rows:
        raise ValueError('empty benchmark sample group')
    latencies = [r.end_ms - r.start_ms for r in rows]
    window_ms = max(r.end_ms for r in rows) - min(r.start_ms for r in rows)
    status_ok = sum(200 <= r.http_status < 300 for r in rows)
    gpumem = [r.gpu_memory_mb for r in rows if r.gpu_memory_mb is not None]
    return {
        'request_count': len(rows), 'success_count': status_ok,
        'non_2xx_count': len(rows) - status_ok, 'success_rate': status_ok/len(rows),
        'p50_latency_ms': nearest_rank(latencies, .50),
        'p95_latency_ms': nearest_rank(latencies, .95),
        'p99_latency_ms': nearest_rank(latencies, .99),
        'mean_latency_ms': avg(latencies),
        'throughput_requests_per_second': divide(len(rows) * 1000, window_ms),
        'measurement_window_ms': window_ms,
        'gpu_peak_mb_observed': max(gpumem) if gpumem else None,
        'gpu_sample_count': len(gpumem),
    }


def evaluate_benchmark(rows: list[BenchmarkRecord]) -> dict:
    if not rows:
        raise ValueError('empty benchmark dataset')
    buckets: dict[tuple[str, int], list[BenchmarkRecord]] = {}
    for row in rows:
        buckets.setdefault((row.scenario, row.concurrency), []).append(row)
    return {'overall': _summarize(rows),
            'by_scenario_and_concurrency': {
                f'{s}|concurrency={c}': _summarize(group)
                for (s, c), group in sorted(buckets.items())},
            'timing_caveat': 'A shared monotonic timestamp origin is required to interpret QPS.'}
