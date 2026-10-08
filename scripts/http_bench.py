"""Dependency-light HTTP benchmark tracer for a locally running RAG API.

python scripts/http_bench.py --base-url http://127.0.0.1:9000 \
    --requests 30 --concurrency 2 --output outputs/bench_local.jsonl

Only sends public SYNTHETIC questions. Remote load requires --allow-remote.
The same perf_counter origin timestamps all requests for meaningful QPS.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
from controlled_rag.evaluation.schemas import BenchmarkRecord


async def benchmark(base_url: str, request_count: int, concurrency: int, *,
                    transport=None, timeout_seconds: float = 30.0) -> list[dict]:
    if not 1 <= request_count <= 50000 or not 1 <= concurrency <= 1024:
        raise ValueError('invalid request/concurrency count')
    if concurrency > request_count:
        raise ValueError('concurrency cannot exceed request count')
    limit = asyncio.Semaphore(concurrency)
    began = time.perf_counter()

    async with httpx.AsyncClient(base_url=base_url, transport=transport,
                                 timeout=timeout_seconds) as client:
        async def one(n: int):
            async with limit:
                start = (time.perf_counter() - began) * 1000
                try:
                    response = await client.post('/chat', json={'query': '如何开启儿童锁'})
                    status = response.status_code
                except httpx.HTTPError:
                    status = 599  # explicitly record failed network requests
                end = (time.perf_counter() - began) * 1000
                return BenchmarkRecord(id=f'req-{n:06d}',scenario='synthetic_chat',
                                       start_ms=start,end_ms=end,http_status=status,
                                       concurrency=concurrency).model_dump()
        return await asyncio.gather(*(one(n) for n in range(request_count)))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Synthetic HTTP request benchmark')
    parser.add_argument('--base-url',default='http://127.0.0.1:9000')
    parser.add_argument('--requests',type=int,default=30)
    parser.add_argument('--concurrency',type=int,default=2)
    parser.add_argument('--timeout-seconds',type=float,default=30.)
    parser.add_argument('--output',default='outputs/bench_local.jsonl')
    parser.add_argument('--allow-remote',action='store_true')
    args=parser.parse_args(argv)
    parsed=urlparse(args.base_url)
    if parsed.scheme not in ('http','https') or not parsed.hostname:
        parser.error('base-url must be a valid HTTP(S) URL')
    if not args.allow_remote and parsed.hostname not in ('localhost','127.0.0.1','::1'):
        parser.error('non-local benchmark requires explicit --allow-remote')
    if args.timeout_seconds <= 0:
        parser.error('timeout-seconds must be positive')
    data=asyncio.run(benchmark(args.base_url,args.requests,args.concurrency,
                               timeout_seconds=args.timeout_seconds))
    target=Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data),encoding='utf-8')
    print(json.dumps({'requests':len(data),'non_2xx':sum(not 200<=r['http_status']<300 for r in data),
                      'trace':str(target),'source':'synthetic prompts sent to real HTTP endpoint'},ensure_ascii=False))


if __name__=='__main__':
    main()
