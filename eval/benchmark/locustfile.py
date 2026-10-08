"""Optional load testing; never include credentials or protected real questions.

locust -f eval/benchmark/locustfile.py --host http://127.0.0.1:9000 \
    --headless -u 2 -r 1 -t 30s --csv outputs/locust/synthetic

Only a synthetic query is sent. This file requires the optional `locust` package.
"""
from __future__ import annotations

from locust import HttpUser, between, task


class ControlledRAGSyntheticUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def synthetic_chat(self):
        with self.client.post('/chat', json={'query': '如何开启儿童锁'},
                              catch_response=True, name='/chat synthetic') as response:
            if response.status_code != 200:
                response.failure(f'HTTP {response.status_code}')
                return
            try:
                payload = response.json()
            except ValueError:
                response.failure('invalid response JSON')
                return
            if payload.get('final_status') not in ('answered', 'abstained'):
                response.failure(f'Unexpected final_status={payload.get("final_status")}')
