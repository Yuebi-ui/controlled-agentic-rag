from fastapi.testclient import TestClient

from controlled_rag.api import create_app


def test_demo_api_only_synthetic():
    with TestClient(create_app()) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["production_ready"] is False
        response = client.post("/chat", json={"query": "如何开启儿童锁"})
        assert response.status_code == 200
        assert response.json()["final_status"] == "answered"
        assert response.json()["citations"][0]["source_uri"].startswith("demo://")
        rejected = client.post("/chat", json={"query": ""})
        assert rejected.status_code == 422
