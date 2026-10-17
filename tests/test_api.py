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


def test_demo_http_key_required_when_explicitly_configured():
    with TestClient(create_app(demo_api_key="test-secret-token")) as client:
        missing = client.post("/chat", json={"query": "演示"})
        assert missing.status_code == 401
        invalid = client.post("/chat", json={"query": "演示"}, headers={"Authorization": "Bearer bad"})
        assert invalid.status_code == 401
        accepted = client.post("/chat", json={"query": "如何开启儿童锁"}, headers={"Authorization": "Bearer test-secret-token"})
        assert accepted.status_code == 200


def test_demo_rejects_nonlocal_clients_without_key():
    with TestClient(create_app(), client=("192.0.2.4", 12345)) as client:
        assert client.post("/chat", json={"query": "演示"}).status_code == 403


def test_demo_uri_redaction_does_not_change_fictional_demo_uri():
    from controlled_rag.api import _demo_public_payload
    obj = {"citations": [
        {"source_uri": "file:///private/manual.pdf", "image_uri": "s3://internal/private.png"},
        {"source_uri": "demo://fake-pdf", "image_uri": "demo://fake-image"},
    ]}
    output = _demo_public_payload(obj)
    assert output["citations"][0] == {"source_uri": None, "image_uri": None}
    assert output["citations"][1] == {"source_uri": "demo://fake-pdf", "image_uri": "demo://fake-image"}
