from fastapi.testclient import TestClient

from sqms_ai_orchestrator.main import app


def test_search_endpoint_uses_real_knowledge_base() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/search",
            json={"query": "como adicionar aprovadores", "flow_id": "procurement", "limit": 5},
        )
        assert response.status_code == 200
        results = response.json()["results"]
        assert results
        assert results[0]["document"] == "RAG_SQMS_CAMPOS.md"


def test_health_reports_indexed_sections() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["sections"] > 100
