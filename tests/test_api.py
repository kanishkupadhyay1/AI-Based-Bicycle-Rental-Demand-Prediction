"""Integration tests for FastAPI backend endpoints."""

from fastapi.testclient import TestClient

from src.api.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        res = client.get("/")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "healthy"
        assert body["models_loaded"] is True
        assert body["dataset_loaded"] is True


def test_metrics_and_comparison_endpoints():
    with TestClient(app) as client:
        m_res = client.get("/api/metrics")
        assert m_res.status_code == 200
        m_body = m_res.json()
        assert len(m_body["metrics"]) == 4
        assert m_body["total_records"] == 17379

        c_res = client.get("/api/model-comparison")
        assert c_res.status_code == 200
        c_body = c_res.json()
        assert len(c_body["recent_time_series"]) > 0
        assert len(c_body["residual_stats"]) == 4


def test_demand_analysis_and_gam_effects_endpoints():
    with TestClient(app) as client:
        d_res = client.get("/api/demand-analysis")
        assert d_res.status_code == 200
        d_body = d_res.json()
        assert len(d_body["hourly_demand"]) == 24

        g_res = client.get("/api/gam-effects")
        assert g_res.status_code == 200
        g_body = g_res.json()
        assert len(g_body["effects"]) == 6


def test_predict_and_predict_all_endpoints():
    payload = {
        "season": 2,
        "yr": 1,
        "mnth": 6,
        "hr": 17,
        "holiday": 0,
        "weekday": 3,
        "workingday": 1,
        "weathersit": 1,
        "temp": 0.68,
        "atemp": 0.64,
        "hum": 0.50,
        "windspeed": 0.15,
    }
    with TestClient(app) as client:
        p_res = client.post("/api/predict", json={**payload, "model": "gam"})
        assert p_res.status_code == 200
        assert p_res.json()["predicted_demand"] >= 0

        all_res = client.post("/api/predict-all", json=payload)
        assert all_res.status_code == 200
        assert len(all_res.json()["predictions"]) == 4


def test_predict_validation_rejects_invalid_hour():
    bad_payload = {
        "season": 2,
        "yr": 1,
        "mnth": 6,
        "hr": 99,
        "holiday": 0,
        "weekday": 3,
        "workingday": 1,
        "weathersit": 1,
        "temp": 0.68,
        "atemp": 0.64,
        "hum": 0.50,
        "windspeed": 0.15,
        "model": "gam",
    }
    with TestClient(app) as client:
        res = client.post("/api/predict", json=bad_payload)
        assert res.status_code == 422
