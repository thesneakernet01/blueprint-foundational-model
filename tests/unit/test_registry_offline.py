# SPDX-License-Identifier: Apache-2.0
"""Off-CML behavior of the Model Registry integration: everything must degrade
to a clear "unavailable + reason", never an exception or a 500."""

import pytest

from tfm_demo import registry


@pytest.fixture(autouse=True)
def _no_cml_env(monkeypatch):
    for var in ("CDSW_API_URL", "CDSW_APIV2_KEY", "CDSW_PROJECT_ID"):
        monkeypatch.delenv(var, raising=False)


def test_available_false_with_reason_off_cml():
    ok, reason = registry.available()
    assert ok is False
    assert "Cloudera" in reason


def test_status_reports_unavailable_without_raising():
    s = registry.status()
    assert s["available"] is False
    assert s["reason"]
    assert s["versions"] == []
    assert s["model"] is None


def test_register_raises_cleanly_off_cml():
    with pytest.raises(RuntimeError, match="unavailable"):
        registry.register_latest(progress=lambda m: None,
                                 on_stage=lambda s: None)


def test_api_routes_gate_off_cml():
    httpx = pytest.importorskip("httpx")           # noqa: F841
    from fastapi.testclient import TestClient
    from tfm_demo.app import app

    with TestClient(app) as client:
        r = client.get("/api/registry")
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is False
        assert body["job"]["state"] == "idle"

        r = client.post("/api/registry/deploy")
        assert r.status_code == 503
        assert "unavailable" in r.json()["error"]

        r = client.get("/api/runs")
        assert r.status_code == 200
        body = r.json()
        assert body["next_budget"]["embed_max"] <= 20000
        assert len(body["schedule"]) == 5
