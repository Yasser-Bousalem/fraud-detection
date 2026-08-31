import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "model_version" in r.json()


def test_version(client):
    r = client.get("/version")
    assert r.status_code == 200
    assert "threshold" in r.json()


def test_score_valid_transaction(client):
    payload = {
        "TransactionAmt": 300.0, "ProductCD": "C", "card1": 13926,
        "card4": "visa", "card6": "credit", "addr1": 325.0,
        "P_emaildomain": "gmail.com", "DeviceType": "mobile", "DeviceInfo": "iOS Device",
    }
    r = client.post("/score", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_score"] <= 1.0
    assert body["decision"] in ("review", "approve")


def test_score_rejects_negative_amount(client):
    r = client.post("/score", json={"TransactionAmt": -50.0})
    assert r.status_code == 422


def test_score_missing_required_field(client):
    r = client.post("/score", json={"ProductCD": "C"})
    assert r.status_code == 422