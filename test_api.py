import pytest
from starlette.testclient import TestClient
from api import app

@pytest.fixture(scope="module")
def client() -> TestClient:
    # using TC as context manager triggers FastAPI's startup/shutdown
    with TestClient(app) as TC:
        yield TC
        
def test_route_valid_query(client: TestClient) -> None:
    response = client.post("/route", json = {"query": "Write Python script to reverse a string."})
    assert response.status_code == 200
    data = response.json()
    assert data["route"] in ["simple", "complex"]
    assert "sim_simple" in data
    
def test_route_empty_string(client: TestClient) -> None:
    response = client.post("/route", json = {"query": ""})
    assert response.status_code == 422 # pydantic validation error
    
def test_route_too_long(client: TestClient) -> None:
    response = client.post("/route", json = {"query": "A" * 2001})
    assert response.status_code == 422

def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"