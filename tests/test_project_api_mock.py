"""Tests for the development-only mock Project API."""

from fastapi.testclient import TestClient

from src.project_api_mock.main import app


def test_mock_project_status():
    with TestClient(app) as client:
        response = client.get("/api/v1/projects/PROJ-123/status")

    assert response.status_code == 200
    assert response.json()["status"] == "In Progress"


def test_mock_project_owner():
    with TestClient(app) as client:
        response = client.get("/api/v1/projects/PROJ-123/owner")

    assert response.status_code == 200
    assert response.json()["owner"]["name"] == "Aarav Mehta"


def test_mock_project_not_found():
    with TestClient(app) as client:
        response = client.get("/api/v1/projects/UNKNOWN-999/status")

    assert response.status_code == 404
