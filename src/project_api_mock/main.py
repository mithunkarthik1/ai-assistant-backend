"""Development-only Project API used for local end-to-end testing.

This service deliberately implements the same HTTP contract expected by the
agent's ProjectApiClient. It must be replaced by the real Project API before
production use.
"""

from copy import deepcopy

from fastapi import FastAPI, HTTPException

app = FastAPI(title="Mock Project API", version="0.1.0")

_PROJECTS = {
    "PROJ-123": {
        "id": "PROJ-123",
        "name": "WorkPilot Platform Upgrade",
        "description": "Internal platform upgrade project for the WorkPilot team.",
        "status": "In Progress",
        "owner": {
            "id": "USR-001",
            "name": "Aarav Mehta",
            "email": "aarav.mehta@example.com",
        },
        "milestones": [
            {"id": "M-001", "name": "Requirements complete", "status": "Completed"},
            {"id": "M-002", "name": "Implementation", "status": "In Progress"},
            {"id": "M-003", "name": "Production rollout", "status": "Planned"},
        ],
    }
}


def _get_project(project_id: str) -> dict:
    project = _PROJECTS.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id!r} was not found.")
    return deepcopy(project)


@app.get("/api/v1/projects/{project_id}")
async def get_project(project_id: str) -> dict:
    """Return complete details for a project."""
    return _get_project(project_id)


@app.get("/api/v1/projects/{project_id}/status")
async def get_project_status(project_id: str) -> dict:
    """Return the current project status."""
    project = _get_project(project_id)
    return {"id": project["id"], "status": project["status"]}


@app.get("/api/v1/projects/{project_id}/owner")
async def get_project_owner(project_id: str) -> dict:
    """Return the project owner."""
    project = _get_project(project_id)
    return {"id": project["id"], "owner": project["owner"]}


@app.get("/api/v1/projects/{project_id}/milestones")
async def get_project_milestones(project_id: str) -> dict:
    """Return project milestones."""
    project = _get_project(project_id)
    return {"id": project["id"], "milestones": project["milestones"]}
