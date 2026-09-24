# WorkPilot AI Assistant Backend

FastAPI backend for the WorkPilot assistant. The application combines the existing policy RAG workflow with a separate Agentic AI workflow for project-related requests.

## Architecture

```text
Frontend
   |
   v
POST /api/v1/assistant/chat
   |
   v
src/assistant/service.py
   |-- policy request  -> src/rag/service.py
   |-- project request -> src/agents/service.py -> Project API
   `-- general request -> src/agents/service.py -> LLM
```

The assistant module is the central orchestrator. The RAG and agent modules own their separate business workflows and do not call each other directly.

## Source structure

```text
src/
├── main.py
├── core/
│   └── config.py              # Shared environment-backed settings
├── database/
│   ├── __init__.py
│   └── connection.py          # SQLAlchemy engine, sessions, initialization
├── assistant/
│   ├── api.py                 # Unified assistant endpoint
│   ├── router.py              # Deterministic intent classification
│   ├── service.py             # Central request orchestration
│   └── schema.py              # Unified request/response models
├── rag/
│   ├── api.py                 # Existing policy endpoint
│   ├── service.py             # Existing RAG workflow
│   ├── schema.py              # RAG API models
│   └── model.py               # RAG persistence model
├── agents/
│   ├── api.py                 # Direct agent endpoint
│   ├── service.py             # Agent, LangGraph, LLM, and Project API workflow
│   ├── schema.py              # Agent models
│   └── errors.py              # Agent and Project API errors
└── project_api_mock/
    └── main.py                # Development-only Project API
```

All project documentation is maintained in this root README.

## Request routing

The unified endpoint classifies requests before invoking the LLM:

- Policy terms such as PTO or leave route to the existing RAG service.
- A project ID or project follow-up routes to the agent and Project API.
- General questions route to the direct LLM path.
- A project request without an ID returns a clarification response.

The main endpoint is:

```text
POST /api/v1/assistant/chat
```

Optional direct endpoints are:

```text
POST /api/v1/chat
POST /api/v1/agents/chat
```

## Project API workflow

The agent supports these operations:

```text
GET /api/v1/projects/{project_id}
GET /api/v1/projects/{project_id}/status
GET /api/v1/projects/{project_id}/owner
GET /api/v1/projects/{project_id}/milestones
```

For local testing, Docker Compose starts `project-api`, a development-only mock service containing sample data for `PROJ-123`. Replace it with the real Project API before production use.

Conversation state is process-local and keyed by `session_id`. The current project ID is retained for follow-up questions. It is not persisted in SQL in this version.

## Configuration

Copy the example file and configure the shared LLM settings:

```powershell
Copy-Item .env.example .env
```

The application supports the existing shared variables:

```env
LLM_API_KEY=...
LLM_MODEL=...
LLM_BASE_URL=...
PROJECT_API_BASE_URL=http://host.docker.internal:9000
PROJECT_API_PATH_PREFIX=/api/v1/projects
PROJECT_API_TIMEOUT_SECONDS=5
```

When running with Docker Compose, the backend internally uses `http://project-api:9000` for the mock service. The Compose setting overrides the host-facing value above.

## Run with Docker Compose

From this directory:

```powershell
docker compose up -d --build --force-recreate --remove-orphans
```

Services:

```text
backend      http://localhost:8001
project-api  http://localhost:9000
postgres     localhost:5433
```

Health check:

```powershell
Invoke-RestMethod http://localhost:8001/api/v1/health
```

Mock Project API check:

```powershell
Invoke-RestMethod http://localhost:9000/api/v1/projects/PROJ-123/status
```

## Test the unified assistant

```powershell
$body = @{
    message = "What is the status of project PROJ-123?"
    session_id = "demo-session"
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://localhost:8001/api/v1/assistant/chat" `
    -ContentType "application/json" `
    -Body $body | ConvertTo-Json -Depth 10
```

Run the automated tests with:

```powershell
python -m pytest tests -q
```

## Logging and error handling

The backend logs the incoming request, selected route, selected tool, Project API request and result, final response, and errors. Project API failures, invalid IDs, timeouts, missing data, and LLM failures are normalized into application-level errors.
