#!/bin/sh
set -e

# Run database migrations with Alembic if DATABASE_URL is available
# and this is not the standalone mock project api
if [ -n "$DATABASE_URL" ] && ! echo "$*" | grep -q "project_api_mock"; then
    echo "==> Running Alembic migrations (alembic upgrade head)..."
    alembic upgrade head
    echo "==> Alembic migrations applied successfully."
fi

# Execute CMD arguments or default to uvicorn
if [ $# -eq 0 ]; then
    echo "==> Starting WorkPilot AI Assistant backend..."
    exec uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
else
    echo "==> Starting service with command: $*"
    exec "$@"
fi
