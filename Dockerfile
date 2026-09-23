FROM python:3.12-slim

WORKDIR /app

# Install system utilities and build dependencies for psycopg/c-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install --no-cache-dir poetry

# Copy dependency definition files
COPY pyproject.toml poetry.lock* ./

# Configure poetry to install packages directly into container python environment
RUN poetry config virtualenvs.create false \
    && poetry lock \
    && poetry install --no-interaction --no-ansi --no-root --without dev

# Copy application source code
COPY . .


EXPOSE 8001

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
