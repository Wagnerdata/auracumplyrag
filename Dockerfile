FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# The kernel SDK lives in the sibling repo (../AuraComply/sdk), outside
# this build context. It arrives as the named build context `sdk`:
# docker compose passes it via `additional_contexts`; a plain build
# needs `--build-context sdk=../AuraComply/sdk`. The `file:` line in
# requirements.txt only resolves on the host, so it is filtered out
# here and the SDK is installed from the copied tree instead.
COPY --from=sdk . /tmp/auracomply-sdk
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install /tmp/auracomply-sdk \
    && grep -v '^auracomply-sdk @ file:' requirements.txt > /tmp/requirements.docker.txt \
    && pip install -r /tmp/requirements.docker.txt \
    && rm -rf /tmp/auracomply-sdk /tmp/requirements.docker.txt

COPY . .

EXPOSE 8001 8501

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
