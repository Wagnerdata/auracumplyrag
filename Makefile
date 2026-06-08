.PHONY: install ingest run-api run-ui test docker-up docker-down clean

PYTHON ?= python
PIP    ?= pip

install:
	$(PIP) install -r requirements.txt

ingest:
	$(PYTHON) ingest.py

run-api:
	uvicorn api:app --reload --host 0.0.0.0 --port 8001

run-ui:
	streamlit run app.py --server.port 8501

test:
	pytest tests/ -v

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	rm -rf chroma_db __pycache__ .pytest_cache **/__pycache__
