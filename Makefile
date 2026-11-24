.PHONY: install ingest index eval eval-gen test lint format app api docker-build docker-run

install:        ## Instala dependências (incluindo dev)
	uv sync --extra dev

ingest:         ## Baixa as leis do Planalto e gera data/processed/chunks.jsonl
	uv run legisrag ingest -v

index:          ## Calcula os embeddings (baixa o modelo na primeira vez)
	uv run legisrag index

eval:           ## Avaliação de recuperação (gratuita, sem API)
	uv run legisrag eval

eval-gen:       ## Avaliação de recuperação + geração (chama a API do Claude, tem custo)
	uv run legisrag eval --generation --name generation

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff format . && uv run ruff check --fix .

app:            ## Interface web em http://localhost:8501
	uv run streamlit run app/streamlit_app.py

api:            ## API em http://localhost:8000/docs
	uv run uvicorn legisrag.api:app --reload

docker-build:
	docker build -t legisrag .

docker-run:
	docker run --rm -p 8501:8501 --env-file .env legisrag
