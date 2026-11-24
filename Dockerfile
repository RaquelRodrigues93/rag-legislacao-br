FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

# Dependências primeiro (camada cacheada enquanto o lockfile não mudar).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY src ./src
COPY app ./app
COPY eval ./eval
COPY data/processed ./data/processed
RUN uv sync --frozen

# Baixa o modelo de embeddings e cria o índice no build: o container já sobe pronto.
RUN uv run legisrag index

EXPOSE 8501 8000
# Padrão: interface Streamlit. Para a API: docker run ... uv run uvicorn legisrag.api:app --host 0.0.0.0
CMD ["uv", "run", "streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
