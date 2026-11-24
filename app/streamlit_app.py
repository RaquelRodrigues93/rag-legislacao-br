"""Demo: streamlit run app/streamlit_app.py"""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import anthropic  # noqa: E402
import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from legisrag.generator import CredentialsError  # noqa: E402
from legisrag.pipeline import get_pipeline  # noqa: E402

load_dotenv()

st.set_page_config(page_title="LegisRAG — LGPD e CDC", page_icon="⚖️", layout="centered")

st.title("⚖️ LegisRAG")
st.caption(
    "Perguntas sobre a **LGPD** e o **Código de Defesa do Consumidor**, respondidas com "
    "citações literais da lei. Não substitui orientação jurídica."
)

EXAMPLES = [
    "Comprei um produto pela internet e me arrependi. Posso devolver?",
    "Posso pedir para uma empresa apagar meus dados?",
    "Qual o valor máximo da multa por descumprir a LGPD?",
    "Meu produto veio com defeito e a loja não consertou em 30 dias. O que posso exigir?",
]


@st.cache_resource(show_spinner="Carregando índice e modelo de embeddings…")
def pipeline():
    return get_pipeline()


with st.sidebar:
    st.subheader("Exemplos")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.question = ex
    k = st.slider("Artigos recuperados (k)", 3, 12, 6)

question = st.text_input("Sua pergunta", key="question", placeholder=EXAMPLES[0])

if question:
    try:
        with st.spinner("Buscando artigos e gerando a resposta…"):
            ans = pipeline().ask(question, k=k)
    except CredentialsError as e:
        st.error(str(e))
        st.stop()
    except anthropic.APIError as e:
        st.error(f"Erro ao chamar a API do Claude: {e}")
        st.stop()

    (st.warning if ans.abstained else st.markdown)(ans.text)

    if ans.citations:
        st.subheader("Citações")
        shown = set()
        for c in ans.citations:
            if (c.source, c.cited_text) in shown:
                continue
            shown.add((c.source, c.cited_text))
            st.markdown(f"**[{c.source}] [{c.label}]({c.url})**")
            st.markdown(f"> {c.cited_text}")

    with st.expander("Artigos recuperados"):
        for r in ans.sources:
            st.markdown(f"**{r.rank}. [{r.article.citation_label}]({r.article.url})**")
            st.caption(r.article.hierarchy)
            st.text(r.article.text[:1500])
