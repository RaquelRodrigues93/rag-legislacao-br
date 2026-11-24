# legisrag

Perguntas e respostas sobre a **LGPD** (Lei nº 13.709/2018) e o **Código de Defesa do
Consumidor** (Lei nº 8.078/1990) com **citações verificáveis**: cada afirmação da resposta aponta
para um trecho literal do artigo de lei, com link para o texto oficial no Planalto.

> Projeto educacional. Não substitui orientação jurídica.

Exemplo ilustrativo do formato da saída:

```
$ legisrag ask "Comprei um produto pela internet e me arrependi. Posso devolver?"

Sim. Em compras feitas fora do estabelecimento comercial, como pela internet, você pode desistir
do contrato no prazo de 7 dias a contar da assinatura ou do recebimento do produto [1]. ...

[1] Código de Defesa do Consumidor, Art. 49: "O consumidor pode desistir do contrato, no prazo
    de 7 dias a contar de sua assinatura ou do ato de recebimento do produto ou serviço..."
    http://www.planalto.gov.br/ccivil_03/leis/l8078compilado.htm#art49
```

## Como funciona

```
Planalto (HTML) ──► parser ──► chunks por artigo ──► embeddings (e5) + BM25
                                                          │
pergunta ──────────────────────────────────► busca (densa | híbrida | BM25)
                                                          │ top-k artigos
                                                          ▼
                                   Claude + API de Citations ──► resposta com trechos citados
```

| Etapa | Arquivo | O que faz |
|---|---|---|
| Fontes | [sources.py](src/legisrag/sources.py) | Lista as leis indexadas. Para incluir outra lei, basta adicionar uma entrada. |
| Ingestão | [ingest.py](src/legisrag/ingest.py) | Baixa o texto compilado do Planalto (com cache em `data/raw/`). |
| Parser | [parser.py](src/legisrag/parser.py) | Converte o HTML em um chunk por artigo, com a hierarquia (título, capítulo, seção). |
| Busca | [retriever.py](src/legisrag/retriever.py) | Busca densa, BM25 ou híbrida (RRF ponderado), agregando os resultados por artigo. |
| Geração | [generator.py](src/legisrag/generator.py) | Envia os artigos ao Claude como documentos com citações habilitadas. |
| Avaliação | [evaluation.py](src/legisrag/evaluation.py) | Métricas de recuperação e de geração sobre [eval/dataset.jsonl](eval/dataset.jsonl). |
| Interfaces | [cli.py](src/legisrag/cli.py), [api.py](src/legisrag/api.py), [streamlit_app.py](app/streamlit_app.py) | CLI, API HTTP (FastAPI) e interface web (Streamlit). |

## Decisões de design

**Um chunk por artigo.** O artigo é a unidade que juristas citam e que o usuário espera ver como
fonte. Chunks de tamanho fixo cortam incisos no meio e misturam artigos vizinhos. Artigos longos
(acima de 1.800 caracteres) são divididos em partes que repetem o caput, para que nenhuma parte
perca o sentido; na hora de responder, as partes são reunidas e o modelo recebe o artigo inteiro.

**Só o texto vigente.** As páginas compiladas do Planalto mantêm as redações revogadas riscadas
logo antes da redação atual. O parser descarta o texto riscado (`<strike>` ou `line-through`),
as anotações editoriais ("Redação dada pela Lei nº ...") e os artigos inteiramente vetados ou
revogados. Sem esse filtro o sistema responderia com a lei desatualizada.

**A hierarquia entra no índice.** "Capítulo II — Do tratamento de dados pessoais > Seção II — Do
tratamento de dados pessoais sensíveis" é indexado junto com o artigo. Isso ajuda quando a
pergunta usa o vocabulário do título e não o do próprio artigo.

**Citações verificáveis com a API de Citations.** Cada artigo recuperado vai ao Claude como um
bloco `document` com `citations.enabled`. A resposta volta com o `cited_text` exato de cada
trecho usado, então a referência aponta para um texto literal da lei, e não para um "[1]" que o
modelo poderia ter inventado.

**Abstenção explícita.** O prompt instrui o modelo a começar com "Não encontrei" quando os
artigos recuperados não respondem à pergunta, em vez de completar com conhecimento próprio. A
avaliação mede esse comportamento com perguntas fora do escopo (direito penal, trabalhista,
tributário).

**Fallback em caso de recusa.** Para os modelos que aceitam, a chamada usa o parâmetro
server-side `fallbacks` (beta `server-side-fallback-2026-07-01`): se o modelo principal recusar a
requisição, a API refaz a chamada em outro modelo. Recusas finais são tratadas e mostradas ao
usuário.

## Avaliação

O conjunto [eval/dataset.jsonl](eval/dataset.jsonl) tem 50 perguntas escritas em linguagem leiga
("Posso pedir para uma empresa apagar meus dados?"), cada uma com os artigos relevantes anotados:
47 no escopo (LGPD e CDC) e 3 fora do escopo.

### Recuperação

Gratuita e determinística (`make eval`). Para cada pergunta, mede se algum artigo relevante
aparece entre os *k* primeiros (Hit@k), quantos dos relevantes foram encontrados (Recall@k) e a
posição do primeiro acerto (MRR).

| Embeddings | Modo | Hit@1 | Hit@3 | Hit@6 | Recall@6 | MRR |
|---|---|---|---|---|---|---|
| — | BM25 | 0.40 | 0.60 | 0.70 | 0.66 | 0.52 |
| multilingual-e5-small | híbrida | 0.49 | 0.75 | 0.81 | 0.77 | 0.62 |
| multilingual-e5-small | densa | 0.55 | 0.77 | 0.85 | 0.79 | 0.68 |
| multilingual-e5-large | híbrida | 0.62 | 0.87 | 0.92 | 0.90 | 0.74 |
| **multilingual-e5-large** | **densa** | **0.75** | **0.87** | **0.94** | **0.92** | **0.82** |

Resultados completos, incluindo as perguntas que falharam, em [eval/results/](eval/results/).

**Na CI**, o workflow [eval.yml](.github/workflows/eval.yml) roda essa avaliação em todo PR que
mexe no código, nos chunks, no dataset ou nas dependências. Ele falha se Hit@3, Hit@6, Recall@6
ou MRR do modo padrão caírem mais de 0.02 em relação ao baseline
[e5-large-dense.json](eval/results/e5-large-dense.json), e publica a tabela no resumo da
execução. Localmente, o mesmo teste é:

```bash
uv run legisrag eval --baseline eval/results/e5-large-dense.json
```

Se uma mudança melhorar os resultados de propósito, ou trocar o modelo de embeddings ou o modo
de busca, gere um novo baseline (`legisrag eval --name <nome>`) e aponte o workflow para ele.

O que os números mostraram:

- **A busca densa venceu a híbrida.** Com perguntas em linguagem leiga, o BM25 sozinho acerta
  pouco ("meu nome ficou negativado" não compartilha palavras com "cadastros e bancos de dados
  de consumidores"), e somar o ranking dele ao denso mais atrapalha do que ajuda. Por isso o
  padrão é `LEGISRAG_SEARCH_MODE=dense`. O modo híbrido continua disponível, com o peso do BM25
  configurável (`LEGISRAG_BM25_WEIGHT`), para perguntas com termos técnicos exatos.
- **O modelo de embeddings maior compensou.** Trocar o e5-small pelo e5-large subiu o MRR de
  0.68 para 0.82. O e5-large roda em CPU; a indexação dos 226 chunks leva poucos minutos na
  primeira vez (inclui o download do modelo).
- **Falhas restantes (modo denso, e5-large):** "Venda casada é proibida?" (CDC, art. 39),
  "Por quanto tempo meu nome pode ficar negativado?" (art. 43) e "O juiz pode inverter o ônus da
  prova?" (art. 6º). São perguntas curtas em que o termo popular não aparece no artigo, ou em
  que a resposta está num inciso de um artigo longo e genérico.

### Geração

`make eval-gen` roda as 50 perguntas pelo pipeline completo (chama a API do Claude, tem custo) e
mede:

- **citation_accuracy**: nas perguntas no escopo, a resposta citou literalmente um artigo
  relevante?
- **false_abstention_rate**: nas perguntas no escopo, com que frequência o sistema disse "Não
  encontrei" quando deveria responder?
- **oos_abstention_rate**: nas perguntas fora do escopo, com que frequência o sistema
  corretamente se absteve?
- tokens médios de entrada e de saída por pergunta.

Na CI ela só roda manualmente: em *Actions → Avaliação → Run workflow*, marque "Também avaliar a
geração". Requer o secret `ANTHROPIC_API_KEY` no repositório.

## Como rodar

Requer Python 3.11+ e [uv](https://docs.astral.sh/uv/).

```bash
make install                 # dependências
cp .env.example .env         # e preencha ANTHROPIC_API_KEY
make index                   # embeddings (baixa o modelo na primeira vez)
```

O arquivo `data/processed/chunks.jsonl` já vem versionado. Para regenerá-lo a partir do site do
Planalto, rode `make ingest` antes de `make index`.

```bash
uv run legisrag ask "Posso pedir para uma empresa apagar meus dados?"
uv run legisrag search "multa máxima da LGPD" --k 5   # só a busca, sem chamar o Claude
make app                     # interface web em http://localhost:8501
make api                     # API em http://localhost:8000/docs
make eval                    # avaliação de recuperação (gratuita)
make test lint               # testes e lint
```

A busca (`search`, `make eval`) funciona sem chave da API; só a geração de respostas precisa dela.

### API

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Qual o valor máximo da multa por descumprir a LGPD?", "k": 6}'
```

Resposta: `answer`, `abstained`, `citations` (fonte, artigo, trecho citado, link) e `sources`
(artigos recuperados, em ordem).

### Docker

```bash
make docker-build            # o índice é criado durante o build
make docker-run              # interface em http://localhost:8501
```

## Configuração

Todas as opções são variáveis de ambiente (veja [.env.example](.env.example) e
[config.py](src/legisrag/config.py)):

| Variável | Padrão | Descrição |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Chave da API do Claude (obrigatória para gerar respostas). |
| `LEGISRAG_CLAUDE_MODEL` | `claude-opus-5-5` | Modelo usado na geração. |
| `LEGISRAG_CLAUDE_EFFORT` | `medium` | Esforço de raciocínio: `low`, `medium` ou `high`. |
| `LEGISRAG_EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | Modelo de embeddings. Rode `make index` após trocar. |
| `LEGISRAG_SEARCH_MODE` | `dense` | `dense`, `hybrid` ou `bm25`. |
| `LEGISRAG_BM25_WEIGHT` | `0.3` | Peso do BM25 no modo híbrido (o denso tem peso 1). |
| `LEGISRAG_TOP_K` | `6` | Artigos enviados ao modelo. |

## Limitações

- Só duas leis indexadas. Leis que remetem a outras (por exemplo, o CDC ao Código Civil) podem
  gerar respostas incompletas.
- O parser foi ajustado ao HTML das páginas compiladas do Planalto; mudanças no layout do site
  podem exigir ajustes.
- O texto é o vigente na data da ingestão. Para atualizar, apague `data/raw/` e rode
  `make ingest index`.

## Licença

[MIT](LICENSE)
