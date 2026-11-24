"""Transforma o HTML de uma lei do Planalto em chunks por artigo.

Decisões de design:
- **Unidade de chunk = artigo.** É a unidade que juristas citam e que o usuário espera ver
  como fonte. Chunks de tamanho fixo cortam incisos no meio e misturam artigos.
- **Texto revogado é descartado.** O Planalto mantém redações antigas riscadas (<strike> ou
  `text-decoration: line-through`) logo antes da redação vigente; sem esse filtro o modelo
  responde com a lei desatualizada.
- **Hierarquia vira contexto.** "Capítulo II — Do Tratamento de Dados Pessoais > Seção II — Do
  Tratamento de Dados Pessoais Sensíveis" entra no texto indexado, o que ajuda muito a busca
  quando a pergunta usa o vocabulário do título e não do artigo.
- **Artigos longos são divididos** em partes, repetindo o caput em cada parte para que nenhuma
  delas perca o sentido.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from bs4 import BeautifulSoup, Tag

from legisrag.sources import LawSource

MAX_CHUNK_CHARS = 1800
_SEP = "\x00"

_BLOCK_TAGS = ["p", "br", "div", "tr", "li", "h1", "h2", "h3", "h4", "table"]

# Anotações editoriais do Planalto, ex.: "(Redação dada pela Lei nº 13.853, de 2019)".
_ANNOTATION_RE = re.compile(
    r"\(\s*(?:Redação dada|Incluíd[oa]|Acrescentad[oa]|Renumerad[oa]|Vide|Regulamento|"
    r"Vigência|Produção de efeitos?|Convertid[oa]|Revogad[oa] pel[oa]|Promulgação|"
    r"Mensagem de veto)[^()]*(?:\([^()]*\)[^()]*)*\)",
    re.IGNORECASE,
)
_NOISE_LINE_RE = re.compile(r"^(Vigência|Regulamento|Texto compilado|Mensagem de veto|\*)$", re.I)

_ARTICLE_RE = re.compile(
    r"^Art\.\s*(?P<num>\d+(?:\.\d{3})?)\s*(?:[º°o]\s*)?(?:-\s*(?P<suf>[A-Z]))?\s*[.\-–]?\s*(?P<body>.*)$"
)
_HEADING_RE = re.compile(r"^(?P<level>TÍTULO|CAPÍTULO|SEÇÃO|Seção|SUBSEÇÃO|Subseção|LIVRO)\s+\S+")
_LEVELS = ["LIVRO", "TÍTULO", "CAPÍTULO", "SEÇÃO", "SUBSEÇÃO"]
_EMPTY_BODY_RE = re.compile(r"^\(?\s*(VETADO|Revogado|revogado)\s*\)?\.?$")
# Dispositivo sem conteúdo (sobra de parágrafo vetado/revogado cujo texto estava riscado).
_BARE_MARKER_RE = re.compile(r"^(§\s*\d+[º°]?|[IVXLC]+\s*[-–]|[a-z]\)|Parágrafo único\.?)$")


@dataclass
class Chunk:
    id: str
    article_id: str
    law_slug: str
    law_name: str
    law_number: str
    article: str
    hierarchy: str
    text: str
    url: str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def citation_label(self) -> str:
        return f"{self.law_name}, {self.article}"

    @property
    def index_text(self) -> str:
        """Texto usado na indexação: inclui lei, hierarquia e artigo como contexto."""
        header = f"{self.law_name} ({self.law_number})"
        if self.hierarchy:
            header += f" — {self.hierarchy}"
        return f"{header}\n{self.article}. {self.text}"


def decode_html(raw: bytes) -> str:
    """Páginas do Planalto costumam vir em windows-1252; algumas em UTF-8."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("windows-1252", errors="replace")


def _is_struck(tag: Tag) -> bool:
    if tag.name == "strike":
        return True
    style = (tag.get("style") or "").replace(" ", "").lower()
    return "line-through" in style


def html_to_lines(html: str) -> list[str]:
    """Extrai o texto vigente da página, uma linha por bloco (parágrafo)."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    for tag in soup.find_all(_is_struck):
        tag.decompose()
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.insert_before(_SEP)
        tag.insert_after(_SEP)

    lines = []
    for raw_line in soup.get_text().split(_SEP):
        line = re.sub(r"\s+", " ", raw_line).strip()
        line = _ANNOTATION_RE.sub("", line)
        line = re.sub(r"\s+([,.;:])", r"\1", line)
        line = re.sub(r"\s{2,}", " ", line).strip()
        if line and not _NOISE_LINE_RE.match(line):
            lines.append(line)
    return _merge_broken_lines(lines)


def _merge_broken_lines(lines: list[str]) -> list[str]:
    """Junta fragmentos que o HTML separou no meio da frase (ex.: um link no meio do texto)."""
    merged: list[str] = []
    for line in lines:
        starts_new = (
            _ARTICLE_RE.match(line)
            or _HEADING_RE.match(line)
            or re.match(r"^(§|Parágrafo único|[IVXLC]+\s*[-–]|[a-z]\)|\d+\))", line)
        )
        if merged and not starts_new and (line[0] in ",;.)" or line[0].islower()):
            merged[-1] = f"{merged[-1]} {line}".replace(" ,", ",").replace(" ;", ";")
        else:
            merged.append(line)
    return merged


def _article_label(num: str, suffix: str | None) -> str:
    n = int(num.replace(".", ""))
    label = f"Art. {num}º" if n < 10 else f"Art. {num}"
    return f"{label}-{suffix}" if suffix else label


def _article_id(slug: str, num: str, suffix: str | None) -> str:
    return f"{slug}-art-{num.replace('.', '')}" + (f"-{suffix}" if suffix else "")


def _level_of(line: str) -> str:
    return line.split()[0].upper().replace("SECAO", "SEÇÃO")


def parse_law(html: str, source: LawSource) -> list[Chunk]:
    lines = html_to_lines(html)
    hierarchy: dict[str, str] = {}
    articles: list[dict] = []
    current: dict | None = None
    pending_heading: str | None = None

    for line in lines:
        if line.startswith("Brasília"):
            break  # fim do texto normativo (assinaturas)

        if _HEADING_RE.match(line):
            level = _level_of(line)
            for lower in _LEVELS[_LEVELS.index(level) :]:
                hierarchy.pop(lower, None)
            hierarchy[level] = line
            pending_heading = level
            continue

        if pending_heading and not _ARTICLE_RE.match(line):
            # Linha logo após "CAPÍTULO II" é o título do capítulo.
            title = line.rstrip("’'\" ").capitalize()
            hierarchy[pending_heading] = f"{hierarchy[pending_heading]} — {title}"
            pending_heading = None
            continue
        pending_heading = None

        m = _ARTICLE_RE.match(line)
        if m:
            num, suf, body = m.group("num"), m.group("suf"), m.group("body").strip()
            art_id = _article_id(source.slug, num, suf)
            current = next((a for a in articles if a["id"] == art_id), None)
            if current is None:
                current = {
                    "id": art_id,
                    "label": _article_label(num, suf),
                    "anchor": f"art{num.replace('.', '')}{suf.lower() if suf else ''}",
                    "hierarchy": " > ".join(hierarchy[lv] for lv in _LEVELS if lv in hierarchy),
                    "lines": [],
                }
                articles.append(current)
            if body:
                current["lines"].append(body)
        elif current is not None and not _BARE_MARKER_RE.match(line):
            current["lines"].append(line)

    chunks: list[Chunk] = []
    for art in articles:
        text_lines = art["lines"]
        if not text_lines or (len(text_lines) == 1 and _EMPTY_BODY_RE.match(text_lines[0])):
            continue  # artigos vetados ou revogados por inteiro
        for i, part in enumerate(_split_article(text_lines)):
            chunks.append(
                Chunk(
                    id=art["id"] if i == 0 else f"{art['id']}#p{i + 1}",
                    article_id=art["id"],
                    law_slug=source.slug,
                    law_name=source.name,
                    law_number=source.number,
                    article=art["label"],
                    hierarchy=art["hierarchy"],
                    text=part,
                    url=f"{source.url}#{art['anchor']}",
                )
            )
    return chunks


def _split_article(lines: list[str]) -> list[str]:
    full = "\n".join(lines)
    if len(full) <= MAX_CHUNK_CHARS:
        return [full]

    caput = lines[0]
    caput_ctx = caput if len(caput) <= 300 else caput[:300].rsplit(" ", 1)[0] + "…"
    parts: list[list[str]] = [[caput]]
    size = len(caput)
    for line in lines[1:]:
        if size + len(line) > MAX_CHUNK_CHARS and len(parts[-1]) > 1:
            parts.append([f"{caput_ctx} […]"])
            size = len(caput_ctx)
        parts[-1].append(line)
        size += len(line) + 1
    return ["\n".join(p) for p in parts]
