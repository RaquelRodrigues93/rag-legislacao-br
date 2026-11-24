from legisrag.parser import html_to_lines, parse_law
from legisrag.sources import LawSource

SOURCE = LawSource(slug="teste", name="Lei Teste", number="Lei nº 1/2020", url="http://x/lei.htm")

HTML = """
<html><body>
<p>LEI Nº 1, DE 2020</p>
<p>CAPÍTULO I</p>
<p>DISPOSIÇÕES PRELIMINARES</p>
<p><a name="art1"></a>Art. 1º Esta Lei dispõe sobre
   testes automatizados.</p>
<p>Parágrafo único. Aplica-se a todos.
   (Incluído pela Lei nº 2, de 2021)</p>
<p>Vigência</p>
<p>CAPÍTULO II</p>
<p>DOS DIREITOS</p>
<p>Seção I</p>
<p>Do Acesso</p>
<p><strike>Art. 10. Texto antigo revogado.</strike></p>
<p><span style="text-decoration: line-through">Art. 10. Outro texto antigo.</span></p>
<p>Art. 10. O titular tem direito a:</p>
<p>I - acesso;</p>
<p><strike>II - redação antiga;</strike></p>
<p>II - correção. (Redação dada pela Lei nº 3, de 2022)</p>
<p>§ 1º</p>
<p>Art. 10-A. Fica criado o órgão.</p>
<p>Art. 11. (VETADO).</p>
<p>Brasília, 1 de janeiro de 2020.</p>
<p>Art. 99. Não deve aparecer.</p>
</body></html>
"""


def test_struck_text_is_removed():
    text = "\n".join(html_to_lines(HTML))
    assert "Texto antigo" not in text
    assert "redação antiga" not in text


def test_annotations_are_removed():
    text = "\n".join(html_to_lines(HTML))
    assert "Incluído pela" not in text
    assert "Redação dada" not in text
    assert "\nVigência\n" not in text


def test_articles_and_hierarchy():
    chunks = {c.article_id: c for c in parse_law(HTML, SOURCE)}
    assert set(chunks) == {"teste-art-1", "teste-art-10", "teste-art-10-A"}

    art1 = chunks["teste-art-1"]
    assert art1.article == "Art. 1º"
    assert art1.hierarchy == "CAPÍTULO I — Disposições preliminares"
    assert art1.text == (
        "Esta Lei dispõe sobre testes automatizados.\nParágrafo único. Aplica-se a todos."
    )

    art10 = chunks["teste-art-10"]
    assert art10.article == "Art. 10"
    assert art10.hierarchy == "CAPÍTULO II — Dos direitos > Seção I — Do acesso"
    assert art10.text == "O titular tem direito a:\nI - acesso;\nII - correção."
    assert art10.url == "http://x/lei.htm#art10"

    assert chunks["teste-art-10-A"].article == "Art. 10-A"


def test_vetoed_articles_and_signatures_are_skipped():
    ids = {c.article_id for c in parse_law(HTML, SOURCE)}
    assert "teste-art-11" not in ids  # vetado
    assert "teste-art-99" not in ids  # depois das assinaturas


def test_long_articles_are_split_with_caput_context():
    numerals = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    incisos = "".join(f"<p>{n} - {'texto do inciso ' * 20};</p>" for n in numerals)
    html = f"<p>Art. 5º Considera-se:</p>{incisos}"
    chunks = parse_law(html, SOURCE)
    assert len(chunks) > 1
    assert all(c.article_id == "teste-art-5" for c in chunks)
    assert chunks[1].id == "teste-art-5#p2"
    assert chunks[1].text.startswith("Considera-se: […]")
    assert all(len(c.text) <= 1800 for c in chunks)
