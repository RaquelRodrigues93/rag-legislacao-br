"""Leis indexadas. Para adicionar uma nova, basta incluir uma entrada aqui."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LawSource:
    slug: str  # identificador curto usado nos ids dos chunks
    name: str  # nome de exibição
    number: str
    url: str


SOURCES: list[LawSource] = [
    LawSource(
        slug="lgpd",
        name="LGPD",
        number="Lei nº 13.709/2018",
        url="http://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709compilado.htm",
    ),
    LawSource(
        slug="cdc",
        name="Código de Defesa do Consumidor",
        number="Lei nº 8.078/1990",
        url="http://www.planalto.gov.br/ccivil_03/leis/l8078compilado.htm",
    ),
]
