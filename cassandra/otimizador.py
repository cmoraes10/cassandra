"""
Otimizador de parâmetros da Cassandra.

Os parâmetros da estratégia, como o horizonte da simulação e o limiar de
confiança, mudam bastante o resultado. Em vez de ficar no chute, o otimizador
varre combinações de valores, roda um backtest em cada uma e devolve a que se
saiu melhor segundo a métrica escolhida.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import pandas as pd

from .backtest import executa_backtest


@dataclass
class ResultadoOtimizacao:
    """Melhor combinação encontrada e a lista completa de tentativas."""

    melhores_parametros: dict
    melhor_valor: float
    melhores_metricas: dict
    tentativas: list[dict]


def otimiza(
    precos: pd.Series,
    grade: dict[str, list],
    capital_inicial: float = 10000.0,
    metrica: str = "sharpe",
    **fixos,
) -> ResultadoOtimizacao:
    """Procura na grade a combinação de parâmetros com melhor desempenho.

    A grade é um dicionário do tipo nome do parâmetro para lista de valores a
    testar. Os parâmetros fixos valem para todos os backtests. A métrica pode ser
    qualquer chave devolvida pelo backtest, como sharpe ou retorno_total.
    """
    tentativas: list[dict] = []
    melhor: dict | None = None

    for combinacao in _combinacoes(grade):
        parametros = {**fixos, **combinacao}
        resultado = executa_backtest(precos, capital_inicial=capital_inicial, **parametros)
        valor = resultado.metricas.get(metrica, float("-inf"))

        tentativa = {"parametros": combinacao, "valor": valor, "metricas": resultado.metricas}
        tentativas.append(tentativa)

        if melhor is None or valor > melhor["valor"]:
            melhor = tentativa

    if melhor is None:
        return ResultadoOtimizacao({}, float("-inf"), {}, [])

    return ResultadoOtimizacao(
        melhores_parametros=melhor["parametros"],
        melhor_valor=melhor["valor"],
        melhores_metricas=melhor["metricas"],
        tentativas=tentativas,
    )


def _combinacoes(grade: dict[str, list]):
    """Gera cada combinação possível dos valores da grade."""
    chaves = list(grade)
    for valores in product(*(grade[chave] for chave in chaves)):
        yield dict(zip(chaves, valores))
