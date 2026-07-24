"""
Backtest da Cassandra.

De nada adianta um sinal bonito que nunca foi testado. O backtest reconstrói o
passado dia a dia, gera os sinais como se a gente estivesse operando na época,
respeita as regras de risco e mede o resultado. É assim que a estratégia prova o
próprio valor antes de arriscar dinheiro de verdade.

Um cuidado importante para não trapacear com o passado, a cada ponto no tempo o
modelo só enxerga os preços até aquele dia. Nada do futuro entra na decisão.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .posicao import Carteira
from .risco import RegrasRisco, dimensiona
from .simulacao import estima_modelo, simula
from .sinal import COMPRA, NEUTRO, VENDA, gera_sinal


@dataclass
class ResultadoBacktest:
    """Curva de patrimônio, operações realizadas e métricas de desempenho."""

    curva_patrimonio: pd.Series
    operacoes: list[dict]
    metricas: dict


def executa_backtest(
    precos: pd.Series,
    capital_inicial: float = 10000.0,
    janela_treino: int = 252,
    horizonte: int = 21,
    n_caminhos: int = 2000,
    limiar_confianca: float = 60.0,
    passo: int = 5,
    regras: RegrasRisco | None = None,
    semente: int | None = 42,
) -> ResultadoBacktest:
    """Roda a estratégia sobre o histórico de um papel.

    A cada intervalo de negociação a rotina atualiza as posições abertas,
    verificando stop e alvo, e em seguida gera um novo sinal com base apenas nos
    preços passados. Quando o sinal aponta uma direção com confiança suficiente e
    há espaço na carteira, uma operação é aberta. No fim tudo é liquidado e as
    métricas são calculadas.
    """
    regras = regras or RegrasRisco()

    minimo = janela_treino + 2 * passo
    if len(precos.dropna()) < minimo:
        raise ValueError(
            f"histórico curto demais para o backtest: {len(precos.dropna())} preços, "
            f"são necessários pelo menos {minimo} para a janela e o passo escolhidos."
        )

    carteira = Carteira(capital=capital_inicial)
    ticker = precos.name or "ATIVO"
    datas = precos.index

    registros: list[tuple] = []

    for i in range(janela_treino, len(precos) - 1, passo):
        historico = precos.iloc[:i]
        preco_hoje = float(precos.iloc[i])
        data = datas[i]

        _atualiza_posicoes(carteira, {ticker: preco_hoje})

        modelo = estima_modelo(historico)
        resultado = simula(modelo, horizonte=horizonte, n_caminhos=n_caminhos, semente=semente)
        sinal = gera_sinal(resultado, limiar_confianca=limiar_confianca)

        pode_abrir = (
            sinal.direcao in (COMPRA, VENDA)
            and ticker not in carteira.posicoes
            and len(carteira.posicoes) < regras.max_posicoes
        )
        if pode_abrir:
            base = carteira.patrimonio({ticker: preco_hoje})
            ordem = dimensiona(base, preco_hoje, sinal.direcao == COMPRA, regras)
            ordem.capital_alocado = min(ordem.capital_alocado, carteira.capital)
            carteira.abre(ticker, ordem, sinal.direcao == COMPRA)

        registros.append((data, carteira.patrimonio({ticker: preco_hoje})))

    # liquida o que sobrou usando o último preço da série
    preco_final = float(precos.iloc[-1])
    for aberto in list(carteira.posicoes):
        carteira.fecha(aberto, preco_final)

    if registros:
        curva = pd.Series(dict(registros))
    else:
        curva = pd.Series(dtype=float)

    periodos_por_ano = 252 / passo
    metricas = _metricas(curva, carteira.historico, capital_inicial, periodos_por_ano)

    return ResultadoBacktest(curva_patrimonio=curva, operacoes=carteira.historico, metricas=metricas)


def _atualiza_posicoes(carteira: Carteira, precos: dict[str, float]) -> None:
    """Fecha posições que bateram no stop ou no alvo."""
    for ticker in list(carteira.posicoes):
        pos = carteira.posicoes[ticker]
        preco = precos.get(ticker)
        if preco is None:
            continue

        if pos.eh_compra:
            bateu_stop = preco <= pos.preco_stop
            bateu_alvo = preco >= pos.preco_alvo
        else:
            bateu_stop = preco >= pos.preco_stop
            bateu_alvo = preco <= pos.preco_alvo

        if bateu_stop or bateu_alvo:
            carteira.fecha(ticker, preco)


def _metricas(
    curva: pd.Series,
    operacoes: list[dict],
    capital_inicial: float,
    periodos_por_ano: float,
) -> dict:
    """Calcula retorno total, índice de Sharpe, rebaixamento e taxa de acerto."""
    if curva.empty:
        return {
            "retorno_total": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "num_operacoes": 0,
            "taxa_acerto": 0.0,
        }

    retorno_total = float(curva.iloc[-1] / capital_inicial - 1)

    variacoes = curva.pct_change().dropna()
    if len(variacoes) > 1 and variacoes.std() > 0:
        sharpe = float(variacoes.mean() / variacoes.std() * np.sqrt(periodos_por_ano))
    else:
        sharpe = 0.0

    pico = curva.cummax()
    rebaixamento = (curva - pico) / pico
    max_drawdown = float(rebaixamento.min())

    if operacoes:
        vitorias = sum(1 for op in operacoes if op["resultado"] > 0)
        taxa_acerto = vitorias / len(operacoes)
    else:
        taxa_acerto = 0.0

    return {
        "retorno_total": retorno_total,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "num_operacoes": len(operacoes),
        "taxa_acerto": taxa_acerto,
    }
