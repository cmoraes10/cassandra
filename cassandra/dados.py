"""
Camada de dados da Cassandra.

Aqui a gente busca o histórico de preços de uma ação usando o yfinance. A bolsa
brasileira exige o sufixo .SA no código do papel, então esse detalhe fica
escondido neste módulo e o resto do sistema não precisa se preocupar com isso.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

MERCADO_BR = "B3"
MERCADO_US = "US"


def normaliza_ticker(ticker: str, mercado: str) -> str:
    """Ajusta o código do papel ao formato que o yfinance espera.

    Na B3 os papéis precisam terminar em .SA, como PETR4.SA. Na bolsa americana
    o código vai puro, como AAPL.
    """
    ticker = ticker.strip().upper()
    if mercado == MERCADO_BR and not ticker.endswith(".SA"):
        return f"{ticker}.SA"
    return ticker


def baixa_precos(
    ticker: str,
    mercado: str = MERCADO_US,
    periodo: str = "2y",
    intervalo: str = "1d",
) -> pd.Series:
    """Baixa a série de preços de fechamento ajustado de um papel.

    Devolve uma série do pandas indexada por data. Levanta erro se o papel não
    existir ou se o período não trouxer dado nenhum.
    """
    simbolo = normaliza_ticker(ticker, mercado)
    bruto = yf.download(
        simbolo,
        period=periodo,
        interval=intervalo,
        auto_adjust=True,
        progress=False,
    )
    if bruto is None or bruto.empty:
        raise ValueError(f"Nenhum dado encontrado para {simbolo}.")

    fechamento = bruto["Close"]
    # dependendo da versão do yfinance o Close pode vir como DataFrame de uma coluna
    if isinstance(fechamento, pd.DataFrame):
        fechamento = fechamento.iloc[:, 0]

    serie = fechamento.dropna()
    if serie.empty:
        raise ValueError(f"Série de {simbolo} ficou vazia depois da limpeza.")

    serie.name = simbolo
    return serie


def retornos_diarios(precos: pd.Series) -> pd.Series:
    """Retorno percentual de um dia para o outro."""
    return precos.pct_change().dropna()
