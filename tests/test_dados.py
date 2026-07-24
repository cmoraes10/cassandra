"""Testes da camada de dados, sem tocar na rede."""

import pandas as pd
import pytest

from cassandra import dados
from cassandra.dados import MERCADO_BR, MERCADO_US, normaliza_ticker, retornos_diarios


def test_ticker_br_ganha_sufixo():
    assert normaliza_ticker("petr4", MERCADO_BR) == "PETR4.SA"
    # não duplica o sufixo se já vier com ele
    assert normaliza_ticker("PETR4.SA", MERCADO_BR) == "PETR4.SA"


def test_ticker_us_fica_puro():
    assert normaliza_ticker(" aapl ", MERCADO_US) == "AAPL"


def test_retornos_diarios():
    precos = pd.Series([100.0, 110.0, 99.0])
    ret = retornos_diarios(precos)
    assert round(ret.iloc[0], 2) == 0.10
    assert round(ret.iloc[1], 2) == -0.10


def test_baixa_precos_trata_dataframe_de_uma_coluna(monkeypatch):
    # simula o yfinance devolvendo o Close como DataFrame de uma coluna
    datas = pd.date_range("2020-01-01", periods=3, freq="B")
    falso = pd.DataFrame({"Close": [10.0, 11.0, 12.0]}, index=datas)
    falso.columns = pd.MultiIndex.from_tuples([("Close", "XPTO")])

    monkeypatch.setattr(dados.yf, "download", lambda *a, **k: falso)
    serie = dados.baixa_precos("XPTO", mercado=MERCADO_US)
    assert list(serie) == [10.0, 11.0, 12.0]


def test_baixa_precos_erro_quando_vazio(monkeypatch):
    monkeypatch.setattr(dados.yf, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(ValueError):
        dados.baixa_precos("NADA", mercado=MERCADO_US)
