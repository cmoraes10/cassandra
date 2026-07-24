"""
Ferramentas compartilhadas pelos testes.

Os testes não acessam a internet. Em vez de baixar preços de verdade, a gente
gera uma série sintética com tendência e ruído controlados, o que deixa os
testes rápidos e previsíveis.
"""

import numpy as np
import pandas as pd
import pytest


def serie_sintetica(n_dias=600, deriva=0.0005, volatilidade=0.02, semente=7, preco_inicial=100.0):
    """Cria uma série de preços simulando um passeio aleatório com tendência."""
    rng = np.random.default_rng(semente)
    retornos = rng.normal(deriva, volatilidade, n_dias)
    precos = preco_inicial * np.exp(np.cumsum(retornos))
    datas = pd.date_range("2020-01-01", periods=n_dias, freq="B")
    serie = pd.Series(precos, index=datas, name="TESTE")
    return serie


@pytest.fixture
def precos_alta():
    """Série com tendência de alta forte o bastante para dominar o ruído."""
    return serie_sintetica(deriva=0.003, volatilidade=0.010)


@pytest.fixture
def precos_baixa():
    """Série com tendência de baixa forte o bastante para dominar o ruído."""
    return serie_sintetica(deriva=-0.003, volatilidade=0.010)
