"""Testes do motor de simulação e da estimativa de regimes."""

import numpy as np
import pandas as pd
import pytest

from cassandra.simulacao import estima_modelo, simula


def test_serie_curta_levanta_erro():
    curta = pd.Series([100.0, 101.0, 99.0], name="CURTA")
    with pytest.raises(ValueError):
        estima_modelo(curta)


def test_modelo_tem_dois_regimes_e_transicao_valida(precos_alta):
    modelo = estima_modelo(precos_alta)
    assert len(modelo.regimes) == 2
    # cada linha da matriz de transição é uma distribuição de probabilidade
    somas = modelo.transicao.sum(axis=1)
    assert np.allclose(somas, 1.0)
    assert modelo.preco_inicial > 0


def test_simulacao_gera_formato_correto(precos_alta):
    modelo = estima_modelo(precos_alta)
    resultado = simula(modelo, horizonte=10, n_caminhos=500, semente=1)
    assert resultado.trajetorias.shape == (500, 11)
    # toda trajetória começa no preço inicial
    assert np.allclose(resultado.trajetorias[:, 0], modelo.preco_inicial)


def test_semente_deixa_resultado_reproduzivel(precos_alta):
    modelo = estima_modelo(precos_alta)
    a = simula(modelo, horizonte=10, n_caminhos=300, semente=99)
    b = simula(modelo, horizonte=10, n_caminhos=300, semente=99)
    assert np.array_equal(a.trajetorias, b.trajetorias)


def test_tendencia_de_alta_puxa_probabilidade_para_cima(precos_alta):
    modelo = estima_modelo(precos_alta)
    resultado = simula(modelo, horizonte=21, n_caminhos=5000, semente=3)
    assert resultado.probabilidade_alta() > 0.5


def test_intervalo_faz_sentido(precos_alta):
    modelo = estima_modelo(precos_alta)
    resultado = simula(modelo, horizonte=21, n_caminhos=5000, semente=3)
    baixo, alto = resultado.intervalo(0.9)
    assert baixo < alto
