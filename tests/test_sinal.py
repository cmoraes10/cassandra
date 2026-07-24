"""Testes da geração de sinal."""

from cassandra.simulacao import estima_modelo, simula
from cassandra.sinal import COMPRA, NEUTRO, VENDA, gera_sinal


def test_alta_vira_sinal_de_compra(precos_alta):
    modelo = estima_modelo(precos_alta)
    resultado = simula(modelo, horizonte=21, n_caminhos=5000, semente=3)
    sinal = gera_sinal(resultado, limiar_confianca=10.0)
    assert sinal.direcao == COMPRA


def test_baixa_vira_sinal_de_venda(precos_baixa):
    modelo = estima_modelo(precos_baixa)
    resultado = simula(modelo, horizonte=21, n_caminhos=5000, semente=3)
    sinal = gera_sinal(resultado, limiar_confianca=10.0)
    assert sinal.direcao == VENDA


def test_limiar_alto_deixa_sinal_neutro(precos_alta):
    modelo = estima_modelo(precos_alta)
    resultado = simula(modelo, horizonte=21, n_caminhos=5000, semente=3)
    sinal = gera_sinal(resultado, limiar_confianca=100.0)
    assert sinal.direcao == NEUTRO


def test_confianca_fica_entre_zero_e_cem(precos_alta):
    modelo = estima_modelo(precos_alta)
    resultado = simula(modelo, horizonte=21, n_caminhos=3000, semente=5)
    sinal = gera_sinal(resultado)
    assert 0.0 <= sinal.confianca <= 100.0
