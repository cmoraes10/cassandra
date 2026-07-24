"""Testes da gestão de risco e da carteira."""

from cassandra.posicao import Carteira
from cassandra.risco import RegrasRisco, dimensiona


def test_dimensiona_compra_posiciona_stop_e_alvo():
    ordem = dimensiona(10000, 100, eh_compra=True, regras=RegrasRisco())
    assert ordem.capital_alocado == 1000
    assert ordem.preco_stop < 100 < ordem.preco_alvo


def test_dimensiona_venda_inverte_stop_e_alvo():
    ordem = dimensiona(10000, 100, eh_compra=False, regras=RegrasRisco())
    assert ordem.preco_alvo < 100 < ordem.preco_stop


def test_carteira_abre_e_fecha_com_lucro():
    carteira = Carteira(capital=10000)
    ordem = dimensiona(10000, 100, eh_compra=True)
    assert carteira.abre("TESTE", ordem, eh_compra=True)
    assert carteira.capital == 9000
    # preço sobe 10 por cento, a operação reservou 1000, então o lucro é 100
    resultado = carteira.fecha("TESTE", 110)
    assert round(resultado, 2) == 100.0
    assert round(carteira.capital, 2) == 10100.0


def test_carteira_venda_lucra_quando_preco_cai():
    carteira = Carteira(capital=10000)
    ordem = dimensiona(10000, 100, eh_compra=False)
    assert carteira.abre("TESTE", ordem, eh_compra=False)
    # preço cai 10 por cento, a venda reservou 1000, então o lucro é 100
    resultado = carteira.fecha("TESTE", 90)
    assert round(resultado, 2) == 100.0
    assert round(carteira.capital, 2) == 10100.0


def test_carteira_venda_perde_quando_preco_sobe():
    carteira = Carteira(capital=10000)
    ordem = dimensiona(10000, 100, eh_compra=False)
    carteira.abre("TESTE", ordem, eh_compra=False)
    # preço sobe 10 por cento, a venda perde 100
    resultado = carteira.fecha("TESTE", 110)
    assert round(resultado, 2) == -100.0


def test_venda_nao_perde_mais_que_o_capital_alocado():
    carteira = Carteira(capital=10000)
    ordem = dimensiona(10000, 100, eh_compra=False)
    carteira.abre("TESTE", ordem, eh_compra=False)
    # preço triplica; sem o piso a perda passaria de 100 por cento do alocado
    resultado = carteira.fecha("TESTE", 300)
    assert round(resultado, 2) == -1000.0


def test_dimensiona_recusa_preco_invalido():
    import pytest

    with pytest.raises(ValueError):
        dimensiona(10000, 0, eh_compra=True)


def test_carteira_recusa_papel_repetido():
    carteira = Carteira(capital=10000)
    ordem = dimensiona(10000, 100, eh_compra=True)
    assert carteira.abre("TESTE", ordem, eh_compra=True)
    assert not carteira.abre("TESTE", ordem, eh_compra=True)


def test_carteira_recusa_sem_caixa():
    carteira = Carteira(capital=500)
    ordem = dimensiona(10000, 100, eh_compra=True)
    assert not carteira.abre("TESTE", ordem, eh_compra=True)


def test_patrimonio_soma_caixa_e_posicoes():
    carteira = Carteira(capital=10000)
    ordem = dimensiona(10000, 100, eh_compra=True)
    carteira.abre("TESTE", ordem, eh_compra=True)
    # com o preço parado, o patrimônio segue igual ao capital inicial
    assert round(carteira.patrimonio({"TESTE": 100}), 2) == 10000.0
