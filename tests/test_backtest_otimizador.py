"""Testes do backtest e do otimizador."""

from cassandra.backtest import executa_backtest
from cassandra.otimizador import otimiza


def test_backtest_roda_e_devolve_metricas(precos_alta):
    resultado = executa_backtest(
        precos_alta,
        capital_inicial=10000,
        janela_treino=252,
        n_caminhos=300,
        passo=10,
        semente=1,
    )
    assert not resultado.curva_patrimonio.empty
    for chave in ("retorno_total", "sharpe", "max_drawdown", "num_operacoes", "taxa_acerto"):
        assert chave in resultado.metricas


def test_backtest_nao_usa_futuro(precos_alta):
    # rodar duas vezes com a mesma semente tem que dar exatamente o mesmo resultado
    a = executa_backtest(precos_alta, n_caminhos=300, passo=10, semente=1)
    b = executa_backtest(precos_alta, n_caminhos=300, passo=10, semente=1)
    assert a.metricas == b.metricas


def test_drawdown_nunca_e_positivo(precos_baixa):
    resultado = executa_backtest(precos_baixa, n_caminhos=300, passo=10, semente=1)
    assert resultado.metricas["max_drawdown"] <= 0


def test_otimizador_encontra_melhor_combinacao(precos_alta):
    grade = {"limiar_confianca": [40.0, 70.0], "horizonte": [10, 21]}
    resultado = otimiza(
        precos_alta,
        grade,
        metrica="sharpe",
        n_caminhos=200,
        passo=15,
        semente=1,
    )
    assert resultado.melhores_parametros
    assert len(resultado.tentativas) == 4
