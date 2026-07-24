"""
Cassandra, um sistema que tenta prever para que lado uma ação tende a andar.

O pacote reúne as peças da estratégia. A camada de dados busca os preços, o
motor de simulação gera os cenários futuros, o gerador de sinal traduz esses
cenários numa decisão, e os módulos de risco, posição, backtest e otimização
cuidam de operar e de validar a ideia sobre o passado.
"""

from .simulacao import estima_modelo, simula, ModeloRegimes, ResultadoSimulacao
from .sinal import gera_sinal, Sinal, COMPRA, VENDA, NEUTRO
from .risco import RegrasRisco, dimensiona
from .posicao import Carteira
from .backtest import executa_backtest, ResultadoBacktest
from .otimizador import otimiza

__all__ = [
    "estima_modelo",
    "simula",
    "ModeloRegimes",
    "ResultadoSimulacao",
    "gera_sinal",
    "Sinal",
    "COMPRA",
    "VENDA",
    "NEUTRO",
    "RegrasRisco",
    "dimensiona",
    "Carteira",
    "executa_backtest",
    "ResultadoBacktest",
    "otimiza",
]
