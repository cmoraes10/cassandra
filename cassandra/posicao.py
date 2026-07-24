"""
Gestão de posição e carteira da Cassandra.

Este módulo é a memória financeira do sistema. Ele controla o dinheiro livre, as
operações abertas e o resultado de cada operação encerrada, tanto durante um
backtest quanto numa simulação de operação em tempo real.

Para manter as contas simples e honestas, cada operação reserva um valor em
dinheiro na entrada. Na saída esse valor volta para o caixa somado ou subtraído
do resultado, calculado pela variação do preço no sentido da aposta.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .risco import Ordem


@dataclass
class Posicao:
    """Uma operação aberta na carteira."""

    ticker: str
    capital: float
    preco_entrada: float
    eh_compra: bool
    preco_stop: float
    preco_alvo: float


@dataclass
class Carteira:
    """Caixa, posições abertas e histórico de operações encerradas."""

    capital: float
    posicoes: dict[str, Posicao] = field(default_factory=dict)
    historico: list[dict] = field(default_factory=list)

    def abre(self, ticker: str, ordem: Ordem, eh_compra: bool) -> bool:
        """Abre uma posição se houver caixa e o papel ainda não estiver em carteira.

        Devolve verdadeiro quando a posição é criada e falso quando a operação é
        recusada por falta de dinheiro ou por já existir posição naquele papel.
        """
        if ticker in self.posicoes or ordem.capital_alocado <= 0:
            return False
        if ordem.capital_alocado > self.capital + 1e-9:
            return False

        self.capital -= ordem.capital_alocado
        self.posicoes[ticker] = Posicao(
            ticker=ticker,
            capital=ordem.capital_alocado,
            preco_entrada=ordem.preco_entrada,
            eh_compra=eh_compra,
            preco_stop=ordem.preco_stop,
            preco_alvo=ordem.preco_alvo,
        )
        return True

    def fecha(self, ticker: str, preco: float) -> float:
        """Encerra a posição e devolve o resultado financeiro da operação."""
        pos = self.posicoes.pop(ticker, None)
        if pos is None:
            return 0.0

        variacao = preco / pos.preco_entrada - 1
        if not pos.eh_compra:
            # numa venda o ganho vem da queda; a perda é limitada ao capital alocado
            variacao = max(-variacao, -1.0)

        resultado = pos.capital * variacao
        self.capital += pos.capital + resultado
        self.historico.append(
            {
                "ticker": ticker,
                "retorno": variacao,
                "resultado": resultado,
                "eh_compra": pos.eh_compra,
            }
        )
        return resultado

    def patrimonio(self, precos: dict[str, float]) -> float:
        """Soma o caixa livre com o valor corrente das posições abertas."""
        total = self.capital
        for ticker, pos in self.posicoes.items():
            preco = precos.get(ticker, pos.preco_entrada)
            variacao = preco / pos.preco_entrada - 1
            if not pos.eh_compra:
                variacao = -variacao
            total += pos.capital * (1 + variacao)
        return total
