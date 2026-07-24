"""
Gestão de risco da Cassandra.

Um sinal bonito não vale nada sem controle de risco. Aqui a gente decide quanto
do capital entra em cada operação e onde ficam os limites de perda e de ganho
que tiram a gente da posição antes de a coisa desandar ou depois de o lucro já
ter valido a pena.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RegrasRisco:
    """Parâmetros que governam o tamanho e os limites de cada operação.

    A fração por trade é a parte do patrimônio que vai para uma única operação.
    O stop e o alvo são as variações de preço que encerram a posição, uma no
    prejuízo e outra no lucro. O máximo de posições evita concentrar tudo de uma
    vez.
    """

    fracao_por_trade: float = 0.10
    stop_loss: float = 0.35
    take_profit: float = 0.80
    max_posicoes: int = 4


@dataclass
class Ordem:
    """Uma operação já dimensionada, pronta para entrar na carteira."""

    capital_alocado: float
    preco_entrada: float
    preco_stop: float
    preco_alvo: float


def dimensiona(
    capital_base: float,
    preco: float,
    eh_compra: bool,
    regras: RegrasRisco | None = None,
) -> Ordem:
    """Monta a ordem a partir do capital disponível e do preço atual.

    Calcula quanto dinheiro vai para a operação e projeta os preços de stop e de
    alvo. Numa compra o stop fica abaixo e o alvo acima do preço de entrada.
    Numa venda a lógica se inverte.
    """
    if preco <= 0:
        raise ValueError("preço precisa ser positivo para dimensionar a ordem.")

    regras = regras or RegrasRisco()
    capital_alocado = capital_base * regras.fracao_por_trade

    if eh_compra:
        preco_stop = preco * (1 - regras.stop_loss)
        preco_alvo = preco * (1 + regras.take_profit)
    else:
        preco_stop = preco * (1 + regras.stop_loss)
        preco_alvo = preco * (1 - regras.take_profit)

    return Ordem(
        capital_alocado=capital_alocado,
        preco_entrada=preco,
        preco_stop=preco_stop,
        preco_alvo=preco_alvo,
    )
