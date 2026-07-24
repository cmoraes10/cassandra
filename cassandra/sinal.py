"""
Geração de sinal da Cassandra.

Depois de simular os cenários, a gente precisa transformar aquela nuvem de
preços num veredito prático, comprar, vender ou ficar de fora, junto de uma nota
de confiança que diz o quanto dá para levar aquele sinal a sério.
"""

from __future__ import annotations

from dataclasses import dataclass

from .simulacao import ResultadoSimulacao

COMPRA = "COMPRA"
VENDA = "VENDA"
NEUTRO = "NEUTRO"


@dataclass
class Sinal:
    """Decisão da estratégia para um papel num dado momento."""

    direcao: str
    confianca: float
    probabilidade_alta: float
    retorno_esperado: float
    preco_atual: float

    def resumo(self) -> str:
        """Frase curta para mostrar o sinal de forma legível."""
        return (
            f"{self.direcao} com confiança {self.confianca:.0f} de 100, "
            f"retorno projetado de {self.retorno_esperado * 100:.1f} por cento"
        )


def gera_sinal(resultado: ResultadoSimulacao, limiar_confianca: float = 60.0) -> Sinal:
    """Traduz o resultado da simulação num sinal de operação.

    A confiança nasce do quanto a probabilidade de alta se afasta do puro cara
    ou coroa. Se metade dos cenários sobe e metade cai, a confiança é zero. Se
    quase todos apontam para o mesmo lado, a confiança chega perto de cem. Só
    quando a confiança passa do limiar é que a Cassandra assume uma direção,
    caso contrário ela prefere ficar de fora.
    """
    prob = resultado.probabilidade_alta()
    retorno = resultado.retorno_esperado()
    confianca = round(abs(prob - 0.5) * 2 * 100, 1)

    if confianca < limiar_confianca or prob == 0.5:
        # confiança abaixo do limiar ou empate perfeito entre alta e baixa, fica de fora
        direcao = NEUTRO
    elif prob > 0.5:
        direcao = COMPRA
    else:
        direcao = VENDA

    return Sinal(
        direcao=direcao,
        confianca=confianca,
        probabilidade_alta=prob,
        retorno_esperado=retorno,
        preco_atual=resultado.preco_inicial,
    )
