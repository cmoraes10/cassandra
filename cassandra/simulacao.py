"""
Motor de simulação da Cassandra.

A ideia central é simular milhares de futuros possíveis para o preço de uma
ação e, a partir dessa nuvem de cenários, medir a chance de o papel subir ou
cair.

Em vez de supor que o mercado se comporta sempre do mesmo jeito, a gente
trabalha com dois regimes, um de alta e um de baixa, cada um com sua própria
tendência e volatilidade. A troca entre esses regimes segue uma cadeia de
Markov, ou seja, o humor de amanhã depende do humor de hoje. Cada trajetória
simulada é um caminho de preço construído dia a dia, sorteando o retorno de
acordo com o regime em que a simulação se encontra naquele momento.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ParametrosRegime:
    """Comportamento do preço dentro de um regime.

    A média é o retorno logarítmico diário típico do regime e o desvio é a
    volatilidade diária.
    """

    media: float
    desvio: float


@dataclass
class ModeloRegimes:
    """Tudo que o simulador precisa para gerar cenários.

    Guarda os parâmetros de cada regime, a matriz de transição entre eles, o
    regime observado mais recentemente e o último preço conhecido, que serve de
    ponto de partida.
    """

    regimes: list[ParametrosRegime]
    transicao: np.ndarray
    regime_atual: int
    preco_inicial: float


def estima_modelo(precos: pd.Series, janela_tendencia: int = 20) -> ModeloRegimes:
    """Estima os regimes a partir do histórico de preços.

    A separação entre alta e baixa é feita olhando a média móvel dos retornos.
    Quando essa média está positiva o dia é rotulado como regime de alta, quando
    está negativa vira regime de baixa. Com os rótulos em mãos a gente calcula a
    tendência e a volatilidade de cada regime e conta com que frequência o
    mercado passa de um para o outro.
    """
    precos = precos.dropna()
    minimo = janela_tendencia + 5
    if len(precos) < minimo:
        raise ValueError(
            f"histórico curto demais para estimar regimes: {len(precos)} preços, "
            f"são necessários pelo menos {minimo}."
        )

    log_ret = np.log(precos / precos.shift(1)).dropna()
    tendencia = log_ret.rolling(janela_tendencia).mean()
    rotulos = (tendencia > 0).astype(int).dropna()
    log_ret = log_ret.loc[rotulos.index]

    desvio_global = float(log_ret.std())
    if not np.isfinite(desvio_global) or desvio_global <= 0:
        # série praticamente sem variação, usa um piso pequeno para não travar a simulação
        desvio_global = 1e-6

    regimes: list[ParametrosRegime] = []
    for r in (0, 1):
        amostra = log_ret[rotulos == r]
        if len(amostra) < 2:
            # sem dados suficientes no regime, cai para a estatística global
            regimes.append(ParametrosRegime(float(log_ret.mean()), desvio_global))
        else:
            desvio = float(amostra.std())
            if not np.isfinite(desvio) or desvio <= 0:
                desvio = desvio_global
            regimes.append(ParametrosRegime(float(amostra.mean()), desvio))

    transicao = _matriz_transicao(rotulos.to_numpy())
    regime_atual = int(rotulos.iloc[-1]) if len(rotulos) else 1

    return ModeloRegimes(
        regimes=regimes,
        transicao=transicao,
        regime_atual=regime_atual,
        preco_inicial=float(precos.iloc[-1]),
    )


def _matriz_transicao(rotulos: np.ndarray) -> np.ndarray:
    """Conta as passagens de um regime para o outro e normaliza em probabilidade.

    Começa com uma contagem de um em cada casa (suavização de Laplace) para
    nunca ter probabilidade zero, o que travaria a simulação num regime só.
    """
    matriz = np.ones((2, 2))
    for atual, proximo in zip(rotulos[:-1], rotulos[1:]):
        matriz[atual, proximo] += 1
    return matriz / matriz.sum(axis=1, keepdims=True)


@dataclass
class ResultadoSimulacao:
    """Saída do simulador, com todas as trajetórias e alguns atalhos úteis."""

    trajetorias: np.ndarray
    preco_inicial: float
    horizonte: int

    @property
    def precos_finais(self) -> np.ndarray:
        """Preço no fim de cada trajetória simulada."""
        return self.trajetorias[:, -1]

    def probabilidade_alta(self) -> float:
        """Fração das trajetórias que terminam acima do preço de partida."""
        return float((self.precos_finais > self.preco_inicial).mean())

    def retorno_esperado(self) -> float:
        """Retorno da mediana dos cenários em relação ao preço de partida.

        A mediana é mais honesta que a média aqui porque não se deixa puxar por
        alguns poucos cenários extremos.
        """
        return float(np.median(self.precos_finais) / self.preco_inicial - 1)

    def intervalo(self, confianca: float = 0.9) -> tuple[float, float]:
        """Faixa de preço que concentra a maior parte dos cenários."""
        margem = (1 - confianca) / 2
        baixo = float(np.quantile(self.precos_finais, margem))
        alto = float(np.quantile(self.precos_finais, 1 - margem))
        return baixo, alto


def simula(
    modelo: ModeloRegimes,
    horizonte: int = 21,
    n_caminhos: int = 25000,
    semente: int | None = None,
) -> ResultadoSimulacao:
    """Gera as trajetórias de preço projetadas para os próximos dias.

    Cada caminho começa no último preço conhecido e avança dia a dia. Em cada
    passo o retorno é sorteado conforme o regime atual daquele caminho, o preço
    é atualizado e o regime pode mudar de acordo com a matriz de transição. O
    horizonte padrão de vinte e um dias corresponde a mais ou menos um mês de
    pregão.
    """
    rng = np.random.default_rng(semente)
    n_regimes = len(modelo.regimes)
    medias = np.array([r.media for r in modelo.regimes])
    desvios = np.array([r.desvio for r in modelo.regimes])

    trajetorias = np.empty((n_caminhos, horizonte + 1))
    trajetorias[:, 0] = modelo.preco_inicial
    regime = np.full(n_caminhos, modelo.regime_atual, dtype=int)

    # limiares acumulados para sortear o próximo regime com uma comparação só
    acumulada = np.cumsum(modelo.transicao, axis=1)

    for passo in range(1, horizonte + 1):
        choque = rng.standard_normal(n_caminhos)
        log_ret = medias[regime] + desvios[regime] * choque
        trajetorias[:, passo] = trajetorias[:, passo - 1] * np.exp(log_ret)

        sorteio = rng.random(n_caminhos)
        novo = regime.copy()
        for r in range(n_regimes):
            mascara = regime == r
            if not mascara.any():
                continue
            limiares = acumulada[r][:-1]
            novo[mascara] = (sorteio[mascara][:, None] > limiares).sum(axis=1)
        regime = novo

    return ResultadoSimulacao(trajetorias, modelo.preco_inicial, horizonte)
