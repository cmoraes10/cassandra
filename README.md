# Cassandra

Um sistema que tenta prever para que lado uma ação tende a andar, simulando
milhares de futuros possíveis e medindo em quantos deles o preço sobe ou cai.

## De onde vem o nome

Cassandra é uma figura da mitologia grega, uma sacerdotisa que recebeu o dom de
enxergar o futuro. O nome foi escolhido de propósito. O projeto vive de olhar
para frente e apontar tendências, só que sem prometer certeza. Assim como a
Cassandra do mito, ele entrega uma leitura do que provavelmente vem, e cabe a
quem escuta decidir o que fazer com isso.

## O que ele faz

Você informa um papel, da bolsa brasileira ou da americana, e a Cassandra faz
três coisas.

Primeiro ela estuda o histórico recente do preço para entender o comportamento
daquele ativo. Depois ela projeta milhares de caminhos possíveis para os
próximos dias. Por fim ela resume tudo num sinal claro, comprar, vender ou ficar
de fora, acompanhado de uma nota de confiança de zero a cem.

Tudo isso aparece num painel web, com o gráfico das trajetórias simuladas e uma
aba de backtest que testa a estratégia sobre o passado.

## Como funciona por dentro

A ideia que sustenta o projeto é que o mercado não age sempre do mesmo jeito. Às
vezes vive um momento de alta, com tendência positiva, e às vezes um momento de
baixa. A Cassandra separa esses dois regimes a partir do histórico e mede a
tendência e a volatilidade de cada um.

A troca entre alta e baixa é tratada como uma cadeia de Markov, o que na prática
quer dizer que o humor de amanhã depende do humor de hoje. Com esses ingredientes
o motor roda uma simulação de Monte Carlo, gerando milhares de trajetórias de
preço, dia após dia, cada uma seguindo o regime em que se encontra naquele
instante.

No fim, a proporção de trajetórias que terminam acima do preço de partida vira a
probabilidade de alta, e a distância dessa probabilidade em relação ao puro cara
ou coroa vira a confiança do sinal.

## Estrutura

```
cassandra/
  dados.py         busca os preços no yfinance
  simulacao.py     estima os regimes e roda o Monte Carlo
  sinal.py         transforma a simulação num sinal de operação
  risco.py         define o tamanho da posição e os limites de perda e ganho
  posicao.py       controla o caixa e as operações abertas
  backtest.py      testa a estratégia sobre o histórico
  otimizador.py    varre parâmetros em busca da melhor configuração
painel/
  app.py           interface web em Streamlit
tests/             testes automatizados
```

## Como rodar

O projeto foi testado com Python 3.12. Instale as dependências e abra o painel.

```bash
pip install -r requirements.txt
streamlit run painel/app.py
```

Para rodar os testes, instale também as dependências de desenvolvimento.

```bash
pip install -r requirements-dev.txt
pytest
```

## Como colocar no ar

O painel roda de graça no Streamlit Community Cloud. Suba o repositório para o
GitHub, entre em share.streamlit.io, aponte para este repositório e indique
`painel/app.py` como arquivo principal. As dependências saem do `requirements.txt`
e o restante é automático.

## Exemplo rápido pelo código

```python
from cassandra.dados import baixa_precos
from cassandra.simulacao import estima_modelo, simula
from cassandra.sinal import gera_sinal

precos = baixa_precos("PETR4", mercado="B3")
modelo = estima_modelo(precos)
resultado = simula(modelo, horizonte=21, n_caminhos=25000)
sinal = gera_sinal(resultado)
print(sinal.resumo())
```

## Aviso

Este é um projeto de estudo, feito para explorar simulação de Monte Carlo,
cadeias de Markov e teste de estratégias. Nada aqui é recomendação de
investimento. Mercado envolve risco e resultado passado não garante resultado
futuro.

## Autor

Feito por Cauã Moraes. Conheça outros projetos em [mowaveone.com](https://mowaveone.com).
