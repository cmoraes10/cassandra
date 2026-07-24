"""
Painel da Cassandra.

Interface web onde você escolhe o mercado e o papel, roda a simulação na hora e
vê para que lado a Cassandra acha que o preço tende a andar. Uma segunda aba
coloca a estratégia à prova sobre o passado.

Para rodar, a partir da raiz do projeto:

    streamlit run painel/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# garante que o pacote da raiz seja encontrado ao rodar pelo streamlit
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cassandra.backtest import executa_backtest
from cassandra.dados import MERCADO_BR, MERCADO_US, baixa_precos
from cassandra.simulacao import estima_modelo, simula
from cassandra.sinal import COMPRA, NEUTRO, VENDA, gera_sinal


st.set_page_config(page_title="Cassandra", page_icon=":material/query_stats:", layout="wide")
st.title("Cassandra")
st.caption(
    "Simula milhares de futuros possíveis para uma ação e mede a chance de ela "
    "subir ou cair. Ferramenta de estudo, não é recomendação de investimento."
)

with st.expander("Como usar em três passos", expanded=True):
    st.markdown(
        """
        1. Na barra lateral, escolha o **mercado** e digite o **papel** que quer analisar, como PETR4 ou AAPL.
        2. Ajuste, se quiser, o **horizonte** (quantos dias à frente projetar) e a **confiança mínima** para a Cassandra assumir uma direção.
        3. Clique em **Rodar simulação**. Ela puxa o preço real, simula milhares de cenários e mostra o resultado.

        O resultado vem em três partes. O **sinal** diz comprar, vender ou ficar de fora. A **confiança**, de 0 a 100,
        diz o quanto a projeção se afasta de um puro cara ou coroa. E o **gráfico de leque** mostra a faixa de
        preços mais provável ao longo do tempo. Na aba Backtest você testa a estratégia sobre o passado.
        """
    )

with st.sidebar:
    st.header("Configuração")
    nome_mercado = st.radio(
        "Mercado",
        ["Bolsa brasileira", "Bolsa americana"],
        help="A bolsa brasileira usa papéis como PETR4 e VALE3. A americana usa AAPL, TSLA e afins.",
    )
    mercado = MERCADO_BR if nome_mercado == "Bolsa brasileira" else MERCADO_US
    exemplo = "PETR4" if mercado == MERCADO_BR else "AAPL"
    ticker = st.text_input(
        "Papel",
        value=exemplo,
        help="O código da ação. Na bolsa brasileira não precisa do sufixo .SA, a Cassandra adiciona sozinha.",
    )
    periodo = st.selectbox(
        "Histórico usado",
        ["1y", "2y", "5y"],
        index=1,
        help="Quanto de passado a Cassandra usa para aprender o comportamento do papel. 2y são dois anos.",
    )
    horizonte = st.slider(
        "Horizonte da simulação em dias",
        5,
        60,
        21,
        help="Quantos dias de pregão à frente a simulação projeta. 21 é cerca de um mês.",
    )
    n_caminhos = st.select_slider(
        "Número de cenários",
        options=[5000, 10000, 25000, 50000],
        value=25000,
        help="Quantos futuros diferentes simular. Mais cenários deixam o resultado mais estável, e um pouco mais lento.",
    )
    limiar = st.slider(
        "Confiança mínima para operar",
        0,
        100,
        60,
        help="O quanto a simulação precisa estar convicta para sugerir uma direção. Abaixo disso, o sinal fica neutro.",
    )


@st.cache_data(show_spinner=False)
def carrega(ticker: str, mercado: str, periodo: str):
    return baixa_precos(ticker, mercado=mercado, periodo=periodo)


def grafico_leque(resultado) -> go.Figure:
    """Monta o gráfico de leque com a mediana e as faixas de probabilidade."""
    trajet = resultado.trajetorias
    dias = list(range(trajet.shape[1]))
    p05, p25, p50, p75, p95 = (np.percentile(trajet, q, axis=0) for q in (5, 25, 50, 75, 95))

    figura = go.Figure()
    figura.add_trace(
        go.Scatter(
            x=dias + dias[::-1],
            y=list(p95) + list(p05[::-1]),
            fill="toself",
            fillcolor="rgba(70,130,180,0.15)",
            line=dict(width=0),
            name="90% dos cenários",
            hoverinfo="skip",
        )
    )
    figura.add_trace(
        go.Scatter(
            x=dias + dias[::-1],
            y=list(p75) + list(p25[::-1]),
            fill="toself",
            fillcolor="rgba(70,130,180,0.30)",
            line=dict(width=0),
            name="50% dos cenários",
            hoverinfo="skip",
        )
    )
    figura.add_trace(
        go.Scatter(x=dias, y=p50, line=dict(color="#e8a838", width=2.5), name="Cenário mediano")
    )
    figura.add_hline(
        y=resultado.preco_inicial,
        line_dash="dot",
        line_color="#888",
        annotation_text="preço de hoje",
        annotation_position="bottom right",
    )
    figura.update_layout(
        title="Para onde o preço tende a caminhar",
        xaxis_title="Dias à frente",
        yaxis_title="Preço projetado",
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return figura


aba_sim, aba_backtest = st.tabs(["Simulação", "Backtest"])

with aba_sim:
    st.write("Ajuste os parâmetros na barra lateral e rode a simulação para o papel escolhido.")
    if st.button("Rodar simulação", type="primary"):
        try:
            with st.spinner(f"Buscando o histórico de {ticker.upper()}..."):
                precos = carrega(ticker, mercado, periodo)
        except Exception as erro:
            st.error(f"Não consegui carregar {ticker}. Confira o código do papel. Detalhe: {erro}")
            st.stop()

        try:
            modelo = estima_modelo(precos)
        except ValueError as erro:
            st.warning(f"Dados insuficientes para {ticker}. {erro}")
            st.stop()

        with st.spinner(f"Simulando {n_caminhos:,} cenários..."):
            resultado = simula(modelo, horizonte=horizonte, n_caminhos=n_caminhos, semente=42)
        sinal = gera_sinal(resultado, limiar_confianca=float(limiar))

        baixo, alto = resultado.intervalo(0.9)
        variacao = sinal.retorno_esperado * 100

        if sinal.direcao == COMPRA:
            st.success(
                f"**Sinal de compra** para {ticker.upper()}. Em {horizonte} dias, a maior parte dos "
                f"cenários termina acima do preço de hoje, com retorno mediano de {variacao:+.1f} por cento."
            )
        elif sinal.direcao == VENDA:
            st.error(
                f"**Sinal de venda** para {ticker.upper()}. Em {horizonte} dias, a maior parte dos "
                f"cenários termina abaixo do preço de hoje, com retorno mediano de {variacao:+.1f} por cento."
            )
        else:
            st.info(
                f"**Sinal neutro** para {ticker.upper()}. A projeção ficou perto de um cara ou coroa, "
                f"abaixo da confiança mínima de {limiar}. Nesse caso a Cassandra prefere ficar de fora."
            )

        col1, col2, col3 = st.columns(3)
        col1.metric("Sinal", sinal.direcao, help="Comprar, vender ou ficar de fora.")
        col2.metric(
            "Confiança",
            f"{sinal.confianca:.0f}/100",
            help="Quão longe de um puro cara ou coroa está a projeção. Quanto maior, mais convicta.",
        )
        col3.metric(
            "Chance de alta",
            f"{sinal.probabilidade_alta * 100:.0f}%",
            help="Fração dos cenários simulados que terminam acima do preço de hoje.",
        )

        st.plotly_chart(grafico_leque(resultado), use_container_width=True)

        with st.expander("Como ler o gráfico"):
            st.markdown(
                f"""
                A linha do meio é o **cenário mediano**, o caminho mais típico entre todos os simulados.
                A faixa mais escura concentra **metade** dos cenários e a mais clara concentra **90 por cento** deles.
                Quanto mais aberto o leque, mais incerto está o futuro daquele papel.

                Na prática, em 90 por cento dos cenários o preço de {ticker.upper()} termina entre
                **{baixo:.2f}** e **{alto:.2f}** daqui a {horizonte} dias.
                """
            )

with aba_backtest:
    st.write(
        "O backtest reconstrói o passado, gera os sinais usando só o que era conhecido em cada dia, "
        "aplica as regras de risco e mede o resultado. É a prova de que a estratégia se sustenta."
    )
    capital = st.number_input(
        "Capital inicial",
        value=10000,
        step=1000,
        help="Quanto dinheiro a estratégia começa administrando no teste.",
    )
    if st.button("Rodar backtest"):
        try:
            with st.spinner(f"Buscando o histórico de {ticker.upper()}..."):
                precos = carrega(ticker, mercado, periodo)
        except Exception as erro:
            st.error(f"Não consegui carregar {ticker}. Detalhe: {erro}")
            st.stop()

        try:
            with st.spinner("Reconstruindo o passado e testando a estratégia..."):
                resultado = executa_backtest(
                    precos,
                    capital_inicial=float(capital),
                    horizonte=horizonte,
                    n_caminhos=2000,
                    limiar_confianca=float(limiar),
                    semente=42,
                )
        except ValueError as erro:
            st.warning(f"Não dá para fazer o backtest de {ticker}. {erro}")
            st.stop()

        m = resultado.metricas
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Retorno total",
            f"{m['retorno_total'] * 100:.1f}%",
            help="Quanto o patrimônio variou do início ao fim do teste.",
        )
        c2.metric(
            "Sharpe",
            f"{m['sharpe']:.2f}",
            help="Retorno ajustado ao risco. Acima de 1 costuma ser considerado bom.",
        )
        c3.metric(
            "Maior queda",
            f"{m['max_drawdown'] * 100:.1f}%",
            help="A pior queda do topo até o fundo do patrimônio durante o teste.",
        )
        c4.metric(
            "Taxa de acerto",
            f"{m['taxa_acerto'] * 100:.0f}%",
            help="Fração das operações que fecharam no lucro.",
        )

        figura = go.Figure()
        figura.add_trace(
            go.Scatter(
                x=resultado.curva_patrimonio.index,
                y=resultado.curva_patrimonio.values,
                mode="lines",
                line=dict(color="#1a9850", width=2),
                name="Patrimônio",
            )
        )
        figura.add_hline(y=float(capital), line_dash="dot", line_color="#888", annotation_text="capital inicial")
        figura.update_layout(
            title="Evolução do patrimônio ao longo do teste",
            xaxis_title="Data",
            yaxis_title="Patrimônio",
            height=420,
        )
        st.plotly_chart(figura, use_container_width=True)
        st.caption(f"Operações realizadas no período: {m['num_operacoes']}.")

st.divider()
st.caption("Feito por Cauã Moraes · [mowaveone.com](https://mowaveone.com)")
