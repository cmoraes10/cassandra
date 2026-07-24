"""
Painel da Cassandra.

Interface web onde você escolhe o mercado e o papel, roda a simulação na hora e
vê para que lado a Cassandra acha que o preço tende a andar. Uma segunda aba
coloca a estratégia à prova sobre o histórico.

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

CORES = {COMPRA: "#1a9850", VENDA: "#d73027", NEUTRO: "#888888"}


st.set_page_config(page_title="Cassandra", page_icon=":material/query_stats:", layout="wide")
st.title("Cassandra")
st.caption(
    "Simula milhares de futuros possíveis para uma ação e mede a chance de ela "
    "subir ou cair. Ferramenta de estudo, não é recomendação de investimento."
)

with st.sidebar:
    st.header("Configuração")
    nome_mercado = st.radio("Mercado", ["Bolsa brasileira", "Bolsa americana"])
    mercado = MERCADO_BR if nome_mercado == "Bolsa brasileira" else MERCADO_US
    exemplo = "PETR4" if mercado == MERCADO_BR else "AAPL"
    ticker = st.text_input("Papel", value=exemplo)
    periodo = st.selectbox("Histórico usado", ["1y", "2y", "5y"], index=1)
    horizonte = st.slider("Horizonte da simulação em dias", 5, 60, 21)
    n_caminhos = st.select_slider(
        "Número de cenários", options=[5000, 10000, 25000, 50000], value=25000
    )
    limiar = st.slider("Confiança mínima para operar", 0, 100, 60)


@st.cache_data(show_spinner=False)
def carrega(ticker: str, mercado: str, periodo: str):
    return baixa_precos(ticker, mercado=mercado, periodo=periodo)


aba_sim, aba_backtest = st.tabs(["Simulação", "Backtest"])

with aba_sim:
    if st.button("Rodar simulação", type="primary"):
        try:
            precos = carrega(ticker, mercado, periodo)
        except Exception as erro:
            st.error(f"Não consegui carregar {ticker}. {erro}")
        else:
            try:
                modelo = estima_modelo(precos)
            except ValueError as erro:
                st.warning(f"Dados insuficientes para {ticker}. {erro}")
                st.stop()
            resultado = simula(modelo, horizonte=horizonte, n_caminhos=n_caminhos, semente=42)
            sinal = gera_sinal(resultado, limiar_confianca=float(limiar))

            col1, col2, col3 = st.columns(3)
            col1.metric("Sinal", sinal.direcao)
            col2.metric("Confiança", f"{sinal.confianca:.0f}/100")
            col3.metric("Retorno projetado", f"{sinal.retorno_esperado * 100:.1f}%")

            st.markdown(
                f"<h4 style='color:{CORES[sinal.direcao]}'>{sinal.resumo()}</h4>",
                unsafe_allow_html=True,
            )

            # leque de trajetórias, mostrando uma amostra para não pesar o gráfico
            amostra = resultado.trajetorias[:: max(1, n_caminhos // 200)]
            figura = go.Figure()
            for caminho in amostra:
                figura.add_trace(
                    go.Scatter(
                        y=caminho,
                        mode="lines",
                        line=dict(width=0.5, color="rgba(70,130,180,0.15)"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            mediana = np.median(resultado.trajetorias, axis=0)
            figura.add_trace(
                go.Scatter(y=mediana, mode="lines", line=dict(width=3, color="#222"), name="Mediana")
            )
            figura.update_layout(
                title="Trajetórias simuladas de preço",
                xaxis_title="Dias à frente",
                yaxis_title="Preço",
                height=460,
            )
            st.plotly_chart(figura, use_container_width=True)

            baixo, alto = resultado.intervalo(0.9)
            st.info(
                f"Em 90 por cento dos cenários o preço termina entre "
                f"{baixo:.2f} e {alto:.2f}."
            )

with aba_backtest:
    st.write(
        "O backtest reconstrói o passado, gera os sinais só com o que era "
        "conhecido em cada dia e mede o resultado da estratégia."
    )
    capital = st.number_input("Capital inicial", value=10000, step=1000)
    if st.button("Rodar backtest"):
        try:
            precos = carrega(ticker, mercado, periodo)
        except Exception as erro:
            st.error(f"Não consegui carregar {ticker}. {erro}")
        else:
            try:
                with st.spinner("Reconstruindo o passado..."):
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
            c1.metric("Retorno total", f"{m['retorno_total'] * 100:.1f}%")
            c2.metric("Sharpe", f"{m['sharpe']:.2f}")
            c3.metric("Maior queda", f"{m['max_drawdown'] * 100:.1f}%")
            c4.metric("Taxa de acerto", f"{m['taxa_acerto'] * 100:.0f}%")

            figura = go.Figure()
            figura.add_trace(
                go.Scatter(
                    x=resultado.curva_patrimonio.index,
                    y=resultado.curva_patrimonio.values,
                    mode="lines",
                    line=dict(color="#1a9850", width=2),
                )
            )
            figura.update_layout(
                title="Evolução do patrimônio",
                xaxis_title="Data",
                yaxis_title="Patrimônio",
                height=420,
            )
            st.plotly_chart(figura, use_container_width=True)
            st.caption(f"Operações realizadas: {m['num_operacoes']}")

st.divider()
st.caption("Feito por Cauã Moraes · [mowaveone.com](https://mowaveone.com)")
