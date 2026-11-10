import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

from hrp import hrp_weights, portfolio_stats

st.set_page_config(page_title="Portafolio HRP", page_icon="📊", layout="centered")
st.title("📊 Constructor de portafolios HRP")
st.caption("Hierarchical Risk Parity con datos de Yahoo Finance (gratis).")

with st.expander("⚙️ Configuración", expanded=True):
    raw = st.text_area("Tickers (separados por coma o espacio)",
                       "AAPL, MSFT, JNJ, XOM, JPM, GLD, TLT, VNQ")
    period = st.selectbox("Histórico", ["1y", "2y", "3y", "5y", "10y"], index=2)
    method = st.selectbox("Linkage", ["single", "ward", "average", "complete"])
    capital = st.number_input("Capital a invertir", min_value=0.0,
                              value=10000.0, step=1000.0)
    run = st.button("Calcular portafolio", type="primary")


@st.cache_data(show_spinner=False, ttl=3600)
def load_prices(tickers: tuple, period: str) -> pd.DataFrame:
    df = yf.download(list(tickers), period=period, auto_adjust=True,
                     progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    return df


if run:
    tickers = tuple(dict.fromkeys(
        t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()))
    if len(tickers) < 2:
        st.error("Ingresa al menos 2 tickers.")
        st.stop()

    with st.spinner("Descargando datos de Yahoo Finance..."):
        prices = load_prices(tickers, period)

    missing = [t for t in tickers if t not in prices.columns
               or prices[t].dropna().empty]
    if missing:
        st.warning(f"Sin datos para: {', '.join(missing)} (se excluyen).")
    prices = prices.drop(columns=missing, errors="ignore")
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.8)).dropna()
    if prices.shape[1] < 2 or len(prices) < 60:
        st.error("No hay suficientes datos comunes para calcular.")
        st.stop()

    returns = prices.pct_change().dropna()
    w, link, order = hrp_weights(returns, method)
    ew = pd.Series(1 / len(w), index=w.index)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Pesos HRP")
        table = pd.DataFrame({
            "Peso": w,
            "Monto": w * capital,
            "Último precio": prices.iloc[-1][w.index],
        })
        table["Acciones aprox."] = (table["Monto"] / table["Último precio"]).round(2)
        st.dataframe(table.sort_values("Peso", ascending=False).style.format(
            {"Peso": "{:.2%}", "Monto": "${:,.0f}",
             "Último precio": "${:,.2f}", "Acciones aprox.": "{:,.2f}"}),
            use_container_width=True)
        st.download_button("Descargar CSV", table.to_csv().encode(),
                           "portafolio_hrp.csv", "text/csv")
    with c2:
        st.subheader("Dendrograma")
        fig, ax = plt.subplots(figsize=(6, 4))
        sch.dendrogram(link, labels=list(returns.columns), ax=ax)
        ax.set_ylabel("Distancia")
        st.pyplot(fig)

    st.subheader("HRP vs. equiponderado (histórico, sin rebalanceo)")
    stats = pd.DataFrame({"HRP": portfolio_stats(w, returns),
                          "Equiponderado": portfolio_stats(ew, returns)})
    st.dataframe(stats.style.format("{:.2%}", subset=pd.IndexSlice[
        ["Retorno anual", "Volatilidad anual"], :]).format(
        "{:.2f}", subset=pd.IndexSlice[["Sharpe (rf=0)"], :]))

    cum = pd.DataFrame({"HRP": (1 + returns[w.index] @ w).cumprod(),
                        "Equiponderado": (1 + returns[w.index] @ ew).cumprod()})
    st.line_chart(cum)

    st.subheader("Matriz de correlación")
    corr = returns[order].corr()
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im = ax2.imshow(corr, cmap="RdYlGn_r", vmin=-1, vmax=1)
    ax2.set_xticks(range(len(order)), order, rotation=90)
    ax2.set_yticks(range(len(order)), order)
    fig2.colorbar(im)
    st.pyplot(fig2)

    st.caption("Solo con fines educativos; no es asesoría financiera. "
               "Los resultados históricos no garantizan rendimientos futuros.")
else:
    st.info("Ingresa tus tickers en la configuración y presiona **Calcular portafolio**.")
