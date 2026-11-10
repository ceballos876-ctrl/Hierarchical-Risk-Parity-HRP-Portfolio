"""Hierarchical Risk Parity (López de Prado, 2016)."""
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform


def _ivp(cov: np.ndarray) -> np.ndarray:
    iv = 1.0 / np.diag(cov)
    return iv / iv.sum()


def _cluster_var(cov: pd.DataFrame, items: list) -> float:
    c = cov.loc[items, items].values
    w = _ivp(c).reshape(-1, 1)
    return float((w.T @ c @ w)[0, 0])


def _quasi_diag(link: np.ndarray) -> list:
    link = link.astype(int)
    order = pd.Series([link[-1, 0], link[-1, 1]])
    n = link[-1, 3]
    while order.max() >= n:
        order.index = range(0, order.shape[0] * 2, 2)
        df0 = order[order >= n]
        i = df0.index
        j = df0.values - n
        order[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        order = pd.concat([order, df1]).sort_index()
        order.index = range(order.shape[0])
    return order.tolist()


def hrp_weights(returns: pd.DataFrame, linkage_method: str = "single"):
    """Devuelve (pesos, matriz de linkage, orden de tickers quasi-diagonal)."""
    cov = returns.cov() * 252
    corr = returns.corr()
    dist = np.sqrt(np.clip((1 - corr.values) / 2, 0, None))
    np.fill_diagonal(dist, 0)
    link = sch.linkage(squareform(dist, checks=False), linkage_method)

    order = _quasi_diag(link)
    tickers = corr.index[order].tolist()

    w = pd.Series(1.0, index=tickers)
    clusters = [tickers]
    while clusters:
        clusters = [
            c[i:j]
            for c in clusters
            for i, j in ((0, len(c) // 2), (len(c) // 2, len(c)))
            if len(c) > 1
        ]
        for k in range(0, len(clusters), 2):
            c0, c1 = clusters[k], clusters[k + 1]
            v0, v1 = _cluster_var(cov, c0), _cluster_var(cov, c1)
            alpha = 1 - v0 / (v0 + v1)
            w[c0] *= alpha
            w[c1] *= 1 - alpha
    return w.sort_index(), link, tickers


def portfolio_stats(weights: pd.Series, returns: pd.DataFrame) -> dict:
    r = returns[weights.index] @ weights
    vol = r.std() * np.sqrt(252)
    ret = r.mean() * 252
    return {"Retorno anual": ret, "Volatilidad anual": vol,
            "Sharpe (rf=0)": ret / vol if vol > 0 else np.nan}


LEVELS = {"Baja": -1, "Neutral": 0, "Moderada": 1, "Alta": 2}


def tilt_weights(w: pd.Series, convictions: dict, strength: float = 1.0,
                 max_weight: float = 1.0) -> pd.Series:
    """Inclina los pesos HRP según la convicción del usuario.

    Factor por activo = (1 + strength) ** nivel, con nivel en {-1, 0, 1, 2}.
    Luego normaliza y aplica un tope por activo, redistribuyendo el exceso
    de forma proporcional entre los activos que no llegaron al tope.
    """
    n = len(w)
    max_weight = max(max_weight, 1.0 / n)
    f = pd.Series({t: (1 + strength) ** LEVELS.get(convictions.get(t, "Neutral"), 0)
                   for t in w.index})
    x = w * f
    x = x / x.sum()
    for _ in range(n):
        over = x > max_weight + 1e-12
        if not over.any():
            break
        excess = (x[over] - max_weight).sum()
        x[over] = max_weight
        free = x < max_weight - 1e-12
        if not free.any():
            break
        x[free] += excess * x[free] / x[free].sum()
    return x / x.sum()
