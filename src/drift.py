"""
drift.py
--------
Monitoramento de drift: PSI (Population Stability Index) entre treino e
dados novos. Em fraude, o comportamento muda rápido (fraudadores se adaptam);
um PSI alto indica que é hora de reavaliar/retreinar o modelo.

Regra prática: PSI < 0,10 estável | 0,10–0,25 atenção | > 0,25 mudança relevante.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected, actual, bins: int = 10, eps: float = 1e-4) -> float:
    expected = pd.Series(expected).dropna().to_numpy(float)
    actual = pd.Series(actual).dropna().to_numpy(float)
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # feature (quase) constante
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def classify_psi(value: float) -> str:
    if np.isnan(value):
        return "n/d"
    return "estável" if value < 0.10 else ("atenção" if value < 0.25 else "mudança relevante")


def drift_report(train: pd.DataFrame, new: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    cols = columns or [c for c in train.columns if pd.api.types.is_numeric_dtype(train[c]) and c in new.columns]
    rows = [{"feature": c, "psi": psi(train[c], new[c])} for c in cols]
    df = pd.DataFrame(rows)
    df["status"] = df["psi"].map(classify_psi)
    return df.sort_values("psi", ascending=False).reset_index(drop=True)
