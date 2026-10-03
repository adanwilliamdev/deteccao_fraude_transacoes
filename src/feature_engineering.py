"""
feature_engineering.py
-----------------------
Engenharia de recursos SEM vazamento de dados (data leakage).

Duas camadas com responsabilidades diferentes:

1. ``add_causal_features`` (sem estado)
   Cada linha usa APENAS informação de transações anteriores no tempo. Por ser
   causal, pode ser calculada no dataset inteiro antes do split temporal sem
   contaminar o teste com o futuro.

   - Tempo: hora do dia, seno/cosseno (continuidade 23h -> 0h), flag madrugada.
   - Valor: log do valor.
   - Velocidade por cartão: nº de transações e soma de valores nas últimas
     1h e 24h, segundos desde a transação anterior.
   - Comportamento do cartão: valor vs. média histórica do próprio cartão,
     categoria nunca vista antes naquele cartão. (Sem contador de histórico:
     seria não estacionário.)
   - Sem ``id_cartao`` (ex.: CSV do Kaggle) cai para features globais.

2. ``TabularPreprocessor`` (com estado, scikit-learn)
   Imputação pela mediana, codificação de categóricas e (opcional) escala
   robusta. É AJUSTADO SOMENTE NO TREINO e faz parte do Pipeline de cada modelo
   — corrigindo o scaler global do projeto original, que vazava estatísticas
   do teste para o treino.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler

WINDOWS = {"1h": 3600, "24h": 86400}


# --------------------------------------------------------------------------- #
# 1) Features causais (sem estado)
# --------------------------------------------------------------------------- #
def _window_stats(times: np.ndarray, amounts: np.ndarray, seconds: int):
    """Para cada linha i (ordenada por tempo): contagem e soma das transações
    ANTERIORES dentro de [t_i - seconds, t_i). Vetorizado com searchsorted."""
    left = np.searchsorted(times, times - seconds, side="left")
    idx = np.arange(len(times))
    csum = np.concatenate([[0.0], np.cumsum(amounts)])
    return (idx - left).astype(float), csum[idx] - csum[left]


def add_causal_features(
    df: pd.DataFrame,
    time_col: str = "Time",
    amount_col: str = "Amount",
    card_col: str | None = "id_cartao",
    category_col: str | None = "categoria_comerciante",
) -> pd.DataFrame:
    """Devolve o DataFrame ordenado por tempo + features causais."""
    out = df.sort_values(time_col, kind="stable").reset_index(drop=True)
    t = out[time_col].to_numpy(dtype=float)
    # Imputação CAUSAL do valor nulo: mediana expansiva apenas do passado
    # (a mediana global usaria informação do futuro).
    raw = out[amount_col].astype(float)
    past_median = raw.expanding(min_periods=1).median().shift(1).bfill()
    amount = raw.fillna(past_median).fillna(0.0).to_numpy(dtype=float)

    # --- Tempo
    sec_day = np.mod(t, 86400)
    hour = np.floor(sec_day / 3600)
    out["hora_do_dia"] = hour.astype(int)
    out["hora_sin"] = np.sin(2 * np.pi * sec_day / 86400)
    out["hora_cos"] = np.cos(2 * np.pi * sec_day / 86400)
    out["is_madrugada"] = (hour <= 5).astype(int)

    # --- Valor
    out["log_amount"] = np.log1p(amount)

    has_card = bool(card_col) and card_col in out.columns
    groups = out.groupby(card_col).indices if has_card else {"__global__": np.arange(len(out))}
    n = len(out)
    feats = {f"{p}_{w}": np.zeros(n) for p in ("cartao_qtd", "cartao_valor_soma") for w in WINDOWS}
    gap = np.zeros(n)
    prior_mean = np.ones(n)
    prior_n = np.zeros(n)

    for idx in groups.values():
        tg, ag = t[idx], amount[idx]
        for w, sec in WINDOWS.items():
            cnt, ssum = _window_stats(tg, ag, sec)
            feats[f"cartao_qtd_{w}"][idx] = cnt
            feats[f"cartao_valor_soma_{w}"][idx] = ssum
        gap[idx] = np.concatenate([[np.nan], np.diff(tg)])
        k = np.arange(len(idx))
        csum = np.concatenate([[0.0], np.cumsum(ag)])[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            pm = np.where(k > 0, csum / np.maximum(k, 1), np.nan)
        prior_mean[idx] = pm
        prior_n[idx] = k

    prefix = "cartao" if has_card else "global"
    for name, arr in feats.items():
        out[name.replace("cartao", prefix, 1)] = arr
    # Sem histórico: gap grande (30 dias) e razão neutra (1.0).
    out[f"{prefix}_seg_desde_anterior"] = np.where(np.isnan(gap), 30 * 86400.0, gap)
    out["gap_tempo_transacao_anterior"] = out[f"{prefix}_seg_desde_anterior"]
    # (O contador de histórico NÃO é exposto como feature: é monotônico no tempo, logo
    #  não estacionário — PSI > 3 entre treino e teste — e só mede 'cold start'.
    #  A razão valor/média histórica abaixo já neutraliza cartões sem histórico.)
    out[f"{prefix}_valor_medio_historico"] = np.where(np.isnan(prior_mean), amount, prior_mean)
    out["valor_vs_media_historica"] = amount / np.maximum(out[f"{prefix}_valor_medio_historico"], 1e-6)

    if has_card and category_col and category_col in out.columns:
        first_seen = out.groupby([card_col, category_col]).cumcount() == 0
        out["categoria_nova_no_cartao"] = first_seen.astype(int)

    if "gap_tempo_transacao_anterior" in out and prefix == "cartao":
        out = out.drop(columns=["gap_tempo_transacao_anterior"])
    return out


def engineer_features(df: pd.DataFrame, time_col: str = "Time", amount_col: str = "Amount") -> pd.DataFrame:
    """Atalho mantido por compatibilidade com a v1 (usa colunas padrão da v2)."""
    return add_causal_features(df, time_col=time_col, amount_col=amount_col)


# --------------------------------------------------------------------------- #
# 2) Pré-processador com estado (ajustado só no treino)
# --------------------------------------------------------------------------- #
class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """Imputação (mediana), codificação ordinal de categóricas e escala robusta
    opcional. Mantém nomes das colunas (útil para SHAP/explicações)."""

    def __init__(self, scale: bool = False):
        self.scale = scale

    def fit(self, X: pd.DataFrame, y=None):
        self.cat_cols_ = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        self.num_cols_ = [c for c in X.columns if c not in self.cat_cols_]
        num = X[self.num_cols_].astype(float)
        self.medians_ = num.median().fillna(0.0)
        self.categories_ = {c: sorted(X[c].dropna().astype(str).unique()) for c in self.cat_cols_}
        self.scaler_ = RobustScaler().fit(num.fillna(self.medians_)) if self.scale and self.num_cols_ else None
        self.feature_names_out_ = list(self.num_cols_) + list(self.cat_cols_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        num = X[self.num_cols_].astype(float).fillna(self.medians_)
        if self.scaler_ is not None:
            num = pd.DataFrame(self.scaler_.transform(num), columns=self.num_cols_, index=X.index)
        cats = {}
        for c in self.cat_cols_:
            mapping = {v: i for i, v in enumerate(self.categories_[c])}
            cats[c] = X[c].astype(str).map(mapping).fillna(-1).astype(float)
        out = pd.concat([num, pd.DataFrame(cats, index=X.index)], axis=1)
        return out[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


def scale_features(df, columns, method: str = "robust"):
    """Mantido por compatibilidade com a v1. ATENÇÃO: ajustar o scaler no
    dataset inteiro vaza informação do teste. Prefira ``TabularPreprocessor``
    dentro de um Pipeline (usado pelo main.py v2)."""
    from sklearn.preprocessing import StandardScaler
    df = df.copy()
    scaler = RobustScaler() if method == "robust" else StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler
