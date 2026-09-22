"""
feature_engineering.py
-----------------------
Etapa 01 do roadmap: Engenharia de Recursos.

- Cria variáveis derivadas: hora do dia, período (madrugada/manhã/tarde/noite),
  diferença de tempo entre transações e frequência de uso.
- Aplica normalização/padronização (StandardScaler / RobustScaler).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


def add_time_features(df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
    """Cria features de horário a partir da coluna de tempo em segundos.

    - hora_do_dia: hora (0-23) derivada do tempo em segundos, módulo 24h.
    - periodo_dia: categoria (madrugada, manha, tarde, noite).
    """
    df = df.copy()
    seconds_in_day = df[time_col] % (24 * 3600)
    df["hora_do_dia"] = (seconds_in_day // 3600).astype(int)

    bins = [-1, 5, 11, 17, 23]
    labels = ["madrugada", "manha", "tarde", "noite"]
    df["periodo_dia"] = pd.cut(df["hora_do_dia"], bins=bins, labels=labels)
    df = pd.get_dummies(df, columns=["periodo_dia"], prefix="periodo", dtype=int)
    return df


def add_transaction_gap_features(
    df: pd.DataFrame, time_col: str = "Time", sort: bool = True
) -> pd.DataFrame:
    """Cria a diferença de tempo (em segundos) entre transações consecutivas,
    simulando o intervalo entre uma transação e a anterior (proxy para
    'velocidade' de uso do cartão).
    """
    df = df.copy()
    if sort:
        df = df.sort_values(time_col).reset_index(drop=True)

    df["gap_tempo_transacao_anterior"] = df[time_col].diff().fillna(0)
    df["gap_tempo_transacao_anterior"] = df["gap_tempo_transacao_anterior"].clip(lower=0)
    return df


def add_amount_features(df: pd.DataFrame, amount_col: str = "Amount") -> pd.DataFrame:
    """Cria features derivadas do valor da transação:
    - log_amount: log1p do valor, para reduzir a assimetria da distribuição.
    - amount_zscore_global: quão atípico é o valor em relação à média global.
    """
    df = df.copy()
    df["log_amount"] = np.log1p(df[amount_col])
    mean_, std_ = df[amount_col].mean(), df[amount_col].std()
    df["amount_zscore_global"] = (df[amount_col] - mean_) / (std_ if std_ > 0 else 1)
    return df


def add_frequency_features(
    df: pd.DataFrame, window: int = 10, amount_col: str = "Amount"
) -> pd.DataFrame:
    """Cria uma proxy de 'frequência de uso do cartão' via média móvel do
    número de transações e do valor médio dentro de uma janela deslizante
    (assumindo que o dataframe já está ordenado pelo tempo).
    """
    df = df.copy()
    df["media_movel_valor"] = (
        df[amount_col].rolling(window=window, min_periods=1).mean()
    )
    df["contagem_movel_transacoes"] = (
        df[amount_col].rolling(window=window, min_periods=1).count()
    )
    return df


def engineer_features(df: pd.DataFrame, time_col: str = "Time", amount_col: str = "Amount") -> pd.DataFrame:
    """Aplica todo o pipeline de feature engineering em sequência."""
    df_fe = add_transaction_gap_features(df, time_col=time_col)
    df_fe = add_time_features(df_fe, time_col=time_col)
    df_fe = add_amount_features(df_fe, amount_col=amount_col)
    df_fe = add_frequency_features(df_fe, amount_col=amount_col)
    return df_fe


def scale_features(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "robust",
) -> tuple[pd.DataFrame, StandardScaler | RobustScaler]:
    """Padroniza/normaliza as colunas numéricas indicadas.

    method: "standard" -> StandardScaler (média 0, desvio 1)
            "robust"   -> RobustScaler (baseado em mediana/IQR, mais
                          resistente a outliers -- recomendado para dados
                          de transações financeiras, que têm outliers reais
                          por natureza).
    """
    df = df.copy()
    scaler = RobustScaler() if method == "robust" else StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler
