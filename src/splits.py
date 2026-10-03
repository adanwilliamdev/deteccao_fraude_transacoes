"""
splits.py
---------
Validação do esquema e divisão dos dados.

Em fraude o split correto é TEMPORAL: treina-se no passado e testa-se no
futuro, como acontece em produção. O split aleatório estratificado (padrão da
v1) deixa transações do mesmo cartão/rajada em treino e teste ao mesmo tempo e
superestima o desempenho. Ele continua disponível (``split="stratified"``)
para comparação.

Três blocos: treino-interno (ajuste do modelo), validação (escolha de modelo,
pesos do ensemble e LIMIAR) e teste (relatório final, nunca usado para decidir).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED = ("Class",)


def validate_dataset(df: pd.DataFrame, target_col: str = "Class", time_col: str = "Time", amount_col: str = "Amount") -> list[str]:
    """Levanta ValueError para problemas fatais e devolve avisos não fatais."""
    if target_col not in df.columns:
        raise ValueError(f"O dataset precisa ter a coluna alvo '{target_col}' (0 = legítima, 1 = fraude).")
    labels = set(pd.Series(df[target_col]).dropna().unique())
    if not labels <= {0, 1}:
        raise ValueError(f"A coluna '{target_col}' deve conter apenas 0/1; encontrado: {sorted(labels)[:5]}")
    n_pos = int((df[target_col] == 1).sum())
    if n_pos < 30:
        raise ValueError(f"Apenas {n_pos} fraudes no dataset — insuficiente para treinar/avaliar com segurança (mín. 30).")
    warns = []
    for col, why in ((time_col, "features temporais/de velocidade e split temporal"), (amount_col, "features de valor e análise de custo")):
        if col not in df.columns:
            warns.append(f"Coluna '{col}' ausente: desativa {why}.")
    if n_pos < 200:
        warns.append(f"Poucas fraudes ({n_pos}): métricas terão alta variância; confie nos intervalos de confiança.")
    return warns


def temporal_split(df: pd.DataFrame, time_col: str = "Time", test_size: float = 0.25, val_size: float = 0.20):
    """Ordena por tempo e corta: [treino-interno | validação | teste]."""
    d = df.sort_values(time_col, kind="stable").reset_index(drop=True)
    n = len(d)
    n_test = int(n * test_size)
    n_trainval = n - n_test
    n_val = int(n_trainval * val_size)
    return d.iloc[: n_trainval - n_val], d.iloc[n_trainval - n_val: n_trainval], d.iloc[n_trainval:]


def stratified_split(df: pd.DataFrame, target_col: str = "Class", test_size: float = 0.25,
                     val_size: float = 0.20, random_state: int = 42):
    trval, test = train_test_split(df, test_size=test_size, stratify=df[target_col], random_state=random_state)
    tr, val = train_test_split(trval, test_size=val_size, stratify=trval[target_col], random_state=random_state)
    return tr.sort_index(), val.sort_index(), test.sort_index()
