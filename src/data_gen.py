"""
data_gen.py
-----------
Gera um dataset sintético de transações financeiras, com a MESMA estrutura
conceitual do problema clássico de detecção de fraude:

    - Base fortemente desbalanceada (fraudes ~0.5% a 2% dos casos)
    - Colunas numéricas anônimas (V1..V14), simulando componentes latentes
    - Amount (valor da transação)
    - Time (segundos desde a primeira transação)
    - Class (0 = legítima, 1 = fraude)

Se o usuário tiver um dataset real (ex: creditcard.csv do Kaggle, ou o
extrato de transações da própria empresa), basta substituir a chamada
`load_dataset()` em main.py para ler o CSV real em vez de gerar dados
sintéticos. A estrutura do pipeline (EDA, feature engineering, balanceamento,
modelos, XAI) continua exatamente a mesma.
"""

import numpy as np
import pandas as pd


def generate_synthetic_transactions(
    n_samples: int = 50_000,
    fraud_ratio: float = 0.015,
    n_features: int = 14,
    random_state: int = 42,
) -> pd.DataFrame:
    """Gera um DataFrame sintético de transações rotuladas (fraude / legítima).

    Parameters
    ----------
    n_samples : int
        Número total de transações a gerar.
    fraud_ratio : float
        Proporção aproximada de transações fraudulentas (classe minoritária).
    n_features : int
        Número de variáveis latentes (V1..Vn) a gerar.
    random_state : int
        Semente para reprodutibilidade.

    Returns
    -------
    pd.DataFrame
        Dataset com colunas: Time, V1..Vn, Amount, Class.
    """
    rng = np.random.default_rng(random_state)

    n_fraud = max(1, int(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud

    # --- Transações legítimas ---------------------------------------------
    legit_features = rng.normal(loc=0.0, scale=1.0, size=(n_legit, n_features))
    legit_amount = np.abs(rng.normal(loc=60, scale=45, size=n_legit))
    legit_amount = np.clip(legit_amount, 1, 2000)

    # Horário de pico durante o dia (transações legítimas concentradas)
    legit_time = rng.normal(loc=14 * 3600, scale=5 * 3600, size=n_legit)
    legit_time = np.mod(legit_time, 24 * 3600)

    # --- Transações fraudulentas --------------------------------------------
    # Deslocamos a distribuição das variáveis latentes e do valor para
    # simular padrões de comportamento anômalo (mais dispersos, valores
    # atípicos, concentração em horários de madrugada). O deslocamento é
    # moderado e apenas parte das colunas é afetada, para gerar sobreposição
    # realista entre as classes (em fraude real, os padrões nunca são
    # perfeitamente separáveis).
    fraud_features = rng.normal(loc=0.0, scale=1.0, size=(n_fraud, n_features))
    n_shifted_cols = max(1, n_features // 3)
    shifted_cols = rng.choice(n_features, size=n_shifted_cols, replace=False)
    fraud_features[:, shifted_cols] += rng.normal(loc=1.4, scale=0.6, size=(n_fraud, n_shifted_cols))
    fraud_features += rng.normal(loc=0.0, scale=0.6, size=fraud_features.shape)  # ruído extra

    fraud_amount = np.abs(rng.normal(loc=180, scale=220, size=n_fraud))
    fraud_amount = np.clip(fraud_amount, 1, 5000)

    fraud_time = rng.normal(loc=7 * 3600, scale=6 * 3600, size=n_fraud)
    fraud_time = np.mod(fraud_time, 24 * 3600)

    # --- Junta tudo ----------------------------------------------------------
    features = np.vstack([legit_features, fraud_features])
    amount = np.concatenate([legit_amount, fraud_amount])
    time_sec = np.concatenate([legit_time, fraud_time])
    labels = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

    columns = [f"V{i+1}" for i in range(n_features)]
    df = pd.DataFrame(features, columns=columns)
    df["Amount"] = amount
    df["Time"] = time_sec
    df["Class"] = labels.astype(int)

    # Introduz alguns valores nulos e duplicados de propósito, para que a
    # etapa de EDA tenha algo relevante para encontrar e tratar.
    dup_idx = rng.choice(df.index, size=int(0.005 * len(df)), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    null_idx = rng.choice(df.index, size=int(0.002 * len(df)), replace=False)
    df.loc[null_idx, "Amount"] = np.nan

    # Embaralha as linhas para não deixar as fraudes agrupadas no final.
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def load_dataset(csv_path: str | None = None, **synthetic_kwargs) -> pd.DataFrame:
    """Carrega o dataset real, se um caminho for informado; caso contrário,
    gera um dataset sintético para que o pipeline funcione de ponta a ponta.
    """
    if csv_path:
        return pd.read_csv(csv_path)
    return generate_synthetic_transactions(**synthetic_kwargs)
