"""
balancing.py
------------
Etapa 02 do roadmap: Tratamento do Desbalanceamento de Dados.

Oferece as três estratégias mais usadas em detecção de fraude:
- Oversampling: SMOTE, ADASYN
- Undersampling: RandomUnderSampler
- Também expõe uma opção "none" para servir de baseline comparativo.

IMPORTANTE: o balanceamento deve ser aplicado SOMENTE no conjunto de
treino, nunca no conjunto de teste/validação -- por isso as funções aqui
recebem X_train, y_train, e não o dataset inteiro.
"""

from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import RandomUnderSampler


def balance_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str = "smote",
    random_state: int = 42,
):
    """Rebalanceia (X_train, y_train) de acordo com a estratégia escolhida.

    strategy: "smote" | "adasyn" | "undersample" | "none"
    """
    if strategy == "none":
        return X_train, y_train

    if strategy == "smote":
        sampler = SMOTE(random_state=random_state)
    elif strategy == "adasyn":
        sampler = ADASYN(random_state=random_state)
    elif strategy == "undersample":
        sampler = RandomUnderSampler(random_state=random_state)
    else:
        raise ValueError(
            f"Estratégia '{strategy}' desconhecida. Use: smote, adasyn, undersample, none."
        )

    X_res, y_res = sampler.fit_resample(X_train, y_train)
    return X_res, y_res


def balance_report(y_before: pd.Series, y_after: pd.Series) -> pd.DataFrame:
    """Compara a distribuição de classes antes e depois do balanceamento."""
    before = y_before.value_counts().rename("antes")
    after = pd.Series(y_after).value_counts().rename("depois")
    return pd.concat([before, after], axis=1).fillna(0).astype(int)
