"""
balancing.py
------------
Tratamento do desbalanceamento — agora DENTRO do Pipeline.

Problema do projeto original: o SMOTE era aplicado antes da validação cruzada,
então amostras sintéticas (derivadas de vizinhos) caíam nos folds de validação
e inflavam o PR-AUC. Aqui o reamostrador é parte do estimador
(``ResampledClassifier``): só atua no ``fit`` e sempre apenas nos dados de
treino de cada fold. ``predict``/``predict_proba`` nunca reamostram.

Estratégias: ``none`` | ``class_weight`` | ``smote`` | ``adasyn`` | ``undersample``
  - class_weight: sem reamostragem; o peso é aplicado pelo próprio modelo
    (é o padrão da v2: costuma preservar melhor as probabilidades).
  - smote: implementação própria (só depende de scikit-learn).
  - adasyn: usa imbalanced-learn (opcional).
  - undersample: subamostragem aleatória da classe majoritária.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neighbors import NearestNeighbors

STRATEGIES = ("none", "class_weight", "smote", "adasyn", "undersample")


def _smote(X: np.ndarray, y: np.ndarray, ratio: float, k: int, rng: np.random.Generator):
    pos, neg = X[y == 1], X[y == 0]
    n_new = int(ratio * len(neg)) - len(pos)
    if n_new <= 0 or len(pos) < 2:
        return X, y
    k = min(k, len(pos) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(pos)
    neigh = nn.kneighbors(pos, return_distance=False)[:, 1:]
    base = rng.integers(0, len(pos), n_new)
    other = neigh[base, rng.integers(0, k, n_new)]
    gap = rng.random((n_new, 1))
    synth = pos[base] + gap * (pos[other] - pos[base])
    # Colunas discretas (ex.: categorias codificadas, flags): evita valores fracionários.
    discrete = [j for j in range(pos.shape[1]) if np.all(pos[:, j] == np.round(pos[:, j])) and len(np.unique(pos[:, j])) <= 30]
    if discrete:
        synth[:, discrete] = np.round(synth[:, discrete])
    return np.vstack([X, synth]), np.concatenate([y, np.ones(n_new, dtype=y.dtype)])


def resample(X, y, strategy: str = "smote", random_state: int = 42, ratio: float = 1.0):
    """Reamostra (X, y). ``ratio`` = minoritária/majoritária desejada (1.0 = 50/50)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Estratégia '{strategy}' desconhecida. Use: {', '.join(STRATEGIES)}.")
    if strategy in ("none", "class_weight"):
        return X, y
    cols = X.columns if isinstance(X, pd.DataFrame) else None
    Xa, ya = np.asarray(X, dtype=float), np.asarray(y)
    rng = np.random.default_rng(random_state)

    if strategy == "smote":
        Xr, yr = _smote(Xa, ya, ratio, 5, rng)
    elif strategy == "undersample":
        pos_idx, neg_idx = np.where(ya == 1)[0], np.where(ya == 0)[0]
        keep = rng.choice(neg_idx, size=min(len(neg_idx), int(len(pos_idx) / ratio)), replace=False)
        idx = np.sort(np.concatenate([pos_idx, keep]))
        Xr, yr = Xa[idx], ya[idx]
    else:  # adasyn
        try:
            from imblearn.over_sampling import ADASYN
        except ImportError as exc:
            raise ImportError("ADASYN requer imbalanced-learn: pip install imbalanced-learn") from exc
        Xr, yr = ADASYN(random_state=random_state, sampling_strategy=ratio).fit_resample(Xa, ya)
    return (pd.DataFrame(Xr, columns=cols) if cols is not None else Xr), yr


class ResampledClassifier(ClassifierMixin, BaseEstimator):
    """Envolve um classificador e reamostra SOMENTE durante o ``fit``."""

    def __init__(self, estimator=None, strategy: str = "smote", ratio: float = 1.0, random_state: int = 42):
        self.estimator = estimator
        self.strategy = strategy
        self.ratio = ratio
        self.random_state = random_state

    def fit(self, X, y):
        Xr, yr = resample(X, np.asarray(y), self.strategy, self.random_state, self.ratio)
        self.estimator_ = clone(self.estimator).fit(Xr, yr)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def balance_report(y_before, y_after) -> pd.DataFrame:
    """Compara a distribuição de classes antes e depois do balanceamento."""
    before = pd.Series(np.asarray(y_before)).value_counts().rename("antes")
    after = pd.Series(np.asarray(y_after)).value_counts().rename("depois")
    return pd.concat([before, after], axis=1).fillna(0).astype(int)


def balance_data(X_train, y_train, strategy: str = "smote", random_state: int = 42):
    """Compatibilidade com a v1."""
    return resample(X_train, y_train, strategy, random_state)
