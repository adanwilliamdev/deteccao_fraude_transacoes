"""
models.py
---------
Fábrica de modelos da v2.

- Supervisionados: Regressão Logística, Random Forest, Extra Trees,
  HistGradientBoosting (sempre disponível, nativo do scikit-learn) e — se
  instalados — XGBoost, LightGBM e CatBoost. Bibliotecas ausentes são
  ignoradas com aviso, em vez de quebrar o pipeline.
- Cada modelo vira um ``Pipeline(prep -> ResampledClassifier)``, então imputação,
  escala e balanceamento são ajustados APENAS no treino (e em cada fold da CV).
- ``ProbaAverager``: ensemble por média (ponderada) de probabilidades de
  pipelines já treinados — sem custo extra de treinamento.
- Não supervisionados: Isolation Forest e One-Class SVM treinados só com
  transações legítimas (detecção de novidade) e avaliados por *score* contínuo
  (PR-AUC/ROC-AUC), não só pelo rótulo -1/1.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

from .balancing import ResampledClassifier
from .feature_engineering import TabularPreprocessor

log = logging.getLogger("fraude")


def optional_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:  # ImportError ou erro de biblioteca nativa
        return None


@dataclass
class ModelZoo:
    random_state: int = 42
    contamination: float = 0.02
    n_jobs: int = -1

    def logistic_regression(self, pos_weight: float = 1.0):
        return LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced", random_state=self.random_state)

    def random_forest(self, pos_weight: float = 1.0):
        return RandomForestClassifier(
            n_estimators=200, max_depth=14, min_samples_leaf=3, class_weight="balanced_subsample",
            n_jobs=self.n_jobs, random_state=self.random_state)

    def extra_trees(self, pos_weight: float = 1.0):
        return ExtraTreesClassifier(
            n_estimators=250, min_samples_leaf=3, class_weight="balanced_subsample",
            n_jobs=self.n_jobs, random_state=self.random_state)

    def hist_gradient_boosting(self, pos_weight: float = 1.0):
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
            class_weight={0: 1.0, 1: float(min(pos_weight, 50.0))}, random_state=self.random_state)

    def xgboost(self, pos_weight: float = 1.0):
        xgb = optional_import("xgboost")
        return None if xgb is None else xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            eval_metric="aucpr", scale_pos_weight=pos_weight, random_state=self.random_state, n_jobs=self.n_jobs)

    def lightgbm(self, pos_weight: float = 1.0):
        lgb = optional_import("lightgbm")
        return None if lgb is None else lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=pos_weight, random_state=self.random_state, n_jobs=self.n_jobs, verbosity=-1)

    def catboost(self, pos_weight: float = 1.0):
        cb = optional_import("catboost")
        return None if cb is None else cb.CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.06, scale_pos_weight=pos_weight,
            random_state=self.random_state, verbose=False, thread_count=self.n_jobs)


# nome -> (método da fábrica, precisa de escala?)
SUPERVISED_SPECS = {
    "Regressao_Logistica": ("logistic_regression", True),
    "Random_Forest": ("random_forest", False),
    "Extra_Trees": ("extra_trees", False),
    "HistGradientBoosting": ("hist_gradient_boosting", False),
    "XGBoost": ("xgboost", False),
    "LightGBM": ("lightgbm", False),
    "CatBoost": ("catboost", False),
}


def get_supervised_pipelines(
    zoo: ModelZoo, pos_weight: float, strategy: str = "class_weight", ratio: float = 1.0, only: list[str] | None = None,
) -> dict[str, Pipeline]:
    """{nome: Pipeline(prep -> reamostragem -> modelo)}. Modelos cuja biblioteca
    não está instalada são pulados (com aviso)."""
    # Só a estratégia "class_weight" aplica peso de classe. Com reamostragem real
    # (smote/undersample/adasyn) o peso é neutralizado (evita compensar 2x);
    # "none" é o baseline sem nenhum tratamento de desbalanceamento.
    weight = pos_weight if strategy == "class_weight" else 1.0
    pipes = {}
    for name, (factory, needs_scale) in SUPERVISED_SPECS.items():
        if only and name not in only:
            continue
        est = getattr(zoo, factory)(pos_weight=weight)
        if est is None:
            log.warning("  -> %s ignorado (biblioteca não instalada).", name)
            continue
        if strategy != "class_weight" and name in ("Regressao_Logistica", "Random_Forest", "Extra_Trees"):
            est.set_params(class_weight=None)
        pipes[name] = Pipeline([
            ("prep", TabularPreprocessor(scale=needs_scale)),
            ("clf", ResampledClassifier(est, strategy=strategy, ratio=ratio, random_state=zoo.random_state)),
        ])
    return pipes


class ProbaAverager(ClassifierMixin, BaseEstimator):
    """Ensemble: média ponderada das probabilidades de pipelines já treinados."""

    def __init__(self, pipelines: dict | None = None, weights: dict | None = None):
        self.pipelines = pipelines
        self.weights = weights

    def fit(self, X=None, y=None):  # já treinados; apenas registra as classes
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        w = {k: (self.weights or {}).get(k, 1.0) for k in self.pipelines}
        total = sum(w.values())
        p1 = sum(w[k] * pipe.predict_proba(X)[:, 1] for k, pipe in self.pipelines.items()) / total
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --------------------------------------------------------------------------- #
# Detecção de anomalias (não supervisionada / novidade)
# --------------------------------------------------------------------------- #
class AnomalyScorer:
    """Treina com transações legítimas e devolve score de fraude (maior = mais anômalo)."""

    def __init__(self, kind: str = "isolation_forest", contamination: float = 0.02, random_state: int = 42,
                 max_fit_samples: int = 8000):
        self.kind, self.contamination, self.random_state, self.max_fit_samples = kind, contamination, random_state, max_fit_samples

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.prep_ = TabularPreprocessor(scale=True).fit(X)
        Xl = self.prep_.transform(X[np.asarray(y) == 0])
        if self.kind == "one_class_svm" and len(Xl) > self.max_fit_samples:
            Xl = Xl.sample(self.max_fit_samples, random_state=self.random_state)
        if self.kind == "isolation_forest":
            self.model_ = IsolationForest(n_estimators=300, contamination=self.contamination,
                                          random_state=self.random_state, n_jobs=-1)
        else:
            self.model_ = OneClassSVM(kernel="rbf", nu=max(self.contamination, 0.005), gamma="scale")
        self.model_.fit(Xl)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return -self.model_.decision_function(self.prep_.transform(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.model_.predict(self.prep_.transform(X)) == -1).astype(int)


def anomaly_scores_to_labels(raw_pred: np.ndarray) -> np.ndarray:
    """Compatibilidade v1: -1 (anomalia) -> 1 (fraude)."""
    return np.where(raw_pred == -1, 1, 0)
