"""
models.py
---------
Etapa 03 do roadmap: Modelos Avançados.

- Baseline: Regressão Logística
- Ensembles: Random Forest, XGBoost, LightGBM, CatBoost
- Não supervisionados para anomalias: Isolation Forest, One-Class SVM

Todos os modelos supervisionados são treinados com `class_weight="balanced"`
(quando suportado) além do balanceamento de dados feito na etapa anterior,
como camada extra de proteção contra o desbalanceamento.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier


@dataclass
class ModelZoo:
    """Fábrica central de modelos usados no projeto."""

    random_state: int = 42
    contamination: float = 0.02  # usado pelos modelos não supervisionados

    def logistic_regression(self) -> LogisticRegression:
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=self.random_state,
        )

    def random_forest(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            n_jobs=-1,
            random_state=self.random_state,
        )

    def xgboost(self, scale_pos_weight: float = 1.0) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def lightgbm(self, scale_pos_weight: float = 1.0) -> LGBMClassifier:
        return LGBMClassifier(
            n_estimators=300,
            max_depth=-1,
            learning_rate=0.08,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    def catboost(self, scale_pos_weight: float = 1.0) -> CatBoostClassifier:
        return CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.08,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            verbose=False,
        )

    def isolation_forest(self) -> IsolationForest:
        return IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def one_class_svm(self) -> OneClassSVM:
        return OneClassSVM(kernel="rbf", nu=self.contamination, gamma="scale")


def get_supervised_models(zoo: ModelZoo, scale_pos_weight: float) -> dict:
    """Retorna um dicionário {nome: modelo} com todos os classificadores
    supervisionados a comparar."""
    return {
        "Regressao_Logistica": zoo.logistic_regression(),
        "Random_Forest": zoo.random_forest(),
        "XGBoost": zoo.xgboost(scale_pos_weight=scale_pos_weight),
        "LightGBM": zoo.lightgbm(scale_pos_weight=scale_pos_weight),
        "CatBoost": zoo.catboost(scale_pos_weight=scale_pos_weight),
    }


def get_anomaly_models(zoo: ModelZoo) -> dict:
    """Retorna os modelos não supervisionados de detecção de anomalia."""
    return {
        "Isolation_Forest": zoo.isolation_forest(),
        "One_Class_SVM": zoo.one_class_svm(),
    }


def anomaly_scores_to_labels(raw_pred: np.ndarray) -> np.ndarray:
    """IsolationForest e OneClassSVM retornam -1 (anomalia) / 1 (normal).
    Convertemos para o padrão do projeto: 1 = fraude, 0 = legítima.
    """
    return np.where(raw_pred == -1, 1, 0)
