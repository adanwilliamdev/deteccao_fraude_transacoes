"""
inference.py
------------
Camada de inferência: empacota tudo o que é necessário para pontuar
transações novas em UM arquivo (.joblib) — pipeline treinado, limiar de
decisão, lista de features e metadados — resolvendo a fragilidade da v1
(modelo e scaler salvos separados, sem o limiar nem a engenharia de features).

Importante: as features por cartão (velocidade, média histórica) dependem do
histórico. Ao pontuar transações novas, passe ``history`` com as transações
recentes (idealmente as últimas 24h+ dos cartões envolvidos).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from .feature_engineering import add_causal_features

TARGET = "Class"


@dataclass
class FraudDetector:
    pipeline: object
    threshold: float
    feature_columns: list
    meta: dict = field(default_factory=dict)
    drop_columns: tuple = ("id_cartao", "Time", TARGET)
    card_col: str | None = "id_cartao"
    engineer: bool = True  # False quando o treino não teve Time/Amount (sem features causais)

    # -- persistência --------------------------------------------------------
    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FraudDetector":
        return joblib.load(path)

    # -- pontuação -----------------------------------------------------------
    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.engineer:
            return df.reset_index(drop=True)
        card = self.card_col if self.card_col in df.columns else None
        return add_causal_features(df, card_col=card)

    def score(self, new: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
        """Retorna as transações novas com: prob_fraude, alerta (0/1) e faixa de risco."""
        new = new.copy()
        new["__novo__"] = True
        if history is not None and len(history):
            hist = history.copy()
            hist["__novo__"] = False
            all_df = pd.concat([hist, new], ignore_index=True)
        else:
            all_df = new
        fe = self._features(all_df)
        is_new = fe["__novo__"].to_numpy(bool)
        X = fe.drop(columns=[c for c in (*self.drop_columns, "__novo__") if c in fe.columns])
        X = X.reindex(columns=self.feature_columns)  # garante ordem/colunas do treino
        prob = self.pipeline.predict_proba(X[is_new])[:, 1]
        out = fe.loc[is_new].drop(columns=["__novo__"]).reset_index(drop=True)
        out["prob_fraude"] = prob
        out["alerta"] = (prob >= self.threshold).astype(int)
        out["faixa_risco"] = np.select(
            [prob >= self.threshold, prob >= 0.5 * self.threshold], ["alto", "médio"], default="baixo")
        return out
