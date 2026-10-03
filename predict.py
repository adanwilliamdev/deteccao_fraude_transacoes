"""
predict.py
==========
Pontua transações novas com o modelo treinado (um único arquivo .joblib).

    python predict.py --model outputs/modelos/modelo_fraude.joblib \
                      --input novas_transacoes.csv \
                      --history historico_recente.csv \
                      --output pontuadas.csv

``--history`` é opcional, mas recomendado: as features por cartão (velocidade,
média histórica) usam as transações anteriores do mesmo cartão.
O CSV precisa das mesmas colunas brutas do treino (sem a coluna Class).
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.inference import FraudDetector


def main():
    ap = argparse.ArgumentParser(description="Pontua transações com o modelo de fraude treinado.")
    ap.add_argument("--model", default="outputs/modelos/modelo_fraude.joblib")
    ap.add_argument("--input", required=True, help="CSV com as transações a pontuar.")
    ap.add_argument("--history", default=None, help="CSV com transações recentes (contexto por cartão).")
    ap.add_argument("--output", default="transacoes_pontuadas.csv")
    ap.add_argument("--top", type=int, default=10, help="Quantas transações de maior risco exibir.")
    a = ap.parse_args()

    det = FraudDetector.load(a.model)
    new = pd.read_csv(a.input)
    hist = pd.read_csv(a.history) if a.history else None
    res = det.score(new, history=hist)
    res.to_csv(a.output, index=False)

    n_alert = int(res["alerta"].sum())
    print(f"Modelo: {det.meta.get('modelo')} | limiar: {det.threshold:.4f}")
    print(f"{len(res)} transações pontuadas -> {n_alert} alertas ({100 * n_alert / max(1, len(res)):.2f}%). Salvo em {a.output}")
    show = [c for c in ("Time", "id_cartao", "Amount", "categoria_comerciante", "prob_fraude", "faixa_risco") if c in res.columns]
    print(res.sort_values("prob_fraude", ascending=False).head(a.top)[show].to_string(index=False))


if __name__ == "__main__":
    main()
