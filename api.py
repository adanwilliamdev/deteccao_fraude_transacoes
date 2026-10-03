"""
api.py — serviço HTTP opcional (FastAPI) para pontuar transações em tempo real.

    pip install fastapi uvicorn
    MODEL_PATH=outputs/modelos/modelo_fraude.joblib uvicorn api:app --port 8000

    POST /score   {"transacoes": [{...}], "historico": [{...}]}   (historico é opcional)
    GET  /health

Observação: as features por cartão precisam do histórico recente. Em produção,
alimente ``historico`` a partir de um feature store/cache (ex.: Redis) com as
últimas 24h do cartão; este exemplo recebe o histórico no próprio request.
"""

from __future__ import annotations

import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference import FraudDetector

MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/modelos/modelo_fraude.joblib")
app = FastAPI(title="API de Detecção de Fraude", version="2.0.0")
_detector: FraudDetector | None = None


def detector() -> FraudDetector:
    global _detector
    if _detector is None:
        _detector = FraudDetector.load(MODEL_PATH)
    return _detector


class ScoreRequest(BaseModel):
    transacoes: list[dict]
    historico: list[dict] | None = None


@app.get("/health")
def health():
    d = detector()
    return {"status": "ok", "modelo": d.meta.get("modelo"), "limiar": d.threshold, "versao": d.meta.get("versao")}


@app.post("/score")
def score(req: ScoreRequest):
    if not req.transacoes:
        raise HTTPException(status_code=422, detail="Envie ao menos uma transação.")
    try:
        res = detector().score(pd.DataFrame(req.transacoes), pd.DataFrame(req.historico) if req.historico else None)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Coluna ausente: {exc}") from exc
    cols = [c for c in ("Time", "id_cartao", "prob_fraude", "alerta", "faixa_risco") if c in res.columns]
    return {"resultados": res[cols].to_dict(orient="records")}
