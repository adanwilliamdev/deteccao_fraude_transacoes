"""
explainability.py
------------------
Explicabilidade (XAI) do modelo.

- Com ``shap`` instalado: importância global (summary) e explicação local
  (waterfall) — como na v1, agora aplicadas ao modelo final do Pipeline.
- SEM ``shap``: fallback 100% scikit-learn, sem perder a funcionalidade:
    * importância global por permutação (queda de PR-AUC ao embaralhar a feature);
    * explicação local por oclusão (quanto o risco, em log-odds, cai ao
      substituir cada feature por seu valor típico) — útil para o analista entender um alerta.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .models import optional_import


def has_shap() -> bool:
    return optional_import("shap") is not None


def split_pipeline(pipe):
    """Devolve (pré-processador, modelo final) de um Pipeline prep -> clf."""
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    return prep, getattr(clf, "estimator_", clf)


# --------------------------------------------------------------------------- #
# SHAP (opcional)
# --------------------------------------------------------------------------- #
def compute_shap_values(pipe, X_background: pd.DataFrame, X_explain: pd.DataFrame, n_background=200, n_explain=500):
    shap = optional_import("shap")
    if shap is None:
        raise ImportError("shap não instalado")
    prep, model = split_pipeline(pipe)
    bg = prep.transform(X_background.sample(n=min(n_background, len(X_background)), random_state=42))
    ex = prep.transform(X_explain.sample(n=min(n_explain, len(X_explain)), random_state=42))
    try:
        values = shap.Explainer(model, bg)(ex)
    except Exception:
        explainer = shap.KernelExplainer(lambda d: model.predict_proba(pd.DataFrame(d, columns=ex.columns))[:, 1], bg)
        raw = explainer.shap_values(ex, nsamples=100)
        values = shap.Explanation(values=raw, base_values=np.full(len(ex), explainer.expected_value),
                                  data=ex.values, feature_names=list(ex.columns))
    if getattr(values, "values", None) is not None and values.values.ndim == 3:
        values = values[:, :, 1]
    return values, ex


def plot_shap_summary(shap_values, output_path: str) -> None:
    shap = optional_import("shap")
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_shap_waterfall(shap_values, index: int, output_path: str) -> None:
    shap = optional_import("shap")
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(shap_values[index], show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def top_features_for_transaction(shap_values, index: int, top_n: int = 5) -> pd.DataFrame:
    df = pd.DataFrame({"feature": shap_values[index].feature_names, "shap_value": shap_values[index].values})
    return df.reindex(df["shap_value"].abs().sort_values(ascending=False).index).head(top_n).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Fallback sem SHAP
# --------------------------------------------------------------------------- #
def global_importance(pipe, X: pd.DataFrame, y, n_repeats: int = 3, max_rows: int = 6000, random_state: int = 42) -> pd.DataFrame:
    """Importância por permutação (queda de PR-AUC), model-agnostic."""
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n_neg = max(0, max_rows - len(pos))
    idx = np.sort(np.concatenate([pos, rng.choice(neg, min(n_neg, len(neg)), replace=False)]))
    r = permutation_importance(pipe, X.iloc[idx], y[idx], scoring="average_precision",
                               n_repeats=n_repeats, random_state=random_state, n_jobs=1)
    return (pd.DataFrame({"feature": X.columns, "queda_pr_auc": r.importances_mean, "desvio": r.importances_std})
            .sort_values("queda_pr_auc", ascending=False).reset_index(drop=True))


def _log_odds(pipe, X: pd.DataFrame) -> float:
    """Log-odds de fraude. Usa decision_function (exato, não satura) quando o modelo
    final tem; senão, logit da probabilidade com recorte mínimo."""
    try:
        prep, est = split_pipeline(pipe)
        if hasattr(est, "decision_function"):
            return float(np.ravel(est.decision_function(prep.transform(X)))[0])
    except Exception:
        pass
    p = float(np.clip(pipe.predict_proba(X)[:, 1][0], 1e-12, 1 - 1e-12))
    return float(np.log(p / (1 - p)))


def local_explanation(pipe, X_row: pd.DataFrame, X_reference: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """Oclusão: para cada feature, troca pelo valor típico (mediana/moda da
    referência) e mede a variação do RISCO. O efeito é medido em log-odds
    (não em probabilidade) porque probabilidades saturadas em ~1,0 escondem o
    peso das variáveis. ``efeito_logit`` > 0 = a feature EMPURRA para 'fraude'."""
    base_p = float(pipe.predict_proba(X_row)[:, 1][0])
    base_lo = _log_odds(pipe, X_row)
    rows = []
    for c in X_row.columns:
        alt = X_row.copy()
        ref = X_reference[c]
        alt[c] = ref.median() if pd.api.types.is_numeric_dtype(ref) else ref.mode().iloc[0]
        rows.append({"feature": c, "valor": X_row[c].iloc[0], "valor_tipico": alt[c].iloc[0],
                     "efeito_logit": base_lo - _log_odds(pipe, alt)})
    df = pd.DataFrame(rows)
    df = df.reindex(df["efeito_logit"].abs().sort_values(ascending=False).index).head(top_n).reset_index(drop=True)
    df.attrs["prob_base"] = base_p
    return df


def plot_importance(df: pd.DataFrame, output_path: str, col: str = "queda_pr_auc", top_n: int = 15,
                    title: str = "Importância global (permutação)") -> None:
    d = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(d["feature"], d[col], xerr=d["desvio"] if "desvio" in d else None, color="#3b6ea5")
    ax.set(xlabel="Queda média do PR-AUC ao embaralhar a feature", title=title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_local_explanation(df: pd.DataFrame, output_path: str, title: str = "Por que esta transação foi sinalizada?") -> None:
    d = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(d["feature"], d["efeito_logit"], color=np.where(d["efeito_logit"] > 0, "#c0392b", "#2e86c1"))
    ax.axvline(0, c="k", lw=0.8)
    ax.set(xlabel="Efeito no log-odds de fraude (vermelho = aumenta o risco)",
           title=f"{title}  (prob. = {df.attrs.get('prob_base', float('nan')):.2f})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
