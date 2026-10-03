"""
evaluation.py
-------------
Avaliação orientada a NEGÓCIO para fraude.

Novidades da v2 em relação à v1:
  - Limiar de decisão otimizado (F-beta, custo esperado ou recall-alvo) em vez
    do 0,5 fixo — escolhido na VALIDAÇÃO e aplicado no TESTE.
  - Custo esperado: custo de revisão por alerta + valor das fraudes perdidas.
  - Métricas extras: MCC, acurácia balanceada, FPR, Brier, precisão/recall no
    top-1% de maior risco (capacidade de revisão da equipe) e recall a FPR fixo.
  - Intervalos de confiança por bootstrap estratificado.
  - Validação cruzada estratificada E temporal (janela expansiva).
  - Gráficos: PR/ROC sobrepostos, análise de limiar, calibração, ganho/lift e
    curva de custo.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_val_score


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def top_k_metrics(y_true: np.ndarray, score: np.ndarray, frac: float = 0.01) -> tuple[float, float]:
    """Precisão e recall ao revisar apenas a fração `frac` de maior risco."""
    k = max(1, int(round(frac * len(score))))
    top = np.argsort(-score, kind="stable")[:k]
    tp = y_true[top].sum()
    return float(tp / k), float(tp / max(1, y_true.sum()))


def recall_at_fpr(y_true: np.ndarray, score: np.ndarray, max_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, score)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def expected_cost(y_true, y_pred, amount, review_cost: float = 5.0, fn_factor: float = 1.0) -> dict:
    """Custo = revisão de cada alerta + (fn_factor x valor) das fraudes não detectadas."""
    y_true, y_pred, amount = map(np.asarray, (y_true, y_pred, amount))
    flagged = int(y_pred.sum())
    missed = float(amount[(y_true == 1) & (y_pred == 0)].sum()) * fn_factor
    baseline = float(amount[y_true == 1].sum()) * fn_factor  # sem modelo: perde tudo
    total = review_cost * flagged + missed
    return {"custo_total": total, "custo_sem_modelo": baseline, "economia": baseline - total,
            "economia_pct": (baseline - total) / baseline if baseline > 0 else 0.0,
            "fraude_perdida": missed, "alertas": flagged}


def evaluate_scores(y_true, score, threshold: float | None = None, is_proba: bool = True,
                    amount=None, review_cost: float = 5.0, fn_factor: float = 1.0) -> dict:
    """Métricas independentes de limiar + métricas no limiar informado."""
    y_true, score = np.asarray(y_true), np.asarray(score, dtype=float)
    m = {"pr_auc": float(average_precision_score(y_true, score)), "roc_auc": float(roc_auc_score(y_true, score))}
    if is_proba:
        m["brier"] = float(brier_score_loss(y_true, np.clip(score, 0, 1)))
    p1, r1 = top_k_metrics(y_true, score, 0.01)
    m.update({"precisao_top1pct": p1, "recall_top1pct": r1, "recall_fpr_0.1pct": recall_at_fpr(y_true, score, 0.001)})
    if threshold is not None:
        pred = (score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        m.update({
            "limiar": float(threshold),
            "precisao": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, pred)),
            "acuracia_balanceada": float(balanced_accuracy_score(y_true, pred)),
            "fpr": float(fp / max(1, fp + tn)),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn), "alertas": int(tp + fp),
        })
        if amount is not None:
            m.update(expected_cost(y_true, pred, amount, review_cost, fn_factor))
    return m


def evaluate_predictions(y_true, y_pred, y_proba=None) -> dict:
    """Interface da v1 (mantida): métricas no rótulo previsto + AUCs se houver score."""
    out = {"precisao": precision_score(y_true, y_pred, zero_division=0),
           "recall": recall_score(y_true, y_pred, zero_division=0),
           "f1_score": f1_score(y_true, y_pred, zero_division=0)}
    if y_proba is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
        out["pr_auc"] = average_precision_score(y_true, y_proba)
    return out


# --------------------------------------------------------------------------- #
# Escolha de limiar (sempre em dados de VALIDAÇÃO, nunca no teste)
# --------------------------------------------------------------------------- #
def threshold_f_beta(y_true, score, beta: float = 1.0) -> float:
    prec, rec, thr = precision_recall_curve(y_true, score)
    prec, rec = prec[:-1], rec[:-1]
    f = (1 + beta**2) * prec * rec / np.maximum(beta**2 * prec + rec, 1e-12)
    return float(thr[int(np.argmax(f))])


def threshold_min_cost(y_true, score, amount, review_cost: float = 5.0, fn_factor: float = 1.0) -> float:
    """Limiar que minimiza: review_cost x nº de alertas + fn_factor x fraude perdida."""
    y_true, score, amount = np.asarray(y_true), np.asarray(score, float), np.asarray(amount, float)
    order = np.argsort(-score, kind="stable")
    s, yy, a = score[order], y_true[order], amount[order]
    total_fraud = float((a * yy).sum())
    k = np.arange(1, len(s) + 1)
    cost = review_cost * k + fn_factor * (total_fraud - np.cumsum(a * yy))
    cost = np.where(np.r_[s[:-1] != s[1:], True], cost, np.inf)  # só corta entre scores distintos
    best = int(np.argmin(cost))
    return float(s[best]) if cost[best] < fn_factor * total_fraud else float(s[0] + 1e-9)


def threshold_recall_target(y_true, score, target: float = 0.8) -> float:
    prec, rec, thr = precision_recall_curve(y_true, score)
    ok = np.where(rec[:-1] >= target)[0]
    if len(ok) == 0:
        return float(thr[0])
    return float(thr[ok[np.argmax(prec[:-1][ok])]])


def pick_threshold(strategy: str, y_true, score, amount=None, review_cost=5.0, fn_factor=1.0, recall_target=0.8) -> float:
    if strategy == "cost":
        if amount is None:
            raise ValueError("A estratégia 'cost' precisa da coluna de valores (Amount).")
        return threshold_min_cost(y_true, score, amount, review_cost, fn_factor)
    if strategy == "f1":
        return threshold_f_beta(y_true, score, 1.0)
    if strategy == "f2":
        return threshold_f_beta(y_true, score, 2.0)
    if strategy == "recall":
        return threshold_recall_target(y_true, score, recall_target)
    if strategy == "default":
        return 0.5
    raise ValueError(f"Estratégia de limiar desconhecida: {strategy}")


# --------------------------------------------------------------------------- #
# Incerteza e validação cruzada
# --------------------------------------------------------------------------- #
def bootstrap_ci(y_true, score, metric=average_precision_score, n_boot: int = 300,
                 alpha: float = 0.05, random_state: int = 42) -> tuple[float, float, float]:
    """IC por bootstrap ESTRATIFICADO (reamostra fraudes e legítimas separadamente)."""
    y_true, score = np.asarray(y_true), np.asarray(score, float)
    rng = np.random.default_rng(random_state)
    pos, neg = np.where(y_true == 1)[0], np.where(y_true == 0)[0]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(metric(y_true[idx], score[idx]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(metric(y_true, score)), float(lo), float(hi)


def stratified_cross_validation(model, X, y, scoring="average_precision", n_splits=5, random_state=42):
    """CV estratificada. Passe um Pipeline com o reamostrador DENTRO (sem vazamento)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=1)


def temporal_cross_validation(model, X: pd.DataFrame, y, n_splits: int = 4) -> pd.DataFrame:
    """Janela expansiva: treina no passado, valida no bloco seguinte (X ordenado no tempo)."""
    y = np.asarray(y)
    rows = []
    for i, (tr, va) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(X), start=1):
        if y[tr].sum() < 5 or y[va].sum() < 2:
            continue
        m = clone(model).fit(X.iloc[tr], y[tr])
        s = m.predict_proba(X.iloc[va])[:, 1]
        rows.append({"fold": i, "n_treino": len(tr), "n_validacao": len(va), "fraudes_validacao": int(y[va].sum()),
                     "pr_auc": average_precision_score(y[va], s), "roc_auc": roc_auc_score(y[va], s)})
    return pd.DataFrame(rows)


def compare_models(results: dict[str, dict], sort_by: str = "pr_auc") -> pd.DataFrame:
    df = pd.DataFrame(results).T
    col = sort_by if sort_by in df.columns else ("recall" if "recall" in df.columns else df.columns[0])
    return df.sort_values(col, ascending=False)


# --------------------------------------------------------------------------- #
# Gráficos
# --------------------------------------------------------------------------- #
def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, model_name: str, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=["Legítima", "Fraude"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Matriz de confusão - {model_name}")
    _save(fig, output_path)


def plot_pr_roc_curves(y_true, y_proba, model_name: str, output_path: str) -> None:
    plot_pr_roc_overlay({model_name: (y_true, y_proba)}, output_path, title=model_name)


def plot_pr_roc_overlay(curves: dict, output_path: str, title: str = "Comparação") -> None:
    """curves: {nome: (y_true, score)} — todos os modelos no mesmo gráfico."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, (y, s) in curves.items():
        p, r, _ = precision_recall_curve(y, s)
        axes[0].plot(r, p, label=f"{name} ({average_precision_score(y, s):.3f})")
        f, t, _ = roc_curve(y, s)
        axes[1].plot(f, t, label=f"{name} ({roc_auc_score(y, s):.3f})")
    base = np.mean(list(curves.values())[0][0])
    axes[0].axhline(base, ls="--", c="gray", lw=1, label=f"acaso ({base:.3f})")
    axes[0].set(xlabel="Recall", ylabel="Precisão", title=f"Precision-Recall — {title}")
    axes[1].plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
    axes[1].set(xlabel="FPR", ylabel="TPR", title=f"ROC — {title}")
    for ax in axes:
        ax.legend(fontsize=7, loc="best")
    _save(fig, output_path)


def plot_threshold_analysis(y_true, score, chosen: dict, output_path: str) -> None:
    """Precisão/recall/F1 em função do limiar, com os limiares escolhidos marcados."""
    prec, rec, thr = precision_recall_curve(y_true, score)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(thr, prec[:-1], label="Precisão")
    ax.plot(thr, rec[:-1], label="Recall")
    ax.plot(thr, f1, label="F1", ls="--")
    for (name, t), c in zip(chosen.items(), plt.cm.tab10.colors[3:]):
        ax.axvline(t, color=c, ls=":", label=f"{name} ({t:.3f})")
    ax.set(xlabel="Limiar de decisão", ylabel="Score", title="Trade-off precisão × recall por limiar")
    ax.legend(fontsize=8)
    _save(fig, output_path)


def plot_calibration(score_by_model: dict, y_true, output_path: str, n_bins: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfeita")
    for name, s in score_by_model.items():
        s = np.asarray(s)
        bins = np.unique(np.quantile(s, np.linspace(0, 1, n_bins + 1)))
        b = np.clip(np.digitize(s, bins[1:-1]), 0, len(bins) - 2)
        xs = [s[b == i].mean() for i in range(len(bins) - 1) if (b == i).any()]
        ys = [np.asarray(y_true)[b == i].mean() for i in range(len(bins) - 1) if (b == i).any()]
        ax.plot(xs, ys, marker="o", ms=3, label=name)
    ax.set(xlabel="Probabilidade prevista", ylabel="Fração observada de fraude",
           title="Calibração (escala log)", xscale="log", yscale="log")
    ax.legend(fontsize=7)
    _save(fig, output_path)


def plot_gain_lift(y_true, score, model_name: str, output_path: str) -> None:
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(score), kind="stable")
    cum = np.cumsum(y_true[order]) / max(1, y_true.sum())
    frac = np.arange(1, len(y_true) + 1) / len(y_true)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(frac, cum, label=model_name)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="aleatório")
    axes[0].set(xlabel="Fração de transações revisadas (maior risco primeiro)", ylabel="Fraudes capturadas",
                title="Curva de ganho acumulado")
    axes[0].legend()
    sel = frac <= 0.2
    axes[1].plot(frac[sel], cum[sel] / frac[sel])
    axes[1].axhline(1, c="k", ls="--", lw=1)
    axes[1].set(xlabel="Fração revisada", ylabel="Lift", title="Lift (até 20% revisados)")
    _save(fig, output_path)


def plot_cost_curve(y_true, score, amount, threshold: float, output_path: str,
                    review_cost: float = 5.0, fn_factor: float = 1.0) -> None:
    ths = np.unique(np.quantile(score, np.linspace(0.5, 0.9999, 250)))
    costs = [expected_cost(y_true, (score >= t).astype(int), amount, review_cost, fn_factor)["custo_total"] for t in ths]
    base = expected_cost(y_true, np.zeros(len(score), int), amount, review_cost, fn_factor)["custo_total"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ths, costs, label="Custo total com o modelo")
    ax.axhline(base, c="gray", ls="--", label="Sem modelo (perde todas as fraudes)")
    ax.axvline(threshold, c="red", ls=":", label=f"Limiar escolhido ({threshold:.3f})")
    ax.set(xlabel="Limiar", ylabel="Custo (moeda do dataset)", title="Custo esperado × limiar")
    ax.legend(fontsize=8)
    _save(fig, output_path)


def plot_model_comparison(comparison_df: pd.DataFrame, output_path: str) -> None:
    cols = [c for c in ["pr_auc", "roc_auc", "precisao", "recall", "f1_score", "mcc"] if c in comparison_df.columns]
    df = comparison_df[cols].dropna(how="all")
    fig, ax = plt.subplots(figsize=(11, 5))
    df.plot(kind="bar", ax=ax, width=0.8)
    ax.set(title="Comparação de modelos (teste, limiar escolhido na validação)", ylabel="Score", ylim=(0, 1.05))
    ax.legend(loc="lower right", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _save(fig, output_path)
