"""
evaluation.py
-------------
Etapa 02/03 do roadmap: Métricas de Avaliação Corretas + Validação Cruzada.

- NUNCA usar acurácia pura como métrica principal em fraude.
- Foco em: Recall, Precisão, F1-Score, PR-AUC e ROC-AUC.
- Validação cruzada estratificada (StratifiedKFold) para preservar a
  proporção de classes em cada fold.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
) -> dict:
    """Calcula o conjunto completo de métricas apropriadas para dados
    desbalanceados. `y_proba` é a probabilidade/score da classe positiva
    (fraude), usada para PR-AUC e ROC-AUC.
    """
    metrics = {
        "precisao": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)
    return metrics


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """Recebe {nome_modelo: {metrica: valor}} e retorna uma tabela comparativa
    ordenada pela métrica mais relevante para fraude: PR-AUC (se existir),
    senão recall.
    """
    df = pd.DataFrame(results).T
    sort_col = "pr_auc" if "pr_auc" in df.columns else "recall"
    return df.sort_values(sort_col, ascending=False)


def stratified_cross_validation(
    model, X, y, scoring: str = "average_precision", n_splits: int = 5, random_state: int = 42
) -> np.ndarray:
    """Roda validação cruzada estratificada (StratifiedKFold), preservando
    a proporção de classes em cada fold -- essencial para dados
    desbalanceados. Por padrão usa 'average_precision' (equivalente ao PR-AUC).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    return scores


def plot_confusion_matrix(y_true, y_pred, model_name: str, output_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Legítima", "Fraude"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Matriz de confusão - {model_name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_pr_roc_curves(y_true, y_proba, model_name: str, output_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    axes[0].plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precisão")
    axes[0].set_title(f"Curva Precision-Recall - {model_name}")
    axes[0].legend()

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    axes[1].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].set_xlabel("Falso Positivo (FPR)")
    axes[1].set_ylabel("Verdadeiro Positivo (TPR)")
    axes[1].set_title(f"Curva ROC - {model_name}")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_model_comparison(comparison_df: pd.DataFrame, output_path: str) -> None:
    metrics_to_plot = [c for c in ["precisao", "recall", "f1_score", "pr_auc", "roc_auc"] if c in comparison_df.columns]
    fig, ax = plt.subplots(figsize=(9, 5))
    comparison_df[metrics_to_plot].plot(kind="bar", ax=ax)
    ax.set_title("Comparação de modelos - métricas apropriadas para dados desbalanceados")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
