"""
explainability.py
------------------
Etapa 03 do roadmap: Explicabilidade do Modelo (XAI).

Usa SHAP (SHapley Additive exPlanations) para tornar as decisões do
melhor modelo auditáveis e transparentes:

- Importância global das variáveis (summary plot).
- Explicação individual de uma transação específica (force/waterfall plot),
  fundamental para a equipe de negócio entender "por que esta transação
  específica foi marcada como fraude".
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def compute_shap_values(model, X_background: pd.DataFrame, X_explain: pd.DataFrame | None = None):
    """Calcula os valores SHAP para o modelo treinado.

    Usa shap.Explainer, que seleciona automaticamente o algoritmo mais
    eficiente disponível (TreeExplainer para modelos baseados em árvore,
    fallback para KernelExplainer/PermutationExplainer em outros casos).
    """
    if X_explain is None:
        X_explain = X_background

    # Amostragem para manter o cálculo rápido em datasets grandes.
    background_sample = X_background.sample(
        n=min(200, len(X_background)), random_state=42
    )
    explain_sample = X_explain.sample(
        n=min(500, len(X_explain)), random_state=42
    )

    try:
        explainer = shap.Explainer(model, background_sample)
        shap_values = explainer(explain_sample)
    except Exception:
        # Fallback genérico e mais lento, mas funciona para qualquer modelo
        # com predict_proba.
        explainer = shap.KernelExplainer(
            lambda data: model.predict_proba(data)[:, 1], background_sample
        )
        raw_values = explainer.shap_values(explain_sample, nsamples=100)
        shap_values = shap.Explanation(
            values=raw_values,
            base_values=np.full(len(explain_sample), explainer.expected_value),
            data=explain_sample.values,
            feature_names=list(explain_sample.columns),
        )

    shap_values = _select_positive_class(shap_values)
    return shap_values, explain_sample


def _select_positive_class(shap_values):
    """Alguns modelos (ex: RandomForestClassifier) retornam valores SHAP com
    uma dimensão extra para cada classe: shape (n_amostras, n_features, n_classes).
    Para classificação binária, sempre selecionamos a classe positiva (fraude,
    índice 1) para manter a interface consistente entre todos os modelos.
    """
    values = getattr(shap_values, "values", None)
    if values is not None and values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


def plot_shap_summary(shap_values, output_path: str) -> None:
    """Gráfico de importância global das variáveis (quais features mais
    influenciam, em média, a decisão do modelo)."""
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_shap_waterfall(shap_values, index: int, output_path: str) -> None:
    """Explica UMA transação específica: mostra quais variáveis empurraram
    a previsão para 'fraude' ou para 'legítima', e o quanto cada uma pesou.
    """
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(shap_values[index], show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def top_features_for_transaction(shap_values, index: int, top_n: int = 5) -> pd.DataFrame:
    """Retorna, em formato tabular, as variáveis que mais contribuíram
    (positiva ou negativamente) para a classificação de uma transação
    específica -- útil para gerar um relatório textual automático para a
    equipe de negócio (ex: 'Esta transação foi marcada como fraude
    principalmente por causa de X e Y').
    """
    values = shap_values[index].values
    features = shap_values[index].feature_names
    df = pd.DataFrame({"feature": features, "shap_value": values})
    df["abs_shap"] = df["shap_value"].abs()
    return df.sort_values("abs_shap", ascending=False).head(top_n).drop(columns="abs_shap")
