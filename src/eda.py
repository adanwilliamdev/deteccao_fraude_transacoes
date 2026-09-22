"""
eda.py
------
Etapa 01 do roadmap: Análise Exploratória dos Dados.

- Estatísticas descritivas
- Verificação de nulos, duplicados e outliers
- Visualizações salvas em disco (não usa plt.show(), pois roda em script)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna estatísticas descritivas (média, mediana, desvio padrão etc.)."""
    return df.describe().T


def check_data_quality(df: pd.DataFrame) -> dict:
    """Verifica nulos, duplicados e proporção de outliers (via IQR) em cada
    coluna numérica.
    """
    report = {
        "n_linhas": len(df),
        "n_colunas": df.shape[1],
        "nulos_por_coluna": df.isnull().sum().to_dict(),
        "total_nulos": int(df.isnull().sum().sum()),
        "linhas_duplicadas": int(df.duplicated().sum()),
    }

    outlier_report = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col == "Class":
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_report[col] = int(n_outliers)
    report["outliers_por_coluna_iqr"] = outlier_report
    return report


def class_balance(df: pd.DataFrame, target_col: str = "Class") -> pd.Series:
    """Retorna a contagem e o percentual de cada classe."""
    counts = df[target_col].value_counts()
    pct = df[target_col].value_counts(normalize=True) * 100
    summary = pd.DataFrame({"contagem": counts, "percentual": pct.round(3)})
    return summary


def clean_basic_issues(df: pd.DataFrame, target_col: str = "Class") -> pd.DataFrame:
    """Remove duplicados exatos e imputa nulos numéricos pela mediana da coluna.
    Retorna um dataframe limpo, pronto para a etapa de feature engineering.
    """
    df_clean = df.drop_duplicates().copy()

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col == target_col:
            continue
        if df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)

    return df_clean.reset_index(drop=True)


def plot_class_balance(df: pd.DataFrame, target_col: str, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=target_col, data=df, ax=ax, hue=target_col, legend=False)
    ax.set_yscale("log")
    ax.set_title("Distribuição das classes (escala log)")
    ax.set_xlabel("Classe (0 = legítima, 1 = fraude)")
    ax.set_ylabel("Contagem (log)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_amount_distribution(df: pd.DataFrame, target_col: str, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.kdeplot(
        data=df, x="Amount", hue=target_col, common_norm=False, fill=True, ax=ax
    )
    ax.set_xlim(0, df["Amount"].quantile(0.99))
    ax.set_title("Distribuição do valor da transação por classe")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, output_path: str) -> None:
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title("Matriz de correlação")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def run_eda(df: pd.DataFrame, target_col: str, output_dir: str) -> dict:
    """Executa a EDA completa e salva os gráficos em `output_dir`.
    Retorna um dicionário com os principais achados (para logging/relatório).
    """
    os.makedirs(output_dir, exist_ok=True)

    stats = descriptive_stats(df)
    quality = check_data_quality(df)
    balance = class_balance(df, target_col)

    stats.to_csv(os.path.join(output_dir, "estatisticas_descritivas.csv"))
    balance.to_csv(os.path.join(output_dir, "balanceamento_classes.csv"))

    plot_class_balance(df, target_col, os.path.join(output_dir, "01_balanco_classes.png"))
    plot_amount_distribution(df, target_col, os.path.join(output_dir, "02_distribuicao_valor.png"))
    plot_correlation_heatmap(df, os.path.join(output_dir, "03_correlacao.png"))

    return {
        "estatisticas_descritivas": stats,
        "qualidade_dados": quality,
        "balanceamento_classes": balance,
    }
