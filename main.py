"""
main.py
=======
Pipeline completo de Detecção de Anomalias em Transações (Fraude).

Etapas executadas, na ordem do roadmap:

  01. Primeiros passos
      - Carregamento do dataset
      - EDA (estatísticas, nulos, duplicados, outliers)
      - Feature Engineering (hora do dia, gaps de tempo, frequência, escala)

  02. Balanceamento e avaliação
      - Split treino/teste estratificado
      - Balanceamento (SMOTE / ADASYN / Undersampling) SOMENTE no treino
      - Métricas corretas (Recall, Precisão, F1, PR-AUC, ROC-AUC)

  03. Modelos avançados e explicabilidade
      - Comparação: Regressão Logística, Random Forest, XGBoost, LightGBM,
        CatBoost
      - Modelos não supervisionados: Isolation Forest, One-Class SVM
      - Validação cruzada estratificada (Stratified K-Fold) do melhor modelo
      - Explicabilidade (SHAP) do melhor modelo

Uso:
    python main.py                       # roda tudo com dados sintéticos
    python main.py --csv caminho.csv     # roda tudo com um CSV real
                                          # (precisa ter uma coluna 'Class')

Todos os artefatos (gráficos, tabelas, modelo treinado, relatório) são
salvos em ./outputs/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))

from src import balancing, data_gen, eda, evaluation, explainability, feature_engineering, models

TARGET_COL = "Class"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline de Detecção de Fraude em Transações")
    parser.add_argument("--csv", type=str, default=None, help="Caminho para um CSV real (precisa ter coluna 'Class').")
    parser.add_argument("--n_samples", type=int, default=50_000, help="Nº de transações sintéticas (se --csv não for usado).")
    parser.add_argument("--fraud_ratio", type=float, default=0.015, help="Proporção de fraudes no dataset sintético.")
    parser.add_argument("--balance_strategy", type=str, default="smote", choices=["smote", "adasyn", "undersample", "none"])
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    eda_dir = os.path.join(OUTPUT_DIR, "eda")
    models_dir = os.path.join(OUTPUT_DIR, "modelos")
    reports_dir = os.path.join(OUTPUT_DIR, "relatorios")
    for d in (eda_dir, models_dir, reports_dir):
        os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 01. Carregamento dos dados
    # ------------------------------------------------------------------ #
    log("Carregando dataset...")
    if args.csv:
        df_raw = data_gen.load_dataset(csv_path=args.csv)
    else:
        df_raw = data_gen.load_dataset(
            n_samples=args.n_samples,
            fraud_ratio=args.fraud_ratio,
            random_state=args.random_state,
        )
    log(f"Dataset carregado: {df_raw.shape[0]} linhas, {df_raw.shape[1]} colunas.")

    # ------------------------------------------------------------------ #
    # 01. EDA
    # ------------------------------------------------------------------ #
    log("Executando EDA (estatísticas, nulos, duplicados, outliers, gráficos)...")
    eda_results = eda.run_eda(df_raw, target_col=TARGET_COL, output_dir=eda_dir)
    log(f"  -> Linhas duplicadas encontradas: {eda_results['qualidade_dados']['linhas_duplicadas']}")
    log(f"  -> Total de valores nulos encontrados: {eda_results['qualidade_dados']['total_nulos']}")
    log(f"  -> Balanceamento de classes:\n{eda_results['balanceamento_classes']}")

    df_clean = eda.clean_basic_issues(df_raw, target_col=TARGET_COL)
    log(f"Dataset limpo (sem duplicados/nulos): {df_clean.shape[0]} linhas.")

    # ------------------------------------------------------------------ #
    # 01. Feature Engineering
    # ------------------------------------------------------------------ #
    log("Aplicando engenharia de recursos (hora do dia, gaps, frequência)...")
    df_fe = feature_engineering.engineer_features(df_clean, time_col="Time", amount_col="Amount")

    feature_cols = [c for c in df_fe.columns if c != TARGET_COL]
    numeric_cols_to_scale = [
        c for c in feature_cols
        if df_fe[c].dtype != "object" and not str(df_fe[c].dtype).startswith("category")
    ]
    df_scaled, scaler = feature_engineering.scale_features(
        df_fe, columns=numeric_cols_to_scale, method="robust"
    )
    log(f"Total de features após engenharia: {len(feature_cols)}")

    # ------------------------------------------------------------------ #
    # 02. Split treino/teste (estratificado, preserva a proporção de classes)
    # ------------------------------------------------------------------ #
    X = df_scaled[feature_cols]
    y = df_scaled[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )
    log(f"Split treino/teste: {len(X_train)} treino / {len(X_test)} teste (estratificado).")

    # ------------------------------------------------------------------ #
    # 02. Balanceamento (SOMENTE no treino)
    # ------------------------------------------------------------------ #
    log(f"Balanceando dados de treino com estratégia: {args.balance_strategy}")
    X_train_bal, y_train_bal = balancing.balance_data(
        X_train, y_train, strategy=args.balance_strategy, random_state=args.random_state
    )
    bal_report = balancing.balance_report(y_train, y_train_bal)
    log(f"Distribuição antes/depois do balanceamento:\n{bal_report}")

    # ------------------------------------------------------------------ #
    # 03. Treinamento e comparação de modelos supervisionados
    # ------------------------------------------------------------------ #
    n_neg = (y_train_bal == 0).sum()
    n_pos = max(1, (y_train_bal == 1).sum())
    scale_pos_weight = n_neg / n_pos

    zoo = models.ModelZoo(random_state=args.random_state, contamination=float(y.mean()))
    supervised_models = models.get_supervised_models(zoo, scale_pos_weight=scale_pos_weight)

    results = {}
    trained_models = {}
    log("Treinando e avaliando modelos supervisionados...")
    for name, model in supervised_models.items():
        t0 = time.time()
        model.fit(X_train_bal, y_train_bal)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = evaluation.evaluate_predictions(y_test.values, y_pred, y_proba)
        results[name] = metrics
        trained_models[name] = model

        evaluation.plot_confusion_matrix(
            y_test, y_pred, name, os.path.join(models_dir, f"matriz_confusao_{name}.png")
        )
        evaluation.plot_pr_roc_curves(
            y_test, y_proba, name, os.path.join(models_dir, f"curvas_pr_roc_{name}.png")
        )
        log(f"  -> {name}: {metrics}  ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------ #
    # 03. Modelos não supervisionados (Isolation Forest, One-Class SVM)
    # ------------------------------------------------------------------ #
    log("Treinando modelos não supervisionados de detecção de anomalias...")
    anomaly_models = models.get_anomaly_models(zoo)
    # Não supervisionados treinam sem rótulo (apenas nas features de treino
    # originais, não balanceadas -- balanceamento não faz sentido aqui).
    for name, model in anomaly_models.items():
        t0 = time.time()
        model.fit(X_train)
        raw_pred = model.predict(X_test)
        y_pred_anom = models.anomaly_scores_to_labels(raw_pred)

        metrics = evaluation.evaluate_predictions(y_test.values, y_pred_anom, y_proba=None)
        results[name] = metrics
        log(f"  -> {name}: {metrics}  ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------ #
    # Comparação final dos modelos
    # ------------------------------------------------------------------ #
    comparison_df = evaluation.compare_models(results)
    comparison_df.to_csv(os.path.join(reports_dir, "comparacao_modelos.csv"))
    evaluation.plot_model_comparison(
        comparison_df.dropna(axis=0, how="any"),
        os.path.join(reports_dir, "comparacao_modelos.png"),
    )
    log(f"\nComparação final de modelos:\n{comparison_df}")

    best_model_name = comparison_df.index[0]
    best_model = trained_models.get(best_model_name)
    log(f"Melhor modelo (supervisionado): {best_model_name}")

    # ------------------------------------------------------------------ #
    # 03. Validação cruzada estratificada do melhor modelo supervisionado
    # ------------------------------------------------------------------ #
    if best_model is not None:
        log("Executando validação cruzada estratificada (Stratified K-Fold) do melhor modelo...")
        cv_scores = evaluation.stratified_cross_validation(
            best_model, X_train_bal, y_train_bal, scoring="average_precision", n_splits=5,
            random_state=args.random_state,
        )
        log(f"  -> PR-AUC por fold: {np.round(cv_scores, 4)}")
        log(f"  -> PR-AUC médio: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # -------------------------------------------------------------- #
        # 03. Explicabilidade (SHAP) do melhor modelo
        # -------------------------------------------------------------- #
        log("Calculando explicabilidade (SHAP) do melhor modelo...")
        try:
            shap_values, explain_sample = explainability.compute_shap_values(
                best_model, X_train_bal, X_test
            )
            explainability.plot_shap_summary(
                shap_values, os.path.join(reports_dir, "shap_importancia_global.png")
            )

            # Explica a primeira transação prevista como fraude, se houver.
            preds_on_sample = best_model.predict(explain_sample)
            fraud_positions = np.where(preds_on_sample == 1)[0]
            explain_idx = int(fraud_positions[0]) if len(fraud_positions) > 0 else 0

            explainability.plot_shap_waterfall(
                shap_values, explain_idx, os.path.join(reports_dir, "shap_transacao_individual.png")
            )
            top_feats = explainability.top_features_for_transaction(shap_values, explain_idx)
            top_feats.to_csv(os.path.join(reports_dir, "shap_top_features_transacao.csv"), index=False)
            log(f"  -> Top features para a transação explicada:\n{top_feats}")
        except Exception as exc:
            log(f"  -> Aviso: não foi possível gerar explicabilidade SHAP ({exc}).")

        # Salva o melhor modelo treinado
        joblib.dump(best_model, os.path.join(models_dir, f"melhor_modelo_{best_model_name}.joblib"))
        joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))

    # ------------------------------------------------------------------ #
    # Relatório final em JSON
    # ------------------------------------------------------------------ #
    final_report = {
        "n_transacoes": int(len(df_raw)),
        "proporcao_fraude_original": float(df_raw[TARGET_COL].mean()),
        "estrategia_balanceamento": args.balance_strategy,
        "melhor_modelo": best_model_name,
        "metricas_todos_modelos": results,
    }
    with open(os.path.join(reports_dir, "relatorio_final.json"), "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False, default=float)

    log(f"\nPipeline concluído com sucesso! Artefatos salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
