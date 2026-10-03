"""
main.py  (v2)
=============
Pipeline completo de detecção de fraude em transações.

    python main.py                              # dados sintéticos, tudo padrão
    python main.py --csv creditcard.csv         # dados reais (coluna 'Class')
    python main.py --config configs/default.yaml
    python main.py --balance_strategy smote --benchmark_balancing

Etapas: EDA -> features causais -> split TEMPORAL (treino | validação | teste)
-> modelos supervisionados (Pipelines sem vazamento) -> ensemble -> detectores
de anomalia -> limiar por custo (escolhido na validação) -> IC por bootstrap ->
CV temporal -> drift (PSI) -> explicabilidade -> modelo empacotado + relatórios.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.base import clone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import balancing, data_gen, drift, eda, evaluation as ev, explainability as xai
from src import feature_engineering as fe, models, splits
from src.config import Config, load_config
from src.inference import FraudDetector

VERSION = "2.0.0"
log = logging.getLogger("fraude")
NON_FEATURES = ("Time", "id_cartao")


INT_COLS = {"alertas", "tp", "fp", "fn", "tn", "n_treino", "n_validacao", "fraudes_validacao", "fold"}


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}", index_name: str = "modelo") -> str:
    """Tabela Markdown sem depender de 'tabulate'."""
    d = df.rename_axis(index_name).reset_index()

    def fmt(col, v):
        if pd.isna(v):
            return ""
        if col in INT_COLS:
            return str(int(v))
        return floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)

    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    rows = ["| " + " | ".join(fmt(c, v) for c, v in zip(d.columns, r)) + " |" for r in d.itertuples(index=False)]
    return head + "\n".join(rows) + "\n"


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    raise TypeError(type(o))


def run(cfg: Config) -> dict:
    t_start = time.time()
    np.random.seed(cfg.random_state)
    out = cfg.output_dir
    d_eda, d_mod, d_rep = (os.path.join(out, s) for s in ("eda", "modelos", "relatorios"))
    for d in (d_eda, d_mod, d_rep):
        os.makedirs(d, exist_ok=True)
    target = cfg.target_col

    # ------------------------------------------------------------------ dados
    log.info("Carregando dataset...")
    df_raw = (data_gen.load_dataset(csv_path=cfg.csv) if cfg.csv else
              data_gen.load_dataset(n_samples=cfg.n_samples, fraud_ratio=cfg.fraud_ratio, random_state=cfg.random_state))
    for w in splits.validate_dataset(df_raw, target_col=target):
        log.warning("  ! %s", w)
    has_time, has_amount = "Time" in df_raw.columns, "Amount" in df_raw.columns
    log.info("Dataset: %d linhas x %d colunas | fraudes: %d (%.3f%%)", *df_raw.shape, int(df_raw[target].sum()), 100 * df_raw[target].mean())

    log.info("EDA...")
    eda_res = eda.run_eda(df_raw, target_col=target, output_dir=d_eda)
    q = eda_res["qualidade_dados"]
    log.info("  duplicados: %d | nulos: %d", q["linhas_duplicadas"], q["total_nulos"])
    df_clean = eda.clean_basic_issues(df_raw, target_col=target)
    log.info("Após limpeza: %d linhas.", len(df_clean))

    # --------------------------------------------------------------- features
    split_mode = cfg.split
    if has_time and has_amount:
        card_col = "id_cartao" if "id_cartao" in df_clean.columns else None
        cat_col = "categoria_comerciante" if "categoria_comerciante" in df_clean.columns else None
        df_fe = fe.add_causal_features(df_clean, card_col=card_col, category_col=cat_col)
        log.info("Features causais criadas (%s).", "por cartão" if card_col else "globais — sem id_cartao")
    else:
        df_fe, split_mode = df_clean, "stratified"
        log.warning("Sem 'Time'/'Amount': features temporais desativadas e split estratificado.")
    feature_cols = [c for c in df_fe.columns if c not in (target, *NON_FEATURES)]
    log.info("Total de features: %d", len(feature_cols))

    # ------------------------------------------------------------------ split
    if split_mode == "temporal":
        tr, va, te = splits.temporal_split(df_fe, "Time", cfg.test_size, cfg.val_size)
    else:
        tr, va, te = splits.stratified_split(df_fe, target, cfg.test_size, cfg.val_size, cfg.random_state)
    for name, part in (("treino", tr), ("validação", va), ("teste", te)):
        log.info("  %-10s %6d linhas | %4d fraudes (%.2f%%)", name, len(part), int(part[target].sum()), 100 * part[target].mean())
    X_tr, y_tr = tr[feature_cols], tr[target].to_numpy()
    X_va, y_va = va[feature_cols], va[target].to_numpy()
    X_te, y_te = te[feature_cols], te[target].to_numpy()
    amt_med = float(tr["Amount"].median()) if has_amount else 1.0
    # Sem 'Amount' não há análise de custo: amt_* = None desliga economia/custo nas métricas.
    amt_va = va["Amount"].fillna(amt_med).to_numpy() if has_amount else None
    amt_te = te["Amount"].fillna(amt_med).to_numpy() if has_amount else None
    strategy_thr = cfg.threshold_strategy if has_amount or cfg.threshold_strategy != "cost" else "f1"
    thr_kw = dict(review_cost=cfg.review_cost, fn_factor=cfg.fn_factor)

    def choose_thr(y, s, a):
        return ev.pick_threshold(strategy_thr, y, s, a, recall_target=cfg.recall_target, **thr_kw)

    # ----------------------------------------------------------- supervisionados
    pos_weight = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    zoo = models.ModelZoo(random_state=cfg.random_state, contamination=float(y_tr.mean()), n_jobs=-1)
    pipes = models.get_supervised_pipelines(zoo, pos_weight, cfg.balance_strategy, cfg.balance_ratio, only=cfg.models)
    log.info("Treinando %d modelos (balanceamento: %s)...", len(pipes), cfg.balance_strategy)

    fitted, s_val, s_te, rows, times = {}, {}, {}, {}, {}
    for name, pipe in pipes.items():
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        times[name] = time.time() - t0
        fitted[name] = pipe
        s_val[name], s_te[name] = pipe.predict_proba(X_va)[:, 1], pipe.predict_proba(X_te)[:, 1]
        log.info("  %-22s PR-AUC val=%.4f | %.1fs", name, ev.evaluate_scores(y_va, s_val[name])["pr_auc"], times[name])

    # ---------------------------------------------------------------- ensemble
    ranked = sorted(fitted, key=lambda n: ev.evaluate_scores(y_va, s_val[n])["pr_auc"], reverse=True)
    if len(ranked) >= 2:
        k = min(cfg.ensemble_top_k, len(ranked))
        members = {n: fitted[n] for n in ranked[:k]}
        wts = {n: ev.evaluate_scores(y_va, s_val[n])["pr_auc"] for n in members}
        ens = models.ProbaAverager(members, wts).fit()
        ens_name = f"Ensemble_top{k}"
        fitted[ens_name] = ens
        s_val[ens_name], s_te[ens_name] = ens.predict_proba(X_va)[:, 1], ens.predict_proba(X_te)[:, 1]
        times[ens_name] = 0.0
        log.info("  %-22s PR-AUC val=%.4f | membros: %s", ens_name, ev.evaluate_scores(y_va, s_val[ens_name])["pr_auc"], ", ".join(members))

    # ---------------------------------------------------- detectores de anomalia
    anomaly_names = []
    if cfg.with_anomaly:
        for kind, label in (("isolation_forest", "Isolation_Forest"), ("one_class_svm", "One_Class_SVM")):
            t0 = time.time()
            sc = models.AnomalyScorer(kind, contamination=float(y_tr.mean()), random_state=cfg.random_state).fit(X_tr, y_tr)
            s_val[label], s_te[label] = sc.score(X_va), sc.score(X_te)
            fitted[label], times[label] = sc, time.time() - t0
            anomaly_names.append(label)
            log.info("  %-22s PR-AUC val=%.4f | %.1fs", label, ev.evaluate_scores(y_va, s_val[label], is_proba=False)["pr_auc"], times[label])

    # --------------------------------------------------- avaliação (teste, limiar da validação)
    thresholds, results, cis = {}, {}, {}
    for name in s_val:
        is_p = name not in anomaly_names
        thresholds[name] = choose_thr(y_va, s_val[name], amt_va)
        m_te = ev.evaluate_scores(y_te, s_te[name], thresholds[name], is_p, amt_te, cfg.review_cost, cfg.fn_factor)
        m_va = ev.evaluate_scores(y_va, s_val[name], is_proba=is_p)
        _, lo, hi = ev.bootstrap_ci(y_te, s_te[name], n_boot=cfg.n_bootstrap, random_state=cfg.random_state)
        m_te.update({"pr_auc_val": m_va["pr_auc"], "pr_auc_ic95_inf": lo, "pr_auc_ic95_sup": hi, "tempo_treino_s": times[name]})
        results[name] = m_te
    comp = pd.DataFrame(results).T.sort_values("pr_auc_val", ascending=False)
    cols_show = ["pr_auc_val", "pr_auc", "pr_auc_ic95_inf", "pr_auc_ic95_sup", "roc_auc", "brier", "precisao", "recall", "f1_score",
                 "mcc", "fpr", "limiar", "alertas", "fraude_perdida", "economia", "economia_pct", "precisao_top1pct", "recall_top1pct", "tempo_treino_s"]
    comp = comp[[c for c in cols_show if c in comp.columns]]
    comp.to_csv(os.path.join(d_rep, "comparacao_modelos.csv"))
    ev.plot_model_comparison(comp, os.path.join(d_rep, "comparacao_modelos.png"))

    sup_names = [n for n in comp.index if n not in anomaly_names]
    best_name = sup_names[0]  # escolhido por PR-AUC na VALIDAÇÃO (não no teste)
    singles = [n for n in sup_names if not n.startswith("Ensemble")]
    best_single = singles[0]
    best = fitted[best_name]
    thr = thresholds[best_name]
    log.info("Melhor modelo (por PR-AUC na validação): %s | limiar (%s) = %.4f", best_name, strategy_thr, thr)
    b = results[best_name]
    econ = f" | economia {100 * b['economia_pct']:.1f}%" if "economia_pct" in b else ""
    log.info("  TESTE -> PR-AUC %.4f [%.4f, %.4f] | precisão %.3f | recall %.3f%s",
             b["pr_auc"], b["pr_auc_ic95_inf"], b["pr_auc_ic95_sup"], b["precisao"], b["recall"], econ)

    # ------------------------------------------------------------------ gráficos
    for name in s_te:
        ev.plot_confusion_matrix(y_te, (s_te[name] >= thresholds[name]).astype(int), name, os.path.join(d_mod, f"matriz_confusao_{name}.png"))
    ev.plot_pr_roc_overlay({n: (y_te, s_te[n]) for n in comp.index}, os.path.join(d_mod, "curvas_pr_roc_todos.png"), "todos os modelos (teste)")
    ev.plot_pr_roc_curves(y_te, s_te[best_name], best_name, os.path.join(d_mod, f"curvas_pr_roc_{best_name}.png"))
    ev.plot_calibration({n: s_te[n] for n in sup_names}, y_te, os.path.join(d_mod, "calibracao.png"))
    ev.plot_gain_lift(y_te, s_te[best_name], best_name, os.path.join(d_mod, "ganho_lift.png"))

    # ------------------------------------------------- estratégias de limiar (melhor)
    thr_rows = {}
    for strat in ("default", "f1", "f2", "recall", "cost"):
        if strat == "cost" and not has_amount:
            continue
        t = ev.pick_threshold(strat, y_va, s_val[best_name], amt_va, recall_target=cfg.recall_target, **thr_kw)
        mm = ev.evaluate_scores(y_te, s_te[best_name], t, True, amt_te, cfg.review_cost, cfg.fn_factor)
        thr_rows[strat] = {k: mm.get(k) for k in ("limiar", "precisao", "recall", "f1_score", "mcc", "alertas", "fraude_perdida", "economia_pct")}
    thr_df = pd.DataFrame(thr_rows).T
    thr_df.to_csv(os.path.join(d_rep, "estrategias_limiar.csv"))
    ev.plot_threshold_analysis(y_va, s_val[best_name], {k: v["limiar"] for k, v in thr_rows.items() if k != "default"}, os.path.join(d_mod, "analise_limiar.png"))
    if has_amount:
        ev.plot_cost_curve(y_te, s_te[best_name], amt_te, thr, os.path.join(d_mod, "curva_custo.png"), cfg.review_cost, cfg.fn_factor)

    # ----------------------------------------------------------- validação cruzada
    cv_df = pd.DataFrame()
    X_trval, y_trval = pd.concat([X_tr, X_va]), np.concatenate([y_tr, y_va])
    log.info("Validação cruzada (%s) de %s...", "temporal" if split_mode == "temporal" else "estratificada", best_single)
    try:
        if split_mode == "temporal":
            cv_df = ev.temporal_cross_validation(clone(pipes[best_single]), X_trval.reset_index(drop=True), y_trval, cfg.cv_splits)
        else:
            sc = ev.stratified_cross_validation(clone(pipes[best_single]), X_trval, y_trval, n_splits=5, random_state=cfg.random_state)
            cv_df = pd.DataFrame({"fold": range(1, len(sc) + 1), "pr_auc": sc})
        if len(cv_df):
            log.info("  PR-AUC por fold: %s | média %.4f ± %.4f", np.round(cv_df["pr_auc"].to_numpy(), 4), cv_df["pr_auc"].mean(), cv_df["pr_auc"].std())
            cv_df.to_csv(os.path.join(d_rep, "validacao_cruzada.csv"), index=False)
    except Exception as exc:  # pragma: no cover
        log.warning("  CV não executada: %s", exc)

    # ------------------------------------------- benchmark de balanceamento (opcional)
    bench_df = pd.DataFrame()
    if cfg.benchmark_balancing:
        log.info("Benchmark de estratégias de balanceamento com %s...", best_single)
        strategies = ["none", "class_weight", "smote", "undersample"] + (["adasyn"] if models.optional_import("imblearn") else [])
        brow = {}
        for st in strategies:
            p = models.get_supervised_pipelines(zoo, pos_weight, st, cfg.balance_ratio, only=[best_single])[best_single]
            t0 = time.time()
            p.fit(X_tr, y_tr)
            sv, stt = p.predict_proba(X_va)[:, 1], p.predict_proba(X_te)[:, 1]
            t_ = choose_thr(y_va, sv, amt_va)
            mm = ev.evaluate_scores(y_te, stt, t_, True, amt_te, cfg.review_cost, cfg.fn_factor)
            brow[st] = {k: mm.get(k) for k in ("pr_auc", "roc_auc", "brier", "precisao", "recall", "f1_score", "alertas", "economia_pct")}
            brow[st]["tempo_s"] = time.time() - t0
            log.info("  %-13s PR-AUC=%.4f | economia=%.1f%% | Brier=%.5f", st, mm["pr_auc"], 100 * mm.get("economia_pct", float("nan")), mm["brier"])
        bench_df = pd.DataFrame(brow).T
        bench_df.to_csv(os.path.join(d_rep, "benchmark_balanceamento.csv"))

    # ----------------------------------------------------------------- drift
    drift_df = drift.drift_report(X_tr, X_te)
    drift_df.to_csv(os.path.join(d_rep, "drift_psi_features.csv"), index=False)
    score_psi = drift.psi(s_val[best_name], s_te[best_name])
    log.info("Drift: %d features com PSI > 0,25 | PSI do score (val -> teste) = %.3f (%s)",
             int((drift_df["psi"] > 0.25).sum()), score_psi, drift.classify_psi(score_psi))

    # ------------------------------------------------------- explicabilidade
    imp_df, local_df = pd.DataFrame(), pd.DataFrame()
    if cfg.explain:
        log.info("Explicabilidade (%s)...", "SHAP" if xai.has_shap() else "permutação + oclusão; shap não instalado")
        try:
            if xai.has_shap():
                sv, sample = xai.compute_shap_values(fitted[best_single], X_tr, X_te)
                xai.plot_shap_summary(sv, os.path.join(d_rep, "shap_importancia_global.png"))
                flagged = np.where(fitted[best_single].predict_proba(X_te.loc[sample.index])[:, 1] >= thresholds[best_single])[0]
                i = int(flagged[0]) if len(flagged) else 0
                xai.plot_shap_waterfall(sv, i, os.path.join(d_rep, "shap_transacao_individual.png"))
                xai.top_features_for_transaction(sv, i).to_csv(os.path.join(d_rep, "shap_top_features_transacao.csv"), index=False)
            imp_df = xai.global_importance(fitted[best_single], X_va, y_va, random_state=cfg.random_state)
            imp_df.to_csv(os.path.join(d_rep, "importancia_permutacao.csv"), index=False)
            xai.plot_importance(imp_df, os.path.join(d_rep, "importancia_permutacao.png"))
            cand = np.where((y_te == 1) & (s_te[best_single] >= thresholds[best_single]))[0]
            # Fraude detectada de score MEDIANO (representativa; a mais extrema seria a menos instrutiva).
            j = int(cand[np.argsort(s_te[best_single][cand])[len(cand) // 2]]) if len(cand) else int(np.argmax(s_te[best_single]))
            local_df = xai.local_explanation(fitted[best_single], X_te.iloc[[j]], X_tr)
            local_df.to_csv(os.path.join(d_rep, "explicacao_transacao_individual.csv"), index=False)
            xai.plot_local_explanation(local_df, os.path.join(d_rep, "explicacao_transacao_individual.png"))
        except Exception as exc:  # pragma: no cover
            log.warning("  Explicabilidade falhou: %s", exc)

    # ---------------------------------------------------------- empacotamento
    detector = FraudDetector(
        pipeline=best, threshold=float(thr), feature_columns=feature_cols, engineer=bool(has_time and has_amount),
        card_col="id_cartao" if "id_cartao" in df_clean.columns else None,
        meta={"versao": VERSION, "modelo": best_name, "estrategia_limiar": strategy_thr,
              "criado_em": time.strftime("%Y-%m-%d %H:%M:%S"), "n_treino": int(len(tr)),
              "metricas_teste": {k: results[best_name].get(k) for k in ("pr_auc", "precisao", "recall", "f1_score", "economia_pct")},
              "config": cfg.to_dict()})
    detector.save(os.path.join(d_mod, "modelo_fraude.joblib"))

    final = {
        "versao": VERSION, "n_transacoes": int(len(df_raw)), "proporcao_fraude_original": float(df_raw[target].mean()),
        "split": split_mode, "estrategia_balanceamento": cfg.balance_strategy, "estrategia_limiar": strategy_thr,
        "melhor_modelo": best_name, "melhor_modelo_individual": best_single, "limiar": float(thr), "criterio_selecao": "PR-AUC na validação",
        "psi_score_val_teste": score_psi, "config": cfg.to_dict(),
        "metricas_todos_modelos": comp.to_dict(orient="index"),
        "estrategias_limiar_melhor_modelo": thr_df.to_dict(orient="index"),
        "validacao_cruzada": cv_df.to_dict(orient="records"),
        "tempo_total_s": time.time() - t_start,
    }
    with open(os.path.join(d_rep, "relatorio_final.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False, default=_jsonable)

    write_markdown_report(os.path.join(d_rep, "relatorio.md"), cfg, final, comp, thr_df, cv_df, bench_df, drift_df, imp_df, local_df, (tr, va, te), target)
    log.info("Pipeline concluído em %.1fs. Artefatos em: %s", time.time() - t_start, os.path.abspath(out))
    return {"final": final, "comparison": comp, "detector": detector, "best_name": best_name}


def write_markdown_report(path, cfg, final, comp, thr_df, cv_df, bench_df, drift_df, imp_df, local_df, parts, target):
    tr, va, te = parts
    b = comp.loc[final["melhor_modelo"]]
    L = [f"# Relatório — Detecção de Fraude (v{VERSION})\n",
         f"Gerado em {time.strftime('%Y-%m-%d %H:%M')} | split **{final['split']}** | balanceamento **{cfg.balance_strategy}** | limiar **{final['estrategia_limiar']}**\n",
         "## Dados\n",
         f"- {final['n_transacoes']:,} transações, {100 * final['proporcao_fraude_original']:.2f}% fraude.",
         f"- Treino {len(tr):,} ({int(tr[target].sum())} fraudes) · Validação {len(va):,} ({int(va[target].sum())}) · Teste {len(te):,} ({int(te[target].sum())}).\n",
         "## Resultado principal\n",
         f"Modelo escolhido pela **validação** (PR-AUC): **{final['melhor_modelo']}**, limiar {final['limiar']:.4f}.\n",
         f"No **teste** (dados que não influenciaram nenhuma decisão): PR-AUC **{b['pr_auc']:.4f}** (IC95% {b['pr_auc_ic95_inf']:.4f}–{b['pr_auc_ic95_sup']:.4f}), "
         f"precisão {b['precisao']:.3f}, recall {b['recall']:.3f}, {int(b['alertas'])} alertas"
         + (f", economia estimada de **{100 * b['economia_pct']:.1f}%** vs. não ter modelo (custo de revisão {cfg.review_cost:g}/alerta)." if 'economia_pct' in b and not pd.isna(b['economia_pct']) else ".") + "\n",
         "## Comparação de modelos (teste; limiar escolhido na validação)\n", md_table(comp.drop(columns=["tempo_treino_s"], errors="ignore")),
         "## Estratégias de limiar (modelo escolhido)\n", md_table(thr_df, index_name="estrategia")]
    if len(cv_df):
        L += ["## Validação cruzada\n", md_table(cv_df.set_index("fold"), index_name="fold"), f"Média PR-AUC: {cv_df['pr_auc'].mean():.4f} ± {cv_df['pr_auc'].std():.4f}\n"]
    if len(bench_df):
        L += ["## Benchmark de balanceamento\n", md_table(bench_df, index_name="estrategia")]
    L += ["## Drift (PSI treino → teste) — top 8\n", md_table(drift_df.head(8).set_index("feature"), index_name="feature")]
    if len(imp_df):
        L += ["## Importância global (permutação) — top 10\n", md_table(imp_df.head(10).set_index("feature"), index_name="feature")]
    if len(local_df):
        L += [f"## Explicação de uma transação sinalizada (modelo individual {final.get('melhor_modelo_individual')}, prob. {local_df.attrs.get('prob_base', float('nan')):.2f})\n",
              "Efeito em log-odds ao trocar cada variável pelo valor típico (positivo = empurra para fraude).\n", md_table(local_df.set_index("feature"), index_name="feature")]
    L += ["## Ressalvas\n",
          "- Com dados sintéticos os números servem para validar o pipeline, **não** para prever desempenho em produção.",
          "- Poucos eventos de fraude no teste → olhe o intervalo de confiança, não só a estimativa pontual.",
          "- Custos (`review_cost`, `fn_factor`) são parâmetros de negócio: ajuste-os à sua realidade antes de usar o limiar.\n"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def parse_args():
    p = argparse.ArgumentParser(description="Pipeline de Detecção de Fraude em Transações (v2)")
    p.add_argument("--config", default=None, help="YAML de configuração (CLI sobrescreve o YAML).")
    p.add_argument("--csv", default=None)
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--fraud_ratio", type=float, default=None)
    p.add_argument("--split", choices=["temporal", "stratified"], default=None)
    p.add_argument("--balance_strategy", choices=list(balancing.STRATEGIES), default=None)
    p.add_argument("--benchmark_balancing", action="store_true", default=None)
    p.add_argument("--models", nargs="+", default=None, help="Subconjunto de modelos (ex.: LightGBM Random_Forest).")
    p.add_argument("--threshold_strategy", choices=["cost", "f1", "f2", "recall", "default"], default=None)
    p.add_argument("--review_cost", type=float, default=None)
    p.add_argument("--fn_factor", type=float, default=None)
    p.add_argument("--no_anomaly", dest="with_anomaly", action="store_false", default=None)
    p.add_argument("--no_explain", dest="explain", action="store_false", default=None)
    p.add_argument("--test_size", type=float, default=None)
    p.add_argument("--n_bootstrap", type=int, default=None, help="Reamostragens do IC por bootstrap.")
    p.add_argument("--cv_splits", type=int, default=None)
    p.add_argument("--random_state", type=int, default=None)
    p.add_argument("--output_dir", default=None)
    a = vars(p.parse_args())
    return load_config(a.pop("config"), **a)


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    run(parse_args())


if __name__ == "__main__":
    main()
