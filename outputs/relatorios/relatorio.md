# Relatório — Detecção de Fraude (v2.0.0)

Gerado em 2026-10-03 15:06 | split **temporal** | balanceamento **class_weight** | limiar **cost**

## Dados

- 60,300 transações, 1.50% fraude.
- Treino 36,003 (530 fraudes) · Validação 9,000 (140) · Teste 15,000 (230).

## Resultado principal

Modelo escolhido pela **validação** (PR-AUC): **Ensemble_top3**, limiar 0.5057.

No **teste** (dados que não influenciaram nenhuma decisão): PR-AUC **0.8547** (IC95% 0.8107–0.8911), precisão 0.644, recall 0.835, 298 alertas, economia estimada de **88.2%** vs. não ter modelo (custo de revisão 5/alerta).

## Comparação de modelos (teste; limiar escolhido na validação)

| modelo | pr_auc_val | pr_auc | pr_auc_ic95_inf | pr_auc_ic95_sup | roc_auc | brier | precisao | recall | f1_score | mcc | fpr | limiar | alertas | fraude_perdida | economia | economia_pct | precisao_top1pct | recall_top1pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ensemble_top3 | 0.8417 | 0.8547 | 0.8107 | 0.8911 | 0.9741 | 0.0124 | 0.6443 | 0.8348 | 0.7273 | 0.7288 | 0.0072 | 0.5057 | 298 | 3458.4408 | 37151.5893 | 0.8825 | 0.9867 | 0.6435 |
| Regressao_Logistica | 0.8312 | 0.8513 | 0.8075 | 0.8914 | 0.9708 | 0.0427 | 0.6763 | 0.8174 | 0.7402 | 0.7391 | 0.0061 | 0.8976 | 278 | 2790.8087 | 37919.2215 | 0.9007 | 1.0000 | 0.6522 |
| HistGradientBoosting | 0.8122 | 0.8435 | 0.7980 | 0.8798 | 0.9795 | 0.0096 | 0.4778 | 0.8435 | 0.6101 | 0.6278 | 0.0144 | 0.3728 | 406 | 3417.2805 | 36652.7497 | 0.8706 | 0.9867 | 0.6435 |
| Random_Forest | 0.8050 | 0.8345 | 0.7922 | 0.8748 | 0.9715 | 0.0057 | 0.4898 | 0.8391 | 0.6186 | 0.6343 | 0.0136 | 0.1704 | 394 | 3300.7026 | 36829.3276 | 0.8748 | 0.9800 | 0.6391 |
| Extra_Trees | 0.7893 | 0.8265 | 0.7762 | 0.8664 | 0.9708 | 0.0067 | 0.5975 | 0.8261 | 0.6934 | 0.6973 | 0.0087 | 0.2410 | 318 | 3485.5735 | 37024.4566 | 0.8794 | 0.9467 | 0.6174 |
| One_Class_SVM | 0.5668 | 0.6479 | 0.5828 | 0.7068 | 0.9228 |  | 0.2719 | 0.8087 | 0.4070 | 0.4565 | 0.0337 | -0.2818 | 684 | 4542.9448 | 34137.0853 | 0.8109 | 0.7333 | 0.4783 |
| Isolation_Forest | 0.5355 | 0.5999 | 0.5400 | 0.6544 | 0.9491 |  | 0.2938 | 0.7652 | 0.4246 | 0.4622 | 0.0286 | -0.0205 | 599 | 3692.9319 | 35412.0982 | 0.8411 | 0.6800 | 0.4435 |

## Estratégias de limiar (modelo escolhido)

| estrategia | limiar | precisao | recall | f1_score | mcc | alertas | fraude_perdida | economia_pct |
|---|---|---|---|---|---|---|---|---|
| default | 0.5000 | 0.6358 | 0.8348 | 0.7218 | 0.7238 | 302 | 3458.4408 | 0.8820 |
| f1 | 0.7281 | 0.9239 | 0.7391 | 0.8213 | 0.8240 | 184 | 5237.3544 | 0.8537 |
| f2 | 0.5715 | 0.7560 | 0.8217 | 0.7875 | 0.7848 | 250 | 3698.9579 | 0.8824 |
| recall | 0.5770 | 0.7590 | 0.8217 | 0.7891 | 0.7864 | 249 | 3698.9579 | 0.8826 |
| cost | 0.5057 | 0.6443 | 0.8348 | 0.7273 | 0.7288 | 298 | 3458.4408 | 0.8825 |

## Validação cruzada

| fold | n_treino | n_validacao | fraudes_validacao | pr_auc | roc_auc |
|---|---|---|---|---|---|
| 1 | 9003 | 9000 | 133 | 0.8265 | 0.9573 |
| 2 | 18003 | 9000 | 128 | 0.7463 | 0.9632 |
| 3 | 27003 | 9000 | 159 | 0.8210 | 0.9589 |
| 4 | 36003 | 9000 | 140 | 0.8312 | 0.9730 |

Média PR-AUC: 0.8063 ± 0.0402

## Benchmark de balanceamento

| estrategia | pr_auc | roc_auc | brier | precisao | recall | f1_score | alertas | economia_pct | tempo_s |
|---|---|---|---|---|---|---|---|---|---|
| none | 0.8566 | 0.9730 | 0.0038 | 0.5643 | 0.8391 | 0.6748 | 342 | 0.9003 | 0.7584 |
| class_weight | 0.8513 | 0.9708 | 0.0427 | 0.6763 | 0.8174 | 0.7402 | 278 | 0.9007 | 1.7791 |
| smote | 0.8511 | 0.9707 | 0.0426 | 0.4816 | 0.8522 | 0.6154 | 407 | 0.8956 | 2.7233 |
| undersample | 0.8424 | 0.9707 | 0.0440 | 0.7093 | 0.7957 | 0.7500 | 258 | 0.8901 | 0.3514 |

## Drift (PSI treino → teste) — top 8

| feature | psi | status |
|---|---|---|
| cartao_seg_desde_anterior | 0.1015 | atenção |
| valor_vs_media_historica | 0.0595 | estável |
| cartao_valor_medio_historico | 0.0500 | estável |
| hora_do_dia | 0.0044 | estável |
| hora_cos | 0.0039 | estável |
| hora_sin | 0.0034 | estável |
| distancia_casa_km | 0.0014 | estável |
| V10 | 0.0014 | estável |

## Importância global (permutação) — top 10

| feature | queda_pr_auc | desvio |
|---|---|---|
| distancia_casa_km | 0.5402 | 0.0082 |
| cartao_qtd_1h | 0.1635 | 0.0078 |
| V10 | 0.0510 | 0.0043 |
| V3 | 0.0461 | 0.0051 |
| V14 | 0.0342 | 0.0013 |
| V7 | 0.0231 | 0.0036 |
| cartao_valor_soma_1h | 0.0119 | 0.0019 |
| cartao_valor_soma_24h | 0.0092 | 0.0006 |
| hora_cos | 0.0065 | 0.0024 |
| cartao_qtd_24h | 0.0060 | 0.0001 |

## Explicação de uma transação sinalizada (modelo individual Regressao_Logistica, prob. 1.00)

Efeito em log-odds ao trocar cada variável pelo valor típico (positivo = empurra para fraude).

| feature | valor | valor_tipico | efeito_logit |
|---|---|---|---|
| distancia_casa_km | 82.3673 | 4.2556 | 7.4658 |
| cartao_valor_soma_1h | 1097.0709 | 0.0000 | 5.1807 |
| cartao_qtd_1h | 2.0000 | 0.0000 | 4.5020 |
| cartao_valor_soma_24h | 1097.0709 | 12.4569 | -2.3115 |
| V14 | 2.4989 | 0.0186 | 1.7930 |
| V7 | 3.8666 | 0.0170 | 1.7864 |

## Ressalvas

- Com dados sintéticos os números servem para validar o pipeline, **não** para prever desempenho em produção.
- Poucos eventos de fraude no teste → olhe o intervalo de confiança, não só a estimativa pontual.
- Custos (`review_cost`, `fn_factor`) são parâmetros de negócio: ajuste-os à sua realidade antes de usar o limiar.
