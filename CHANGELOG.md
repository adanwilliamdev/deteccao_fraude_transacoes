# Changelog

## 2.0.0

### Correções metodológicas (afetam a validade dos resultados)
- **Vazamento do scaler**: o `RobustScaler` era ajustado no dataset inteiro antes do split. Agora o pré-processamento (`TabularPreprocessor`) é ajustado só no treino, dentro do Pipeline.
- **Vazamento na validação cruzada**: SMOTE rodava antes da CV, deixando amostras sintéticas nos folds de validação. O reamostrador agora é parte do estimador (`ResampledClassifier`) e só atua no `fit`.
- **Seleção do melhor modelo no teste**: a v1 escolhia o modelo pelo PR-AUC do teste. Agora a escolha, os pesos do ensemble e o limiar usam a **validação**; o teste só reporta.
- **Imputação/estatísticas globais**: mediana de nulos e z-score do valor usavam o dataset inteiro (inclusive o futuro). Imputação agora é causal/no pipeline; `amount_zscore_global` (transformação monótona sem ganho) foi removido.
- **Split temporal** por padrão (treino no passado, teste no futuro); o estratificado aleatório continua disponível (`--split stratified`).
- **Features causais**: cada linha usa só o passado; coberto por teste automatizado (`test_no_future_leakage`).
- Removido o contador de histórico por cartão: monotônico no tempo, logo não estacionário (PSI > 3 entre treino e teste).

### Novidades
- Features de comportamento por cartão: velocidade (1h/24h), tempo desde a transação anterior, valor vs. média histórica, categoria nova; hora cíclica (seno/cosseno).
- Gerador sintético com cartões, canal, categoria, distância e fraudes em rajada (incl. ~30% "furtivas").
- Limiar de decisão otimizado (custo esperado, F1, F2, recall-alvo) na validação; custo = revisão por alerta + valor da fraude perdida.
- Métricas: MCC, acurácia balanceada, FPR, Brier, precisão/recall no top-1%, recall a FPR fixo; IC95% por bootstrap estratificado.
- CV temporal (janela expansiva); benchmark de estratégias de balanceamento (`--benchmark_balancing`).
- Ensemble por média ponderada de probabilidades; Extra Trees e HistGradientBoosting (sempre disponíveis); XGBoost/LightGBM/CatBoost/SHAP/ADASYN viraram opcionais.
- Anomalias: Isolation Forest / One-Class SVM treinados só com legítimas e avaliados por score contínuo (PR-AUC).
- Explicabilidade sem SHAP (permutação + oclusão em log-odds); SHAP continua suportado.
- Gráficos: PR/ROC sobrepostos, análise de limiar, calibração, ganho/lift, curva de custo, fraude por hora/canal/categoria.
- Monitoramento de drift (PSI) por feature e do score.
- Modelo empacotado em um único `.joblib` (pipeline + limiar + features + metadados); `predict.py` (CLI) e `api.py` (FastAPI).
- Configuração por YAML + CLI; relatório Markdown automático; 21 testes; Makefile e Dockerfile.

### Verificação
Os testes e a execução completa foram feitos com o núcleo (scikit-learn). Os caminhos que dependem de bibliotecas opcionais (XGBoost, LightGBM, CatBoost, SHAP, ADASYN, FastAPI) e o Dockerfile não puderam ser executados no ambiente de desenvolvimento (sem acesso à rede). Rode `make test` e `python main.py --benchmark_balancing` no seu ambiente após `pip install -r requirements.txt`.

### Compatibilidade
- `engineer_features`, `scale_features`, `balance_data`, `evaluate_predictions`, `compare_models` e `stratified_cross_validation` continuam existindo.
- Mudou: `get_supervised_models`/`get_anomaly_models` foram substituídos por `get_supervised_pipelines`/`AnomalyScorer`; `compute_shap_values` agora recebe o Pipeline; nomes de saída em `outputs/` (agora `modelo_fraude.joblib` único, em vez de modelo + scaler) e o dataset sintético (novas colunas).
