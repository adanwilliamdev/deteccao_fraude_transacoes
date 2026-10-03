# Detecção de Fraude em Transações — v2

Pipeline de Machine Learning de ponta a ponta para identificar transações
fraudulentas: EDA → features sem vazamento → split temporal → modelos e
ensemble → limiar por custo de negócio → intervalos de confiança → drift →
explicabilidade → modelo empacotado, CLI e API.

Funciona **imediatamente** com um dataset sintético (cartões, canal, categoria,
distância, fraudes em rajada) e aceita o seu CSV real (coluna `Class`).

```bash
pip install -r requirements.txt        # ou requirements-core.txt (só scikit-learn)
python main.py                         # tudo, com dados sintéticos
python main.py --csv creditcard.csv    # dados reais (Kaggle: Time, V1..V28, Amount, Class)
make test                              # 21 testes
```

## O que mudou em relação à v1 (e por quê)

A v1 já tinha o roteiro certo, mas alguns detalhes metodológicos **inflavam as
métricas**. A v2 corrige isso e acrescenta o que falta para uso real
(detalhes no [CHANGELOG](CHANGELOG.md)):

| Tema | v1 | v2 |
|---|---|---|
| Scaler / imputação | ajustados no dataset inteiro (vazamento) | ajustados só no treino, dentro do Pipeline |
| SMOTE + CV | SMOTE antes da CV (sintéticos na validação) | reamostragem **dentro** do estimador, só no `fit` |
| Escolha do modelo | pelo PR-AUC do **teste** | pela **validação**; o teste só reporta |
| Split | aleatório estratificado | **temporal** (passado → futuro); estratificado opcional |
| Features | hora, gap global, rolling por linha | **por cartão**: velocidade 1h/24h, valor vs. média do cartão, categoria nova, hora cíclica (causais) |
| Decisão | limiar fixo 0,5 | limiar por **custo esperado**, F1, F2 ou recall-alvo |
| Incerteza | métrica pontual | **IC95%** por bootstrap + CV temporal |
| Métricas | precisão, recall, F1, AUCs | + MCC, FPR, Brier, precisão/recall no top-1%, economia em R$ |
| Modelos | 5 + 2 anomalias (libs obrigatórias) | + Extra Trees, HistGB, **ensemble**; XGB/LGBM/CatBoost/SHAP opcionais |
| Produção | `.joblib` + scaler soltos | 1 arquivo (pipeline+limiar+features), `predict.py`, `api.py`, PSI de drift |

## Estrutura

```
├── main.py                 # orquestra o pipeline (função run(cfg) reutilizável)
├── predict.py              # CLI: pontua transações novas
├── api.py                  # serviço FastAPI (opcional)
├── configs/default.yaml    # configuração
├── notebook_colab.ipynb    # passo a passo interativo
├── tests/test_fraude.py    # 21 testes (vazamento, custo, SMOTE, PSI, e2e...)
└── src/
    ├── data_gen.py            # dataset sintético / carregamento de CSV
    ├── eda.py                 # estatísticas, qualidade, gráficos (hora, canal, categoria)
    ├── splits.py              # validação de esquema + split temporal/estratificado
    ├── feature_engineering.py # features causais + TabularPreprocessor
    ├── balancing.py           # SMOTE (próprio), ADASYN, undersampling, ResampledClassifier
    ├── models.py              # modelos, ensemble, detectores de anomalia
    ├── evaluation.py          # métricas, limiares, custo, bootstrap, CV, gráficos
    ├── explainability.py      # SHAP (opcional) / permutação + oclusão
    ├── drift.py               # PSI
    ├── inference.py           # FraudDetector (modelo empacotado)
    └── config.py
```

## Como o pipeline decide

```
dados → limpeza → features causais → [ treino | validação | teste ]  (ordem temporal)
         │
         ├─ treino    : ajusta pré-processamento, reamostragem e modelos
         ├─ validação : escolhe modelo, pesos do ensemble e LIMIAR
         └─ teste     : só reporta (nenhuma decisão olha para ele)
```

- **Features causais**: cada linha usa apenas transações *anteriores* do mesmo
  cartão (teste automatizado garante que remover o futuro não altera nada).
  Sem `id_cartao` (ex.: dataset do Kaggle) cai para features globais.
- **Custo de negócio**: `custo = review_cost × nº de alertas + fn_factor × valor das fraudes perdidas`.
  O limiar `cost` minimiza isso. Ajuste `review_cost` e `fn_factor` à sua realidade.
- **Ensemble**: média das probabilidades dos top-k modelos, ponderada pelo PR-AUC de validação.
- **Anomalias**: Isolation Forest/One-Class SVM treinados só com legítimas, avaliados por score.

## Resultados de exemplo (dados sintéticos, 60k transações, 1,5% fraude)

Teste temporal: 15.000 transações, 230 fraudes. Limiar `cost` escolhido na validação, `review_cost = 5`.

| Modelo | PR-AUC (IC95%) | Precisão | Recall | Economia vs. sem modelo |
|---|---|---|---|---|
| Ensemble_top3 *(escolhido pela validação)* | 0,855 (0,811–0,891) | 0,644 | 0,835 | 88,2% |
| Regressão Logística | 0,851 (0,808–0,891) | 0,676 | 0,817 | 90,1% |
| HistGradientBoosting | 0,844 (0,798–0,880) | 0,478 | 0,843 | 87,1% |
| Random Forest | 0,835 (0,792–0,875) | 0,490 | 0,839 | 87,5% |
| Extra Trees | 0,826 (0,776–0,866) | 0,597 | 0,826 | 87,9% |
| One-Class SVM | 0,648 (0,583–0,707) | 0,272 | 0,809 | 81,1% |
| Isolation Forest | 0,600 (0,540–0,654) | 0,294 | 0,765 | 84,1% |

**Leitura honesta:** os ICs dos cinco primeiros se sobrepõem — não há vencedor
estatístico entre eles. O dataset sintético foi desenhado para ter sinal claro
(distância e velocidade); serve para validar o pipeline, **não** para prever
desempenho em produção. Observações que valem para dados reais também:

- **Balanceamento** (`--benchmark_balancing`): `none`, `class_weight`, `smote` e
  `undersample` ficaram dentro de ~1,5 p.p. de PR-AUC, mas as probabilidades
  pioram muito com reamostragem/pesos (Brier 0,004 → ~0,043). Se você usa a
  probabilidade em si (não só o ranking), calibre ou prefira `none` + limiar.
- **Limiar**: a estratégia escolhida muda muito o perfil de alertas, com o mesmo modelo
  (ex.: F1 → 184 alertas, precisão 0,92 e recall 0,74; custo → 298 alertas, precisão 0,64 e recall 0,84).
- Todos os artefatos desta execução estão em `outputs/` (incluindo `relatorios/relatorio.md`).

## Uso

### Treinar
```bash
python main.py --balance_strategy smote --benchmark_balancing
python main.py --threshold_strategy f2 --review_cost 12 --fn_factor 0.8
python main.py --models LightGBM Random_Forest --split stratified
python main.py --config configs/default.yaml --output_dir runs/exp1
```
A CLI sobrescreve o YAML. Todas as opções: `python main.py -h`.

### Pontuar transações novas
```bash
python predict.py --model outputs/modelos/modelo_fraude.joblib \
                  --input novas.csv --history historico_recente.csv --output pontuadas.csv
```
Saída: `prob_fraude`, `alerta` (0/1) e `faixa_risco` (baixo/médio/alto). As features
por cartão precisam do histórico recente (`--history`, ao menos as últimas 24h).

```python
from src.inference import FraudDetector
det = FraudDetector.load("outputs/modelos/modelo_fraude.joblib")
resultado = det.score(df_novas, history=df_historico)
```

### API (opcional)
```bash
pip install -r requirements-api.txt
uvicorn api:app --port 8000      # POST /score  |  GET /health
```

## Arquivos gerados (`outputs/`)

- `eda/` — estatísticas, balanceamento, correlação, fraude por hora/canal/categoria, ausentes.
- `modelos/` — `modelo_fraude.joblib`, matrizes de confusão, PR/ROC (todos os modelos), análise de limiar, calibração, ganho/lift, curva de custo.
- `relatorios/` — `relatorio.md` (resumo legível), `relatorio_final.json`, `comparacao_modelos.csv/.png`, `estrategias_limiar.csv`, `validacao_cruzada.csv`, `benchmark_balanceamento.csv`, `drift_psi_features.csv`, importância (SHAP ou permutação) e explicação de uma transação.

## Usando dados reais

O CSV precisa de `Class` (0/1). `Time` e `Amount` ativam features temporais, split
temporal e análise de custo. Colunas opcionais que enriquecem o modelo:
`id_cartao`, `categoria_comerciante`, `canal`, `distancia_casa_km` (ou equivalentes —
renomeie ou ajuste `add_causal_features`). Qualquer outra coluna numérica entra como feature.

## Limitações e próximos passos

- Features por cartão exigem histórico no momento da inferência (use um feature store/cache em produção).
- Sem calibração de probabilidade (isotônica/Platt) — o limiar escolhido na validação compensa, mas as probabilidades absolutas não são confiáveis com `class_weight`/SMOTE.
- Ideias: ajuste de hiperparâmetros (Optuna), features de grafo (cartão–comerciante), aprendizado com custo por transação, retreino agendado acionado por PSI.
