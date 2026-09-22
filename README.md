# Detecção de Anomalias em Transações (Detecção de Fraude)

Pipeline completo de Machine Learning para identificar transações
financeiras suspeitas, seguindo o roadmap clássico usado no mercado:

1. **Primeiros passos**: EDA + Feature Engineering
2. **Balanceamento e avaliação**: SMOTE/ADASYN/Undersampling + métricas corretas
3. **Modelos avançados e explicabilidade**: Ensembles + Isolation Forest/One-Class SVM + SHAP + Cross-Validation

O projeto já roda **de ponta a ponta** com um dataset sintético gerado
automaticamente (mesma estrutura do dataset clássico de fraude em cartão
de crédito). Para usar seus **dados reais**, basta apontar para o seu CSV.

## Estrutura do projeto

```
fraud_detection/
├── main.py                      # Orquestra o pipeline inteiro
├── requirements.txt
├── notebook_colab.ipynb         # Versão em notebook (Google Colab)
├── src/
│   ├── data_gen.py              # Geração/carregamento do dataset
│   ├── eda.py                   # Análise Exploratória dos Dados
│   ├── feature_engineering.py   # Criação de variáveis + normalização
│   ├── balancing.py             # SMOTE / ADASYN / Undersampling
│   ├── models.py                # Regressão Logística, RF, XGBoost,
│   │                             #  LightGBM, CatBoost, Isolation Forest,
│   │                             #  One-Class SVM
│   ├── evaluation.py            # Recall, Precisão, F1, PR-AUC, ROC-AUC,
│   │                             #  Stratified K-Fold
│   └── explainability.py        # SHAP (importância global + individual)
└── outputs/                     # Gerado a cada execução
    ├── eda/                     # Gráficos e tabelas da EDA
    ├── modelos/                 # Matrizes de confusão, curvas PR/ROC,
    │                             #  modelo treinado (.joblib)
    └── relatorios/               # Comparação de modelos, SHAP, JSON final
```

## Instalação

```bash
pip install -r requirements.txt
```

## Como executar

### Com dados sintéticos (funciona imediatamente, sem nenhum arquivo)

```bash
python main.py
```

Parâmetros opcionais:

```bash
python main.py --n_samples 50000 --fraud_ratio 0.015 --balance_strategy smote
```

### Com o seu dataset real

O CSV precisa ter uma coluna chamada `Class` (0 = legítima, 1 = fraude) e,
idealmente, colunas `Time` e `Amount` (como no dataset clássico do Kaggle
"Credit Card Fraud Detection"). Qualquer coluna numérica extra é aproveitada
automaticamente pelo pipeline.

```bash
python main.py --csv caminho/para/seu_dataset.csv
```

### Estratégias de balanceamento disponíveis

| Estratégia     | Tipo           | Quando usar |
|----------------|----------------|-------------|
| `smote`        | Oversampling   | Padrão. Bom equilíbrio geral. |
| `adasyn`       | Oversampling   | Quando a fronteira entre classes é mais complexa. |
| `undersample`  | Undersampling  | Quando a base é muito grande e você quer treinar mais rápido. |
| `none`         | Nenhum         | Baseline para comparação. |

```bash
python main.py --balance_strategy adasyn
```

## O que o pipeline faz, passo a passo

### 1. EDA (`src/eda.py`)
- Estatísticas descritivas (média, mediana, desvio padrão).
- Detecção de nulos, duplicados e outliers (método IQR).
- Limpeza automática (remove duplicados, imputa nulos pela mediana).
- Gráficos: balanço de classes, distribuição do valor por classe,
  matriz de correlação.

### 2. Feature Engineering (`src/feature_engineering.py`)
- `hora_do_dia` e `periodo_dia` (madrugada/manhã/tarde/noite) a partir do
  tempo da transação.
- `gap_tempo_transacao_anterior`: intervalo entre transações consecutivas.
- `log_amount` e `amount_zscore_global`: tratamento da assimetria do valor.
- `media_movel_valor` e `contagem_movel_transacoes`: proxy de frequência
  de uso do cartão (janela deslizante).
- Padronização via `RobustScaler` (mais resistente a outliers que o
  `StandardScaler` — importante porque outliers em fraude costumam ser
  o próprio sinal que queremos capturar, não erro de medição).

### 3. Balanceamento (`src/balancing.py`)
- SMOTE, ADASYN e Random Under Sampler, aplicados **somente no conjunto de
  treino** (nunca no teste, para não inflar artificialmente as métricas).

### 4. Modelos (`src/models.py`)
- Baseline: Regressão Logística (`class_weight="balanced"`).
- Ensembles: Random Forest, XGBoost, LightGBM, CatBoost (todos com peso de
  classe ajustado via `scale_pos_weight`).
- Não supervisionados: Isolation Forest e One-Class SVM (detectam anomalia
  sem usar o rótulo durante o treino).

### 5. Avaliação (`src/evaluation.py`)
- **Nunca usa acurácia pura** como critério de decisão.
- Métricas: Precisão, Recall, F1-Score, PR-AUC, ROC-AUC.
- Validação cruzada estratificada (`StratifiedKFold`) do melhor modelo.
- Gráficos: matriz de confusão, curvas Precision-Recall e ROC, comparação
  entre todos os modelos.

### 6. Explicabilidade (`src/explainability.py`)
- SHAP: importância global das variáveis (quais pesam mais, em média).
- SHAP waterfall: explica **uma transação específica** — essencial para a
  equipe de negócio auditar por que aquela transação foi marcada como
  suspeita.

## Saídas geradas (pasta `outputs/`)

- `eda/estatisticas_descritivas.csv`, `eda/balanceamento_classes.csv`
- `eda/01_balanco_classes.png`, `02_distribuicao_valor.png`, `03_correlacao.png`
- `modelos/matriz_confusao_<modelo>.png`, `curvas_pr_roc_<modelo>.png`
- `modelos/melhor_modelo_<nome>.joblib`, `modelos/scaler.joblib`
- `relatorios/comparacao_modelos.csv` e `.png`
- `relatorios/shap_importancia_global.png`, `shap_transacao_individual.png`
- `relatorios/relatorio_final.json` — resumo de tudo, para consumo por
  outros sistemas/dashboards.

## Usando o modelo treinado depois

```python
import joblib
import pandas as pd

modelo = joblib.load("outputs/modelos/melhor_modelo_<NOME>.joblib")
scaler = joblib.load("outputs/modelos/scaler.joblib")

# novas_transacoes: DataFrame com as MESMAS colunas usadas no treino
# (aplique src.feature_engineering.engineer_features antes de escalar)
probabilidades = modelo.predict_proba(novas_transacoes)[:, 1]
```

## Notas importantes

- O dataset sintético existe apenas para o projeto funcionar de ponta a
  ponta sem depender de download externo. Ele imita a estrutura e o
  desafio real (forte desbalanceamento, sobreposição parcial entre classes)
  mas **não substitui dados reais** para uso em produção.
- Para o dataset real do Kaggle ("Credit Card Fraud Detection"), baixe o
  `creditcard.csv` e rode `python main.py --csv creditcard.csv`.
