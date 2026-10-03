"""Configuração central do pipeline (CLI + YAML opcional)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class Config:
    # Dados
    csv: str | None = None
    n_samples: int = 60_000
    fraud_ratio: float = 0.015
    random_state: int = 42
    target_col: str = "Class"
    # Divisão
    split: str = "temporal"            # temporal | stratified
    test_size: float = 0.25
    val_size: float = 0.20
    # Desbalanceamento
    balance_strategy: str = "class_weight"   # class_weight | smote | adasyn | undersample | none
    balance_ratio: float = 1.0
    benchmark_balancing: bool = False
    # Modelos
    models: list | None = None         # subconjunto, ex.: ["LightGBM", "Random_Forest"]
    ensemble_top_k: int = 3
    with_anomaly: bool = True
    # Decisão / negócio
    threshold_strategy: str = "cost"   # cost | f1 | f2 | recall | default
    review_cost: float = 5.0           # custo de revisar 1 alerta (moeda do dataset)
    fn_factor: float = 1.0             # fração do valor da fraude perdida tratada como prejuízo
    recall_target: float = 0.80
    # Validação / incerteza
    cv_splits: int = 4
    n_bootstrap: int = 300
    # Saídas
    output_dir: str = "outputs"
    explain: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(yaml_path: str | None = None, **overrides) -> Config:
    """Mescla: padrões < YAML < overrides explícitos (não-None) da CLI."""
    values = {}
    if yaml_path:
        import yaml
        with open(yaml_path, encoding="utf-8") as f:
            values.update(yaml.safe_load(f) or {})
    values.update({k: v for k, v in overrides.items() if v is not None})
    valid = {f.name for f in fields(Config)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"Chaves de configuração desconhecidas: {sorted(unknown)}")
    return Config(**values)
