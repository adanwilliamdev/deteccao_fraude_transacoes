"""
data_gen.py
-----------
Geração / carregamento do dataset de transações.

O gerador sintético (v2) simula um cenário bem mais realista que o original:

    - Cada transação pertence a um cartão (``id_cartao``) com perfil próprio
      (valor médio, horário preferido, categorias preferidas, % online).
    - Fraudes ocorrem em **rajadas** (cartão comprometido faz várias compras
      em poucos minutos), com valores acima do perfil do cartão, madrugada,
      canal online, longe de casa e em categorias de risco.
    - ~30% das fraudes são "furtivas": imitam o comportamento normal do cartão.
      Isso gera sobreposição real entre as classes (como em dados de produção).
    - Colunas latentes V1..Vn (estilo dataset do Kaggle), com sobreposição.
    - Duplicados e nulos injetados de propósito para a EDA ter o que tratar.

Compatível com o CSV clássico do Kaggle (Time, V1..V28, Amount, Class): colunas
de cartão/categoria/canal são opcionais no restante do pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIAS = [
    "supermercado", "restaurante", "combustivel", "vestuario", "eletronicos",
    "viagem", "saude", "entretenimento", "servicos", "joalheria",
]
CATEGORIAS_RISCO = [4, 5, 9]  # eletronicos, viagem, joalheria


def generate_synthetic_transactions(
    n_samples: int = 60_000,
    fraud_ratio: float = 0.015,
    n_features: int = 14,
    n_cards: int | None = None,
    n_days: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Gera transações rotuladas (0 = legítima, 1 = fraude) ordenadas no tempo.

    Colunas: Time (segundos desde o início), id_cartao, categoria_comerciante,
    canal, distancia_casa_km, V1..Vn, Amount, Class.
    """
    rng = np.random.default_rng(random_state)
    n_fraud = max(1, int(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud
    n_cards = n_cards or max(50, n_samples // 20)
    n_cat = len(CATEGORIAS)

    # ---- Perfis dos cartões -------------------------------------------------
    card_mu = rng.normal(3.8, 0.5, n_cards)            # log do valor médio
    card_hour = rng.normal(15, 3.5, n_cards)           # horário preferido
    card_online = rng.beta(2, 5, n_cards)              # propensão a compra online
    card_cat = rng.dirichlet(np.ones(n_cat) * 0.6, n_cards)
    activity = rng.pareto(2.5, n_cards) + 1
    activity = activity / activity.sum()
    cat_cum = card_cat.cumsum(axis=1)

    def draw_cat(cards, u=None):
        u = rng.random(len(cards)) if u is None else u
        return np.minimum((u[:, None] > cat_cum[cards]).sum(axis=1), n_cat - 1)

    # ---- Legítimas -----------------------------------------------------------
    c_l = rng.choice(n_cards, n_legit, p=activity)
    hour_l = np.mod(rng.normal(card_hour[c_l], 3.0), 24)
    t_l = rng.integers(0, n_days, n_legit) * 86400 + hour_l * 3600 + rng.uniform(0, 60, n_legit)
    amt_l = np.exp(rng.normal(card_mu[c_l], 0.7))
    cat_l = draw_cat(c_l)
    onl_l = rng.random(n_legit) < card_online[c_l]
    dist_l = rng.exponential(6.0, n_legit)

    # ---- Fraudes (em rajadas) ------------------------------------------------
    sizes, total = [], 0
    while total < n_fraud:
        k = min(6, 1 + rng.poisson(1.6), n_fraud - total)
        sizes.append(int(k))
        total += int(k)
    sizes = np.array(sizes)
    n_ev = len(sizes)
    ev_card = rng.choice(n_cards, n_ev)
    ev_stealth = rng.random(n_ev) < 0.30
    ev_hour = np.where(
        rng.random(n_ev) < 0.55,
        np.mod(rng.normal(3, 2.5, n_ev), 24),
        rng.uniform(0, 24, n_ev),
    )
    ev_hour = np.where(ev_stealth, np.mod(rng.normal(card_hour[ev_card], 3.0), 24), ev_hour)
    ev_start = rng.integers(0, n_days, n_ev) * 86400 + ev_hour * 3600

    c_f = np.repeat(ev_card, sizes)
    stealth = np.repeat(ev_stealth, sizes)
    gaps = rng.exponential(600.0, n_fraud)
    t_f = np.empty(n_fraud)
    pos = 0
    for e, k in enumerate(sizes):
        g = gaps[pos:pos + k].copy()
        g[0] = 0.0
        t_f[pos:pos + k] = ev_start[e] + np.cumsum(g)
        pos += k

    amt_f = np.where(
        stealth,
        np.exp(rng.normal(card_mu[c_f] + 0.1, 0.6)),
        np.exp(rng.normal(card_mu[c_f] + 1.1, 0.8)),
    )
    risky = rng.random(n_fraud) < 0.6
    cat_f = np.where(
        stealth, draw_cat(c_f),
        np.where(risky, rng.choice(CATEGORIAS_RISCO, n_fraud), rng.integers(0, n_cat, n_fraud)),
    )
    onl_f = np.where(stealth, rng.random(n_fraud) < card_online[c_f], rng.random(n_fraud) < 0.8)
    dist_f = np.where(stealth, rng.exponential(10.0, n_fraud), rng.exponential(250.0, n_fraud))

    # ---- Variáveis latentes V1..Vn ------------------------------------------
    v_l = rng.normal(0.0, 1.0, (n_legit, n_features))
    v_f = rng.normal(0.0, 1.0, (n_fraud, n_features))
    shifted = rng.choice(n_features, size=max(1, n_features // 3), replace=False)
    shift = rng.normal(1.4, 0.6, (n_fraud, len(shifted))) * np.where(stealth, 0.5, 1.0)[:, None]
    v_f[:, shifted] += shift
    v_f += rng.normal(0.0, 0.6, v_f.shape)

    # ---- Montagem ------------------------------------------------------------
    df = pd.DataFrame(np.vstack([v_l, v_f]), columns=[f"V{i + 1}" for i in range(n_features)])
    df.insert(0, "Time", np.concatenate([t_l, t_f]))
    df.insert(1, "id_cartao", np.concatenate([c_l, c_f]).astype(int))
    df.insert(2, "categoria_comerciante", np.array(CATEGORIAS)[np.concatenate([cat_l, cat_f])])
    df.insert(3, "canal", np.where(np.concatenate([onl_l, onl_f]), "online", "presencial"))
    df.insert(4, "distancia_casa_km", np.concatenate([dist_l, dist_f]))
    df["Amount"] = np.clip(np.concatenate([amt_l, amt_f]), 1, 8000)
    df["Class"] = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)]).astype(int)

    df = df.sort_values("Time", kind="stable").reset_index(drop=True)

    # Sujeira controlada: duplicados e nulos (a EDA/limpeza precisa tratá-los).
    dup_idx = rng.choice(df.index, size=int(0.005 * len(df)), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True).sort_values("Time", kind="stable")
    df = df.reset_index(drop=True)
    df.loc[rng.choice(df.index, int(0.002 * len(df)), replace=False), "Amount"] = np.nan
    df.loc[rng.choice(df.index, int(0.003 * len(df)), replace=False), "distancia_casa_km"] = np.nan
    return df


def load_dataset(csv_path: str | None = None, **synthetic_kwargs) -> pd.DataFrame:
    """Lê um CSV real (se informado) ou gera o dataset sintético."""
    if csv_path:
        return pd.read_csv(csv_path)
    return generate_synthetic_transactions(**synthetic_kwargs)
