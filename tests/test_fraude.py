"""Testes do pipeline de fraude. Rode com:  python -m unittest discover -s tests -v   (ou pytest)."""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import balancing, data_gen, drift, evaluation as ev, feature_engineering as fe, splits
from src.config import load_config
from src.inference import FraudDetector
from src.models import ModelZoo, get_supervised_pipelines


def small_df(n=6000, seed=1):
    return data_gen.generate_synthetic_transactions(n_samples=n, fraud_ratio=0.03, random_state=seed).drop_duplicates()


class TestData(unittest.TestCase):
    def test_shape_ratio_and_columns(self):
        df = data_gen.generate_synthetic_transactions(n_samples=5000, fraud_ratio=0.02, random_state=0)
        self.assertGreaterEqual(len(df), 5000)
        self.assertAlmostEqual(df["Class"].mean(), 0.02, delta=0.004)
        for c in ("Time", "Amount", "Class", "id_cartao", "canal"):
            self.assertIn(c, df.columns)
        self.assertTrue(df["Time"].is_monotonic_increasing)

    def test_reproducible(self):
        a = data_gen.generate_synthetic_transactions(n_samples=2000, random_state=3)
        b = data_gen.generate_synthetic_transactions(n_samples=2000, random_state=3)
        pd.testing.assert_frame_equal(a, b)

    def test_validate_dataset(self):
        with self.assertRaises(ValueError):
            splits.validate_dataset(pd.DataFrame({"x": [1, 2]}))
        with self.assertRaises(ValueError):
            splits.validate_dataset(pd.DataFrame({"Class": [0, 1, 2] * 50}))
        with self.assertRaises(ValueError):  # poucas fraudes
            splits.validate_dataset(pd.DataFrame({"Class": [0] * 500 + [1] * 5}))


class TestFeatures(unittest.TestCase):
    def test_no_future_leakage(self):
        """Features das primeiras N linhas não podem mudar se o futuro for removido."""
        df = small_df(4000)
        full = fe.add_causal_features(df)
        n = 2000
        part = fe.add_causal_features(df.sort_values("Time").head(n))
        new_cols = [c for c in full.columns if c not in df.columns]
        self.assertTrue(new_cols)
        for c in new_cols:
            np.testing.assert_allclose(part[c].to_numpy(float), full[c].head(n).to_numpy(float), equal_nan=True, err_msg=c)

    def test_velocity_counts_are_correct(self):
        df = pd.DataFrame({"Time": [0, 100, 200, 4000, 4100], "id_cartao": [1, 1, 1, 1, 2],
                           "Amount": [10.0, 20.0, 30.0, 40.0, 50.0], "Class": [0] * 5})
        out = fe.add_causal_features(df, category_col=None)
        self.assertEqual(out["cartao_qtd_1h"].tolist(), [0, 1, 2, 0, 0])  # linha 4: as 3 anteriores saíram da janela de 1h
        self.assertEqual(out["cartao_valor_soma_1h"].tolist(), [0, 10, 30, 0, 0])
        self.assertEqual(out["cartao_qtd_24h"].tolist(), [0, 1, 2, 3, 0])

    def test_preprocessor_uses_train_statistics_only(self):
        tr = pd.DataFrame({"a": [1.0, 2.0, 3.0], "c": ["x", "y", "x"]})
        te = pd.DataFrame({"a": [np.nan, 100.0], "c": ["z", "x"]})
        prep = fe.TabularPreprocessor(scale=False).fit(tr)
        out = prep.transform(te)
        self.assertEqual(out["a"].iloc[0], 2.0)        # mediana do TREINO
        self.assertEqual(out["c"].iloc[0], -1)         # categoria inédita
        self.assertFalse(out.isna().any().any())


class TestBalancing(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = pd.DataFrame(rng.normal(size=(1000, 4)), columns=list("abcd"))
        self.y = np.r_[np.zeros(950, int), np.ones(50, int)]
        self.X.loc[self.y == 1, "a"] += 2

    def test_smote_balances_and_keeps_originals(self):
        Xr, yr = balancing.resample(self.X, self.y, "smote", 0, 1.0)
        self.assertEqual((yr == 1).sum(), (yr == 0).sum())
        np.testing.assert_allclose(Xr.iloc[:1000].to_numpy(), self.X.to_numpy())
        self.assertGreater(Xr.loc[yr == 1, "a"].mean(), 1.0)  # sintéticos ficam na região da minoria

    def test_undersample_and_none(self):
        _, yr = balancing.resample(self.X, self.y, "undersample", 0)
        self.assertEqual((yr == 1).sum(), (yr == 0).sum())
        Xn, yn = balancing.resample(self.X, self.y, "none")
        self.assertEqual(len(yn), 1000)
        with self.assertRaises(ValueError):
            balancing.resample(self.X, self.y, "inexistente")

    def test_wrapper_resamples_only_in_fit(self):
        from sklearn.linear_model import LogisticRegression
        clf = balancing.ResampledClassifier(LogisticRegression(), "smote").fit(self.X, self.y)
        self.assertEqual(clf.predict_proba(self.X.iloc[:7]).shape, (7, 2))


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = (rng.random(15000) < 0.02).astype(int)
        self.s = np.clip(rng.normal(0.1 + 0.5 * self.y, 0.15), 0, 1)
        self.a = np.exp(rng.normal(4, 1, 15000))

    def test_cost_threshold_is_optimal(self):
        t = ev.threshold_min_cost(self.y, self.s, self.a)
        best = ev.expected_cost(self.y, (self.s >= t).astype(int), self.a)["custo_total"]
        grid = min(ev.expected_cost(self.y, (self.s >= g).astype(int), self.a)["custo_total"] for g in np.unique(self.s)[::15])
        self.assertLessEqual(best, grid + 1e-6)
        self.assertLessEqual(best, ev.expected_cost(self.y, (self.s >= 0.5).astype(int), self.a)["custo_total"] + 1e-6)

    def test_recall_target(self):
        t = ev.threshold_recall_target(self.y, self.s, 0.8)
        self.assertGreaterEqual(ev.evaluate_scores(self.y, self.s, t)["recall"], 0.8)

    def test_expected_cost_edges(self):
        y, a = np.array([1, 0, 1]), np.array([100.0, 10.0, 50.0])
        c = ev.expected_cost(y, np.zeros(3, int), a, review_cost=5)
        self.assertEqual((c["custo_total"], c["economia"]), (150.0, 0.0))
        c = ev.expected_cost(y, np.array([1, 0, 1]), a, review_cost=5)
        self.assertEqual((c["custo_total"], c["economia"]), (10.0, 140.0))

    def test_bootstrap_ci_brackets_estimate(self):
        est, lo, hi = ev.bootstrap_ci(self.y, self.s, n_boot=100)
        self.assertLessEqual(lo, est)
        self.assertGreaterEqual(hi, est)

    def test_perfect_scores(self):
        m = ev.evaluate_scores(self.y, self.y.astype(float), 0.5)
        self.assertEqual((m["pr_auc"], m["recall"], m["precisao"]), (1.0, 1.0, 1.0))


class TestDriftSplitConfig(unittest.TestCase):
    def test_psi(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=5000)
        self.assertLess(drift.psi(a, rng.normal(size=5000)), 0.05)
        self.assertGreater(drift.psi(a, rng.normal(1.5, 1, 5000)), 0.25)

    def test_temporal_split_order_and_sizes(self):
        df = small_df(3000)
        tr, va, te = splits.temporal_split(df)
        self.assertLessEqual(tr["Time"].max(), va["Time"].min())
        self.assertLessEqual(va["Time"].max(), te["Time"].min())
        self.assertEqual(len(tr) + len(va) + len(te), len(df))

    def test_config(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.yaml")
            with open(p, "w") as f:
                f.write("n_samples: 1234\nsplit: stratified\n")
            cfg = load_config(p, split="temporal", review_cost=None)
            self.assertEqual((cfg.n_samples, cfg.split, cfg.review_cost), (1234, "temporal", 5.0))
            with open(p, "w") as f:
                f.write("chave_invalida: 1\n")
            with self.assertRaises(ValueError):
                load_config(p)


class TestInferenceAndEndToEnd(unittest.TestCase):
    def test_detector_roundtrip(self):
        df = fe.add_causal_features(small_df(6000))
        cols = [c for c in df.columns if c not in ("Class", "Time", "id_cartao")]
        pipe = get_supervised_pipelines(ModelZoo(n_jobs=1), 30.0, "class_weight", only=["Regressao_Logistica"])["Regressao_Logistica"]
        pipe.fit(df[cols], df["Class"])
        det = FraudDetector(pipe, 0.5, cols, card_col="id_cartao")
        raw = small_df(6000)
        history, new = raw.iloc[:-50], raw.iloc[-50:].drop(columns=["Class"])
        res = det.score(new, history=history)
        self.assertEqual(len(res), 50)
        self.assertTrue({"prob_fraude", "alerta", "faixa_risco"} <= set(res.columns))
        self.assertTrue(res["prob_fraude"].between(0, 1).all())
        with tempfile.TemporaryDirectory() as d:
            det.save(os.path.join(d, "m.joblib"))
            res2 = FraudDetector.load(os.path.join(d, "m.joblib")).score(new, history=history)
            np.testing.assert_allclose(res["prob_fraude"], res2["prob_fraude"])

    def test_pipeline_end_to_end(self):
        import main
        with tempfile.TemporaryDirectory() as d:
            cfg = load_config(None, n_samples=6000, fraud_ratio=0.03, output_dir=d, models=["Regressao_Logistica", "HistGradientBoosting"],
                              with_anomaly=False, explain=False, n_bootstrap=20, cv_splits=3)
            res = main.run(cfg)
            for f in ("relatorios/relatorio.md", "relatorios/relatorio_final.json", "modelos/modelo_fraude.joblib", "relatorios/comparacao_modelos.csv"):
                self.assertTrue(os.path.exists(os.path.join(d, f)), f)
            self.assertGreater(res["comparison"]["pr_auc"].max(), 0.4)


class TestRealDataFormats(unittest.TestCase):
    """Formatos de CSV real: estilo Kaggle (sem cartão) e sem Time/Amount."""

    def _run(self, df, **kw):
        import main
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "dados.csv")
            df.to_csv(path, index=False)
            cfg = load_config(None, csv=path, output_dir=os.path.join(d, "out"), models=["Regressao_Logistica"],
                              with_anomaly=False, n_bootstrap=20, cv_splits=3, **kw)
            res = main.run(cfg)
            self.assertTrue(os.path.exists(os.path.join(d, "out", "modelos", "modelo_fraude.joblib")))
            return res

    def test_kaggle_like_without_card_columns(self):
        df = data_gen.generate_synthetic_transactions(n_samples=6000, fraud_ratio=0.03, random_state=2)
        df = df.drop(columns=["id_cartao", "categoria_comerciante", "canal", "distancia_casa_km"])
        res = self._run(df, explain=False)
        self.assertEqual(res["final"]["split"], "temporal")

    def test_without_time_and_amount(self):
        df = data_gen.generate_synthetic_transactions(n_samples=6000, fraud_ratio=0.03, random_state=2)
        df = df.drop(columns=["Time", "Amount", "id_cartao"])
        res = self._run(df, explain=True)
        self.assertEqual(res["final"]["split"], "stratified")
        self.assertNotIn("economia_pct", res["comparison"].columns)
        det = res["detector"]
        scored = det.score(df.drop(columns=["Class"]).head(20))   # não pode exigir Time
        self.assertEqual(len(scored), 20)


if __name__ == "__main__":
    unittest.main()
