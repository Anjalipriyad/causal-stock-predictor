"""
ablation.py
-----------
Ablation study runner for the paper's Table 3 (ablation table).

The original backtester.regime_backtest() only compares:
    - pcmci_causal
    - all_features
    - random

This is insufficient for the paper. Reviewers will ask:
    "Is PCMCI better than Granger alone?"
    "Is the ensemble better than just LightGBM?"
    "Is causal selection adding value over just using fewer features?"

This module adds:
    1. granger_only:   LightGBM trained on Granger-selected features
    2. arima_only:     ARIMA model alone (no LightGBM/XGB)
    3. lgbm_only:      LightGBM alone (no blending) on causal features
    4. top_k_features: LightGBM on top-k features by LGBM importance
                       (sanity check: is PCMCI adding value over
                       just picking the features the model already likes?)

Paper Table 3 structure:
    Model               | bull | covid_crash | recovery | rate_hike | ai_bull
    pcmci_causal        |  DA  |     DA      |    DA    |    DA     |   DA
    granger_only        |  DA  |     DA      |    DA    |    DA     |   DA
    all_features        |  DA  |     DA      |    DA    |    DA     |   DA
    lgbm_only (causal)  |  DA  |     DA      |    DA    |    DA     |   DA
    arima_only          |  DA  |     DA      |    DA    |    DA     |   DA
    top_k_importance    |  DA  |     DA      |    DA    |    DA     |   DA
    random              |  DA  |     DA      |    DA    |    DA     |   DA

Usage:
    from ml.src.evaluation.ablation import AblationRunner
    runner = AblationRunner()
    results = runner.run(df, ticker="AAPL", causal_features=features,
                         granger_features=granger_features)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.src.data.loader import _load_config
from ml.src.evaluation.metrics import Metrics
from ml.src.evaluation.regime_splitter import RegimeSplitter

logger = logging.getLogger(__name__)


class AblationRunner:
    """
    Runs the full ablation study comparing all model variants.
    Produces the table needed to establish PCMCI's contribution.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.cfg     = _load_config(config_path)
        self.metrics = Metrics(config_path)
        self.splitter = RegimeSplitter(config_path)
        self.target   = self.cfg["model"]["target"]

        bt = self.cfg["evaluation"]["backtest"]
        self.min_test_samples = bt["min_test_samples"]

    def run(
        self,
        df: pd.DataFrame,
        ticker: str,
        causal_features: list[str],
        granger_features: list[str],
        top_k: int = 10,
        config_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Run all ablation variants across all regimes.

        Args:
            df:               Full feature matrix
            ticker:           Stock ticker
            causal_features:  PCMCI-selected features
            granger_features: Granger-only selected features
            top_k:            k for top-k LGBM importance baseline
            config_path:      Optional config override

        Returns:
            Multi-index DataFrame: (model, regime) → metrics dict
        """
        logger.info(f"[ablation] Running full ablation study for {ticker} ...")

        all_feature_cols = [c for c in df.columns if c != self.target]
        top_k_features   = self._get_top_k_features(
            df, causal_features, ticker, k=top_k, config_path=config_path
        )

        # All model variants to test
        variants = {
            "pcmci_causal":      causal_features,
            "granger_only":      granger_features,
            "all_features":      all_feature_cols,
            "top_k_importance":  top_k_features,
        }

        regime_splits = self.splitter.split_all(df)
        results       = []

        # ── Gradient boosting variants (all use full ensemble blending) ────
        for model_name, features in variants.items():
            if not features:
                logger.warning(f"[ablation] {model_name}: empty feature list, skipping.")
                continue
            logger.info(f"[ablation] Testing variant: {model_name} ({len(features)} features)")

            for regime_name, regime_df in regime_splits.items():
                if len(regime_df) < self.min_test_samples:
                    continue

                regime_start = self.splitter.regimes[regime_name][0]
                train_df     = df.loc[:regime_start].iloc[:-1]
                if len(train_df) < 200:
                    continue

                row = self._run_ensemble_variant(
                    train_df, regime_df, ticker, features,
                    model_name, regime_name, config_path
                )
                if row:
                    results.append(row)

        # ── LightGBM only (no blending) on causal features ─────────────────
        for regime_name, regime_df in regime_splits.items():
            if len(regime_df) < self.min_test_samples:
                continue
            regime_start = self.splitter.regimes[regime_name][0]
            train_df     = df.loc[:regime_start].iloc[:-1]
            if len(train_df) < 200:
                continue

            row = self._run_lgbm_only(
                train_df, regime_df, ticker, causal_features, regime_name, config_path
            )
            if row:
                results.append(row)

        # ── ARIMA only ──────────────────────────────────────────────────────
        for regime_name, regime_df in regime_splits.items():
            if len(regime_df) < self.min_test_samples:
                continue
            regime_start = self.splitter.regimes[regime_name][0]
            train_df     = df.loc[:regime_start].iloc[:-1]
            if len(train_df) < 200:
                continue

            row = self._run_arima_only(
                train_df, regime_df, ticker, regime_name
            )
            if row:
                results.append(row)

        # ── Random baseline ─────────────────────────────────────────────────
        for regime_name, regime_df in regime_splits.items():
            if self.target not in regime_df.columns:
                continue
            y_true = regime_df[self.target].dropna()
            if len(y_true) < self.min_test_samples:
                continue
            rng    = np.random.default_rng(42)
            y_rand = pd.Series(
                rng.choice([-0.01, 0.01], size=len(y_true)),
                index=y_true.index,
            )
            scores = self.metrics.compute_all(y_rand, y_true, label=f"random/{regime_name}")
            scores["model"]  = "random"
            scores["regime"] = regime_name
            scores["n_test"] = len(y_true)
            results.append(scores)

        if not results:
            logger.error("[ablation] No results computed.")
            return pd.DataFrame()

        result_df = (
            pd.DataFrame(results)
            .set_index(["model", "regime"])
            .sort_index()
        )

        logger.info(f"[ablation] Done. {len(result_df)} (model, regime) combinations.")
        self._print_da_table(result_df)
        return result_df

    # -----------------------------------------------------------------------
    # Individual variant runners
    # -----------------------------------------------------------------------

    def _run_ensemble_variant(
        self,
        train_df:    pd.DataFrame,
        test_df:     pd.DataFrame,
        ticker:      str,
        features:    list[str],
        model_name:  str,
        regime_name: str,
        config_path: Optional[str],
    ) -> Optional[dict]:
        """Train full ensemble and evaluate on test_df."""
        from ml.src.ensemble import Ensemble
        try:
            ensemble = Ensemble(config_path)
            ensemble.train_all(train_df, ticker, features)
            preds = ensemble.predict_historical(test_df, features)
            if "actual_return" not in preds.columns:
                return None
            scores = self.metrics.compute_all(
                preds["predicted_return"],
                preds["actual_return"],
                label=f"{model_name}/{regime_name}",
            )
            scores["model"]  = model_name
            scores["regime"] = regime_name
            scores["n_test"] = len(test_df)
            return scores
        except Exception as e:
            logger.warning(f"[ablation] {model_name}/{regime_name}: {e}")
            return None

    def _run_lgbm_only(
        self,
        train_df:    pd.DataFrame,
        test_df:     pd.DataFrame,
        ticker:      str,
        features:    list[str],
        regime_name: str,
        config_path: Optional[str],
    ) -> Optional[dict]:
        """Train LightGBM alone (no blending) and evaluate."""
        from ml.src.models.lgbm_model import LGBMModel
        try:
            model = LGBMModel(config_path)
            X_tr, X_va, _, y_tr, y_va, _ = model.prepare_data(train_df, features)
            X_tr_s, X_va_s, _ = model.scale(X_tr, X_va, X_tr, ticker)   # scale
            model.fit(X_tr_s, y_tr, X_va_s, y_va)

            # Scale test set
            feat_cols  = [c for c in features if c in test_df.columns]
            X_test     = test_df[feat_cols]
            X_test_s   = model.transform(X_test)
            y_pred     = pd.Series(model.predict_raw(X_test_s), index=X_test.index)
            y_true     = test_df[self.target] if self.target in test_df.columns else None

            if y_true is None:
                return None

            scores = self.metrics.compute_all(y_pred, y_true, label=f"lgbm_only/{regime_name}")
            scores["model"]  = "lgbm_only"
            scores["regime"] = regime_name
            scores["n_test"] = len(test_df)
            return scores
        except Exception as e:
            logger.warning(f"[ablation] lgbm_only/{regime_name}: {e}")
            return None

    def _run_arima_only(
        self,
        train_df:    pd.DataFrame,
        test_df:     pd.DataFrame,
        ticker:      str,
        regime_name: str,
    ) -> Optional[dict]:
        """Run ARIMA model alone and evaluate directional accuracy."""
        from ml.src.models.arima_model import ARIMAModel
        try:
            model = ARIMAModel()

            # Need a feature DataFrame for interface compatibility
            dummy_X = train_df[[c for c in train_df.columns if c != self.target]].iloc[:, :1]
            y_train = train_df[self.target]

            model.fit(dummy_X, y_train)

            # For test: get rolling one-step predictions on val set for meta-learner
            dummy_X_test = test_df[[c for c in test_df.columns if c != self.target]].iloc[:, :1]
            y_pred       = pd.Series(
                model.predict_raw(dummy_X_test),
                index=test_df.index,
            )
            y_true = test_df[self.target] if self.target in test_df.columns else None

            if y_true is None:
                return None

            scores = self.metrics.compute_all(y_pred, y_true, label=f"arima_only/{regime_name}")
            scores["model"]  = "arima_only"
            scores["regime"] = regime_name
            scores["n_test"] = len(test_df)
            return scores
        except Exception as e:
            logger.warning(f"[ablation] arima_only/{regime_name}: {e}")
            return None

    def _get_top_k_features(
        self,
        df: pd.DataFrame,
        causal_features: list[str],
        ticker: str,
        k: int,
        config_path: Optional[str],
    ) -> list[str]:
        """
        Get top-k features by LightGBM feature importance on full training set.
        This is the "model-selected features" baseline — comparing PCMCI's
        causal selection against what the model would pick on its own.
        """
        from ml.src.models.lgbm_model import LGBMModel
        try:
            all_feats  = [c for c in df.columns if c != self.target]
            model      = LGBMModel(config_path)
            X_tr, X_va, _, y_tr, y_va, _ = model.prepare_data(df, all_feats)
            X_tr_s, X_va_s, _ = model.scale(X_tr, X_va, X_tr, f"{ticker}_topk")
            model.fit(X_tr_s, y_tr, X_va_s, y_va)
            importance  = model.feature_importance("gain")
            top_k_feats = list(importance.head(k).index)
            logger.info(
                f"[ablation] Top-{k} features by LGBM importance: {top_k_feats}"
            )
            return top_k_feats
        except Exception as e:
            logger.warning(f"[ablation] Could not compute top-k features: {e}")
            return causal_features  # fallback

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------

    def _print_da_table(self, result_df: pd.DataFrame) -> None:
        """Print directional accuracy table for the paper."""
        if "directional_accuracy" not in result_df.columns:
            return

        try:
            pivot = result_df["directional_accuracy"].unstack(level="regime")
            W     = 90
            print(f"\n{'='*W}")
            print(f"  ABLATION TABLE — DIRECTIONAL ACCURACY BY MODEL AND REGIME")
            print(f"  (Option B: each regime tested on data the model never saw)")
            print(f"{'='*W}")
            header = f"  {'Model':<22}" + "".join(f"  {c[:12]:>12}" for c in pivot.columns)
            print(header)
            print(f"  {'-'*22}" + "".join(f"  {'-'*12}" for _ in pivot.columns))
            for model_name, row in pivot.iterrows():
                print(
                    f"  {str(model_name):<22}" +
                    "".join(
                        f"  {v:>12.4f}" if not pd.isna(v) else f"  {'N/A':>12}"
                        for v in row
                    )
                )
            print(f"\n  Random baseline ≈ 0.50 | Target > 0.52")
            print(f"{'='*W}")
        except Exception as e:
            logger.warning(f"[ablation] Could not print DA table: {e}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from ml.src.causal.granger import GrangerCausality
    from ml.src.causal.selector import CausalSelector

    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--ticker",  type=str, required=True)
    parser.add_argument("--market",  type=str, default="us",
                        choices=["us", "india"])
    parser.add_argument("--top-k",   type=int, default=10)
    args   = parser.parse_args()

    ticker = args.ticker.upper()
    cfg    = _load_config()

    if args.market == "india" or ticker in ("NIFTY", "^NSEI"):
        from ml.src.data.nifty_loader import NiftyLoader
        feat_path = NiftyLoader().out_dir / "NIFTY_features.csv"
        target    = "log_return_5d"
    else:
        from ml.src.features.pipeline import FeaturePipeline
        feat_path = FeaturePipeline().features_dir / f"{ticker}_features.csv"
        target    = cfg["model"]["target"]

    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)

    # Load causal features
    causal_features  = CausalSelector().load(ticker)

    # Rerun Granger to get Granger-only features
    train_end        = int(len(df) * cfg["model"]["train_ratio"])
    df_train         = df.iloc[:train_end]
    granger          = GrangerCausality()
    granger_results  = granger.run(df_train, target=target, verbose=False)
    granger_features = granger.get_causal_features(granger_results)

    runner  = AblationRunner()
    results = runner.run(df, ticker, causal_features, granger_features, top_k=args.top_k)
    print(results)