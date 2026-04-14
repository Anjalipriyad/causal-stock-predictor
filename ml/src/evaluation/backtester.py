"""
backtester.py
-------------
Walk-forward backtesting engine for the ensemble model.

Two evaluation modes:
    1. walk_forward(df, ticker)
       — Slides a training window forward in time, predicts on the next
         step, computes metrics. Avoids lookahead bias completely.

    2. regime_backtest(df, ticker)
       — Evaluates the model on each market regime independently.
         This is the core experiment of the paper — Table 2.
         Shows our model degrades less than baselines under regime shifts.

Also compares against two baselines:
    - all_features:  LightGBM trained on ALL features (no causal selection)
    - random:        Random directional predictor

Usage:
    from ml.src.evaluation.backtester import Backtester
    bt = Backtester()

    # Paper Table 2
    results = bt.regime_backtest(df, ticker="AAPL")
    print(results)

    # Walk-forward curve
    wf = bt.walk_forward(df, ticker="AAPL", causal_features=features)
"""

import logging
from pathlib import Path
from typing import Optional
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

from ml.src.data.loader import _load_config
from ml.src.ensemble import Ensemble
from ml.src.evaluation.metrics import Metrics
from ml.src.evaluation.regime_splitter import RegimeSplitter
from ml.src.causal.selector import CausalSelector
from ml.src.causal.granger import GrangerCausality
from ml.src.causal.pcmci import PCMCIDiscovery

logger = logging.getLogger(__name__)


class Backtester:
    """
    Walk-forward and regime-split backtesting for the causal ensemble.
    """

    def __init__(self, config_path: Optional[str] = None, cfg: Optional[dict] = None, market: Optional[str] = None):
        # Allow passing an already-loaded config dict (for runtime overrides, e.g. NIFTY)
        self.cfg = cfg if cfg is not None else _load_config(config_path)
        self.metrics = Metrics(config_path)
        # Pass market through to RegimeSplitter so India regimes can be used when needed
        self.splitter = RegimeSplitter(config_path, market=market)
        # CausalSelector accepts an optional cfg so pass through to keep overrides
        self.selector = CausalSelector(config_path, cfg=self.cfg)

        bt = self.cfg["evaluation"]["backtest"]
        self.initial_train_years = bt["initial_train_years"]
        self.step_size_months    = bt["step_size_months"]
        self.min_test_samples    = bt["min_test_samples"]
        self.target_col          = self.cfg["model"]["target"]

    # -----------------------------------------------------------------------
    # Walk-forward backtest
    # -----------------------------------------------------------------------

    def walk_forward(
        self,
        df: pd.DataFrame,
        ticker: str,
        causal_features: list[str],
        config_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Walk-forward backtesting with expanding training window.

        For each step:
            1. Train ensemble on all data up to current window
            2. Predict on next step_size_months
            3. Compute metrics
            4. Advance window

        Args:
            df:               Full feature matrix
            ticker:           Stock ticker
            causal_features:  Causal feature list from selector

        Returns:
            DataFrame with columns per window:
                window_start, window_end, n_test,
                directional_accuracy, sharpe_ratio, rmse, max_drawdown
        """
        logger.info(f"[backtester] Walk-forward backtest for {ticker} ...")

        results   = []
        start_idx = self._find_initial_train_end(df)

        if start_idx is None:
            logger.error("[backtester] Not enough data for walk-forward.")
            return pd.DataFrame()

        step = pd.DateOffset(months=self.step_size_months)
        window_start = df.index[0]
        window_end   = df.index[start_idx]

        while window_end < df.index[-1]:
            # Test window
            test_start = window_end + pd.Timedelta(days=1)
            test_end   = min(window_end + step, df.index[-1])

            train_df = df.loc[window_start:window_end]
            test_df  = df.loc[test_start:test_end]

            if len(test_df) < self.min_test_samples:
                window_end = test_end
                continue

            try:
                # Train a fresh ensemble on this window
                ensemble = Ensemble(config_path)
                X_test, y_test = ensemble.train_all(train_df, ticker, causal_features)

                # Predict on test window
                preds = ensemble.predict_historical(test_df, causal_features)

                if "actual_return" not in preds.columns:
                    window_end = test_end
                    continue

                scores = self.metrics.compute_all(
                    preds["predicted_return"],
                    preds["actual_return"],
                    label=f"{test_start.date()}→{test_end.date()}",
                )
                scores.update({
                    "window_start": str(window_start.date()),
                    "window_end":   str(window_end.date()),
                    "test_start":   str(test_start.date()),
                    "test_end":     str(test_end.date()),
                    "n_train":      len(train_df),
                    "n_test":       len(test_df),
                })
                results.append(scores)

            except Exception as e:
                logger.warning(
                    f"[backtester] Window {test_start.date()} failed: {e}"
                )

            window_end = test_end

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        logger.info(
            f"[backtester] Walk-forward complete. "
            f"{len(result_df)} windows evaluated."
        )
        return result_df

    # -----------------------------------------------------------------------
    # Regime backtest — the paper's main experiment
    # -----------------------------------------------------------------------

    def regime_backtest(
        self,
        df: pd.DataFrame,
        ticker: str,
        causal_features: list[str],
        config_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Evaluate the ensemble on each market regime independently.
        Also runs all_features and random baselines for comparison.

        This produces Table 2 of the paper:
            - Rows    = model variants (all_features, granger, pcmci_causal, random)
            - Columns = metrics per regime (DA, Sharpe, RMSE, MDD)

        Args:
            df:              Full feature matrix (labelled by RegimeSplitter)
            ticker:          Stock ticker
            causal_features: PCMCI causal feature list

        Returns:
            Multi-index DataFrame: (model, regime) → metrics
        """
        logger.info(f"[backtester] Regime backtest for {ticker} ...")

        all_feature_cols = [
            c for c in df.columns if c != self.target_col
        ]
        regime_splits = self.splitter.split_all(df)
        results       = []

        # For each regime, train on pre-regime data and evaluate both:
        #  - pcmci_causal: recompute causal features using only pre-regime train data
        #  - all_features: baseline trained on all available features
        for regime_name, regime_df in regime_splits.items():
            if len(regime_df) < self.min_test_samples:
                logger.warning(
                    f"[backtester] Skipping {regime_name} — only {len(regime_df)} samples."
                )
                continue

            # Use pre-regime data for training if possible
            regime_start = self.splitter.regimes[regime_name][0]
            train_df     = df.loc[:regime_start].iloc[:-1]

            if len(train_df) < 200:
                logger.warning(
                    f"[backtester] Not enough training data for {regime_name}, skipping."
                )
                continue

            # ------------------ PCMCI causal variant ------------------
            try:
                # Granger on full pre-regime training set
                granger = GrangerCausality()
                granger_results = granger.run(train_df, target=self.target_col, verbose=False)

                # PCMCI on last 50% of pre-regime training data (consistent with pipeline)
                df_pcmci = train_df.iloc[-int(len(train_df) * 0.5):]
                pcmci = PCMCIDiscovery()
                pcmci_results = pcmci.run(df_pcmci, target=self.target_col)

                # Select features using selector; do not overwrite saved global file
                try:
                    features_pcmci = self.selector.select(ticker, granger_results, pcmci_results, save=False)
                except Exception as e:
                    logger.warning(f"[backtester] Causal selector failed for {regime_name}: {e}")
                    features_pcmci = []

                if features_pcmci:
                    # Use regime-specific ticker name so save() does NOT
                    # overwrite production model files on disk
                    regime_ticker = f"{ticker}_{regime_name}_eval"
                    # Pass a fresh ensemble to avoid leakage/state issues
                    fresh = Ensemble(config_path=self.config_path)
                    fresh.train_all(train_df, regime_ticker, features_pcmci)
                    preds = fresh.predict_historical(regime_df, features_pcmci)
                    if "actual_return" in preds.columns:
                        scores = self.metrics.compute_all(
                            preds["predicted_return"],
                            preds["actual_return"],
                            label=f"pcmci_selected/{regime_name}",
                        )
                        # Bootstrap CI for directional accuracy
                        try:
                            lo, hi = self.metrics.bootstrap_da_ci(
                                preds["predicted_return"], preds["actual_return"]
                            )
                            scores["directional_accuracy_ci_lower"] = lo
                            scores["directional_accuracy_ci_upper"] = hi
                        except Exception:
                            scores["directional_accuracy_ci_lower"] = float("nan")
                            scores["directional_accuracy_ci_upper"] = float("nan")

                        scores["model"] = "pcmci_selected"
                        scores["regime"] = regime_name
                        scores["n_test"] = len(regime_df)
                        results.append(scores)
                else:
                    logger.warning(f"[backtester] No causal features for {regime_name}; skipping pcmci_causal.")

            except Exception as e:
                logger.warning(f"[backtester] pcmci_causal/{regime_name} failed: {e}")

            # ------------------ All-features baseline ------------------
            try:
                ensemble_all = Ensemble(config_path)
                ensemble_all.train_all(train_df, ticker, all_feature_cols)
                preds_all = ensemble_all.predict_historical(regime_df, all_feature_cols)
                if "actual_return" in preds_all.columns:
                    scores_all = self.metrics.compute_all(
                        preds_all["predicted_return"],
                        preds_all["actual_return"],
                        label=f"all_features/{regime_name}",
                    )
                    try:
                        lo, hi = self.metrics.bootstrap_da_ci(
                            preds_all["predicted_return"], preds_all["actual_return"]
                        )
                        scores_all["directional_accuracy_ci_lower"] = lo
                        scores_all["directional_accuracy_ci_upper"] = hi
                    except Exception:
                        scores_all["directional_accuracy_ci_lower"] = float("nan")
                        scores_all["directional_accuracy_ci_upper"] = float("nan")

                    scores_all["model"] = "all_features"
                    scores_all["regime"] = regime_name
                    scores_all["n_test"] = len(regime_df)
                    results.append(scores_all)

                # ------------------ Momentum / Mean-reversion / Buy-and-hold baselines
                try:
                    # y_true for this regime
                    if self.target_col not in regime_df.columns:
                        raise KeyError("target missing")
                    y_true = regime_df[self.target_col].dropna()
                    if len(y_true) >= self.min_test_samples:
                        # Momentum baseline: use 'momentum_5d' if present, else prior return
                        if "momentum_5d" in regime_df.columns:
                            sig = regime_df.loc[y_true.index, "momentum_5d"].fillna(0.0)
                            y_pred_mom = pd.Series(
                                np.where(sig >= 0, 0.01, -0.01), index=y_true.index
                            )
                        else:
                            prev = regime_df.loc[y_true.index, self.target_col].shift(1).fillna(0.0)
                            y_pred_mom = pd.Series(np.where(prev >= 0, 0.01, -0.01), index=y_true.index)

                        mom_scores = self.metrics.compute_all(y_pred_mom, y_true, label=f"momentum/{regime_name}")
                        lo, hi = self.metrics.bootstrap_da_ci(y_pred_mom, y_true)
                        mom_scores["directional_accuracy_ci_lower"] = lo
                        mom_scores["directional_accuracy_ci_upper"] = hi
                        mom_scores["model"] = "momentum"
                        mom_scores["regime"] = regime_name
                        mom_scores["n_test"] = len(y_true)
                        results.append(mom_scores)

                        # Mean-reversion: opposite of momentum
                        y_pred_mr = -1 * y_pred_mom
                        mr_scores = self.metrics.compute_all(y_pred_mr, y_true, label=f"mean_reversion/{regime_name}")
                        lo, hi = self.metrics.bootstrap_da_ci(y_pred_mr, y_true)
                        mr_scores["directional_accuracy_ci_lower"] = lo
                        mr_scores["directional_accuracy_ci_upper"] = hi
                        mr_scores["model"] = "mean_reversion"
                        mr_scores["regime"] = regime_name
                        mr_scores["n_test"] = len(y_true)
                        results.append(mr_scores)

                        # Buy-and-hold baseline: always predict UP
                        y_pred_bh = pd.Series(0.01, index=y_true.index)
                        bh_scores = self.metrics.compute_all(y_pred_bh, y_true, label=f"buy_and_hold/{regime_name}")
                        lo, hi = self.metrics.bootstrap_da_ci(y_pred_bh, y_true)
                        bh_scores["directional_accuracy_ci_lower"] = lo
                        bh_scores["directional_accuracy_ci_upper"] = hi
                        bh_scores["model"] = "buy_and_hold"
                        bh_scores["regime"] = regime_name
                        bh_scores["n_test"] = len(y_true)
                        results.append(bh_scores)

                except Exception:
                    # If any baseline fails for this regime, continue gracefully
                    pass

            except Exception as e:
                logger.warning(f"[backtester] all_features/{regime_name} failed: {e}")

        # Random baseline
        for regime_name, regime_df in regime_splits.items():
            if self.target_col not in regime_df.columns:
                continue
            y_true  = regime_df[self.target_col].dropna()
            if len(y_true) < self.min_test_samples:
                continue
            rng     = np.random.default_rng(42)
            y_rand  = pd.Series(
                rng.choice([-0.01, 0.01], size=len(y_true)),
                index=y_true.index,
            )
            scores  = self.metrics.compute_all(y_rand, y_true, label=f"random/{regime_name}")
            scores["model"]  = "random"
            scores["regime"] = regime_name
            scores["n_test"] = len(y_true)
            results.append(scores)

        if not results:
            logger.error("[backtester] No regime results computed.")
            return pd.DataFrame()

        result_df = (
            pd.DataFrame(results)
            .set_index(["model", "regime"])
            .sort_index()
        )

        logger.info(
            f"[backtester] Regime backtest complete.\n{result_df}"
        )
        return result_df

    # -----------------------------------------------------------------------
    # Single evaluation — test set only
    # -----------------------------------------------------------------------

    def evaluate_test_set(
        self,
        ticker: str,
        causal_features: list[str],
        df: pd.DataFrame,
        config_path: Optional[str] = None,
    ) -> dict[str, float]:
        """
        Train on train+val, evaluate on held-out test set.
        Quick sanity check after training.

        Returns:
            Dict of metric scores on the test set.
        """
        ensemble = Ensemble(config_path)
        X_test_s, y_test = ensemble.train_all(df, ticker, causal_features)

        preds = ensemble.predict_historical(
            pd.concat([X_test_s, y_test], axis=1),
            causal_features,
        )

        return self.metrics.compute_all(
            preds["predicted_return"],
            preds["actual_return"],
            label=f"{ticker} test set",
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _find_initial_train_end(
        self, df: pd.DataFrame
    ) -> Optional[int]:
        """
        Find the index position corresponding to initial_train_years
        of data from the start of df.
        Returns None if not enough data.
        """
        start = df.index[0]
        end   = start + relativedelta(years=self.initial_train_years)

        mask = df.index <= end
        if mask.sum() < 200:
            return None

        return int(mask.sum()) - 1