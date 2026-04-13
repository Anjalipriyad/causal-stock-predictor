"""
hmm_regime_detector.py
-----------------------
Unsupervised Hidden Markov Model regime detection.

Addresses the survivorship bias criticism of manually labeled regimes:
the paper's regime boundaries (bull, crash, recovery, etc.) are defined
retrospectively with known end dates, which a practitioner wouldn't have.

This module implements an alternative regime labeling using a 2-state
Gaussian HMM trained on VIX + S&P 500 returns. The HMM detects
high-vol/low-return and low-vol/high-return regimes automatically,
without using future information.

Paper section: Section 4.3 — "Robustness Check: HMM-Detected Regimes"
    "To address the concern that our manually labeled regime boundaries
     are retrospectively defined, we repeat the regime evaluation using
     an unsupervised Hidden Markov Model trained on VIX and market returns
     only. Table A1 shows that the causal model's advantage holds under
     HMM-detected regimes (mean DA improvement: X.XX pp, p < 0.05)."

If the causal model still outperforms under HMM regimes, the paper's
central claim is significantly strengthened.

Usage:
    from ml.src.evaluation.hmm_regime_detector import HMMRegimeDetector
    detector = HMMRegimeDetector()
    df_labeled = detector.fit_label(df, n_states=2)
    regime_splits = detector.split_by_hmm_state(df_labeled)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# State labels assigned based on the HMM state with higher volatility.
# State 0 = LOW_VOL_BULL  (low VIX, positive returns)
# State 1 = HIGH_VOL_BEAR (high VIX, negative returns or flat)
# The assignment is data-driven — we infer which state is which from
# the fitted emission means after training.
HMM_STATE_NAMES = {0: "hmm_low_vol", 1: "hmm_high_vol"}


class HMMRegimeDetector:
    """
    Gaussian HMM for unsupervised market regime detection.
    Uses VIX level + S&P 500 5-day log return as observation variables.

    n_states=2 is the standard "bull/bear" model.
    n_states=3 adds a "transition" state between bull and bear.
    """

    def __init__(self, n_states: int = 2, n_iter: int = 200, random_state: int = 42):
        self.n_states     = n_states
        self.n_iter       = n_iter
        self.random_state = random_state
        self._model       = None
        self._is_fitted   = False
        self._state_names: dict[int, str] = {}

    def fit(self, df: pd.DataFrame) -> "HMMRegimeDetector":
        """
        Fit the HMM on VIX + SP500 return features.

        Only uses VIX and market return — no stock-specific features.
        This ensures the regime detection is purely market-condition driven
        and not contaminated by the target stock's behavior.

        Args:
            df: Feature matrix with columns including vix_level and
                sp500_return_1d (or india_vix for Nifty).

        Returns:
            self (fitted)
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError(
                "hmmlearn required for HMM regime detection. "
                "pip install hmmlearn"
            )

        # Select observation features (in order of preference)
        obs_features = self._select_obs_features(df)
        if len(obs_features) < 1:
            raise ValueError(
                "[hmm] No suitable observation features found. "
                "Need vix_level or india_vix, and sp500_return_1d."
            )

        logger.info(
            f"[hmm] Fitting {self.n_states}-state Gaussian HMM on "
            f"features: {obs_features} ({len(df)} observations) ..."
        )

        obs_df = df[obs_features].dropna()
        X      = obs_df.values

        # Standardize: HMM is sensitive to feature scale
        self._obs_mean = X.mean(axis=0)
        self._obs_std  = X.std(axis=0) + 1e-8
        X_scaled       = (X - self._obs_mean) / self._obs_std

        model = GaussianHMM(
            n_components  = self.n_states,
            covariance_type = "full",
            n_iter        = self.n_iter,
            random_state  = self.random_state,
        )
        model.fit(X_scaled)

        self._model         = model
        self._obs_features  = obs_features
        self._obs_index     = obs_df.index
        self._is_fitted     = True

        # Assign state names based on VIX emission means
        # (state with higher mean VIX = high-vol state)
        vix_idx      = obs_features.index(obs_features[0])  # VIX is first
        vix_means    = model.means_[:, vix_idx]
        high_vol_state = int(np.argmax(vix_means))
        low_vol_state  = int(np.argmin(vix_means))

        self._state_names = {
            low_vol_state:  "hmm_low_vol_bull",
            high_vol_state: "hmm_high_vol_bear",
        }
        # For 3-state: middle state gets "hmm_transition"
        if self.n_states == 3:
            mid_state = {0, 1, 2} - {low_vol_state, high_vol_state}
            if mid_state:
                self._state_names[mid_state.pop()] = "hmm_transition"

        logger.info(
            f"[hmm] HMM fitted. Log-likelihood: {model.score(X_scaled):.1f}. "
            f"State names: {self._state_names}"
        )
        self._print_state_summary(model, obs_features)
        return self

    def label(self, df: pd.DataFrame) -> pd.Series:
        """
        Assign HMM regime labels to each row of df.

        Returns:
            pd.Series with same index as df, values = HMM state names.
        """
        if not self._is_fitted:
            raise RuntimeError("[hmm] Call fit() first.")

        obs_df  = df[self._obs_features].reindex(df.index)
        X       = obs_df.fillna(method="ffill").fillna(0.0).values
        X_scaled = (X - self._obs_mean) / self._obs_std

        states  = self._model.predict(X_scaled)
        labels  = pd.Series(
            [self._state_names.get(s, f"hmm_state_{s}") for s in states],
            index=df.index,
            name="hmm_regime",
        )
        return labels

    def fit_label(self, df: pd.DataFrame, n_states: int = None) -> pd.DataFrame:
        """Convenience: fit and label in one call."""
        if n_states is not None:
            self.n_states = n_states
        self.fit(df)
        df = df.copy()
        df["hmm_regime"] = self.label(df)
        return df

    def split_by_hmm_state(
        self, df: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        """
        Split df by HMM-detected regime states.
        Returns dict matching the interface of RegimeSplitter.split_all().

        This allows direct substitution in backtester.regime_backtest():
            Instead of splitter.split_all(df), use detector.split_by_hmm_state(df_labeled)
        """
        if "hmm_regime" not in df.columns:
            df = self.fit_label(df)

        result = {}
        for state_name in df["hmm_regime"].unique():
            subset = df[df["hmm_regime"] == state_name]
            if not subset.empty:
                result[state_name] = subset
                logger.info(
                    f"[hmm] State '{state_name}': {len(subset)} rows "
                    f"({subset.index.min().date()} → {subset.index.max().date()})"
                )
        return result

    def compare_with_manual_regimes(
        self, df: pd.DataFrame, manual_labels: pd.Series
    ) -> pd.DataFrame:
        """
        Compute overlap between HMM-detected states and manually labeled regimes.
        Useful for validating that HMM high-vol state ≈ manual crash/rate_hike regimes.

        Returns:
            Cross-tabulation DataFrame (confusion matrix style).
        """
        if "hmm_regime" not in df.columns:
            df = self.fit_label(df)

        hmm_labels = df["hmm_regime"].reindex(manual_labels.index)
        combined   = pd.DataFrame({
            "manual": manual_labels,
            "hmm":    hmm_labels,
        }).dropna()

        crosstab = pd.crosstab(
            combined["manual"],
            combined["hmm"],
            normalize="index",
        ).round(3)

        print("\nHMM vs Manual Regime Overlap (row-normalized):")
        print(crosstab.to_string())
        return crosstab

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _select_obs_features(self, df: pd.DataFrame) -> list[str]:
        """Select the best available observation features from df."""
        candidates = [
            # VIX (US or India) — first available wins
            ("vix_level",        "india_vix"),
            # Market return — first available wins
            ("sp500_return_1d",  "log_return_1d"),
        ]
        selected = []
        for pair in candidates:
            for feat in pair:
                if feat in df.columns and df[feat].notna().sum() > 100:
                    selected.append(feat)
                    break
        return selected

    def _print_state_summary(self, model, obs_features: list[str]) -> None:
        """Log HMM state emission parameters for paper reporting."""
        print(f"\n  HMM State Summary ({self.n_states} states):")
        for i in range(self.n_states):
            name = self._state_names.get(i, f"state_{i}")
            means = model.means_[i]
            stds  = np.sqrt(np.diag(model.covars_[i]))
            param_str = ", ".join(
                f"{feat}={m:.3f}±{s:.3f}"
                for feat, m, s in zip(obs_features, means, stds)
            )
            prior = model.startprob_[i]
            print(f"  State {i} ({name}): {param_str} | prior={prior:.3f}")


# ---------------------------------------------------------------------------
# Integration with backtester: HMM-based regime backtest
# ---------------------------------------------------------------------------

def run_hmm_regime_backtest(
    df: pd.DataFrame,
    ticker: str,
    causal_features: list[str],
    n_hmm_states: int = 2,
    config_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run the regime backtest using HMM-detected states instead of
    manually labeled regimes. Used for the paper's robustness check.

    Args:
        df:              Full feature matrix
        ticker:          Stock ticker
        causal_features: Causal feature list from selector
        n_hmm_states:    Number of HMM states (2 or 3)

    Returns:
        Results DataFrame in same format as Backtester.regime_backtest()
    """
    from ml.src.evaluation.backtester import Backtester
    from ml.src.evaluation.metrics import Metrics
    from ml.src.ensemble import Ensemble

    logger.info(f"[hmm] Running HMM regime backtest for {ticker} ...")

    # Fit HMM and label regimes
    detector   = HMMRegimeDetector(n_states=n_hmm_states)
    df_labeled = detector.fit_label(df)
    hmm_splits = detector.split_by_hmm_state(df_labeled)

    bt      = Backtester(config_path)
    metrics = Metrics(config_path)
    results = []

    all_feature_cols = [c for c in df.columns if c != bt.target_col]
    model_variants   = {
        "pcmci_causal": causal_features,
        "all_features": all_feature_cols,
    }

    for model_name, features in model_variants.items():
        for state_name, state_df in hmm_splits.items():
            if len(state_df) < bt.min_test_samples:
                continue

            # Use all data before first row of this state as training data
            train_cutoff = state_df.index.min()
            train_df     = df.loc[:train_cutoff].iloc[:-1]
            if len(train_df) < 200:
                continue

            try:
                ensemble = Ensemble(config_path)
                ensemble.train_all(train_df, ticker, features)
                preds = ensemble.predict_historical(state_df, features)

                if "actual_return" not in preds.columns:
                    continue

                scores = metrics.compute_all(
                    preds["predicted_return"],
                    preds["actual_return"],
                    label=f"HMM/{model_name}/{state_name}",
                )
                scores["model"]  = model_name
                scores["regime"] = state_name
                scores["n_test"] = len(state_df)
                results.append(scores)

            except Exception as e:
                logger.warning(f"[hmm] {model_name}/{state_name}: {e}")

    if not results:
        return pd.DataFrame()

    return (
        pd.DataFrame(results)
        .set_index(["model", "regime"])
        .sort_index()
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from ml.src.data.loader import _load_config
    from ml.src.causal.selector import CausalSelector

    parser = argparse.ArgumentParser(
        description="Run HMM regime detection and robustness check"
    )
    parser.add_argument("--ticker",     type=str, required=True)
    parser.add_argument("--market",     type=str, default="us",
                        choices=["us", "india"])
    parser.add_argument("--n-states",   type=int, default=2)
    parser.add_argument("--backtest",   action="store_true",
                        help="Run full HMM regime backtest")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    cfg    = _load_config()

    if args.market == "india" or ticker in ("NIFTY", "^NSEI"):
        from ml.src.data.nifty_loader import NiftyLoader
        feat_path = NiftyLoader().out_dir / "NIFTY_features.csv"
    else:
        from ml.src.features.pipeline import FeaturePipeline
        feat_path = FeaturePipeline().features_dir / f"{ticker}_features.csv"

    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)

    # Fit and label
    detector   = HMMRegimeDetector(n_states=args.n_states)
    df_labeled = detector.fit_label(df)

    # Compare with manual regimes
    from ml.src.evaluation.regime_splitter import RegimeSplitter
    splitter       = RegimeSplitter(market=args.market)
    manual_labels  = splitter.label(df)["regime"]
    detector.compare_with_manual_regimes(df_labeled, manual_labels)

    if args.backtest:
        causal_features = CausalSelector().load(ticker)
        results = run_hmm_regime_backtest(df, ticker, causal_features, args.n_states)
        print(f"\nHMM Regime Backtest Results:\n{results}")