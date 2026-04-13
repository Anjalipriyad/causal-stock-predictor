"""
stability.py
------------
PCMCI causal feature stability analysis.

Addresses the central weakness in the paper's causal discovery claim:
running PCMCI once on a single window is insufficient to claim the
discovered features are robust. This module tests whether the same
features are discovered across multiple rolling windows.

Paper section: Section 3.2 — "Stability of Causal Feature Discovery"

Key output metric: Jaccard similarity between feature sets discovered
in different sub-windows. If mean Jaccard > 0.6, the features are
considered stable and the causal claim is valid.

If Jaccard < 0.4, the paper must either:
    (a) use "union" strategy instead of "intersection", or
    (b) report stability as a limitation and justify the single-window
        approach using data availability constraints.

Usage:
    from ml.src.causal.stability import PCMCIStabilityAnalyzer
    analyzer = PCMCIStabilityAnalyzer()
    report = analyzer.run(df_train, target="excess_return_5d", ticker="AAPL")
    print(report.summary())
"""

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

from ml.src.data.loader import _load_config

logger = logging.getLogger(__name__)


@dataclass
class StabilityReport:
    ticker: str
    target: str
    n_windows: int
    window_feature_sets: list[set]
    window_date_ranges: list[tuple[str, str]]
    pairwise_jaccard: list[float]
    mean_jaccard: float
    std_jaccard: float
    stable_core: set           # features appearing in ALL windows
    majority_core: set         # features appearing in > 50% of windows
    any_window: set            # union across all windows
    verdict: str               # "STABLE" | "MODERATE" | "UNSTABLE"

    def summary(self) -> str:
        lines = [
            f"\n{'='*65}",
            f"  PCMCI STABILITY REPORT — {self.ticker}",
            f"{'='*65}",
            f"  Windows tested:     {self.n_windows}",
            f"  Mean Jaccard:       {self.mean_jaccard:.3f}  (std={self.std_jaccard:.3f})",
            f"  Verdict:            {self.verdict}",
            f"",
            f"  Pairwise Jaccard scores:",
        ]
        for i, j in enumerate(self.pairwise_jaccard):
            lines.append(f"    Window pair {i+1}: {j:.3f}")

        lines += [
            f"",
            f"  Stable core features ({len(self.stable_core)}) — in ALL windows:",
        ]
        for f in sorted(self.stable_core):
            lines.append(f"    • {f}")

        lines += [
            f"",
            f"  Majority core features ({len(self.majority_core)}) — in >50% of windows:",
        ]
        for f in sorted(self.majority_core - self.stable_core):
            lines.append(f"    ○ {f}")

        lines += [
            f"",
            f"  Feature sets per window:",
        ]
        for i, (fset, (start, end)) in enumerate(
            zip(self.window_feature_sets, self.window_date_ranges)
        ):
            lines.append(f"    Window {i+1} ({start} → {end}): {sorted(fset)}")

        lines.append(f"{'='*65}")
        return "\n".join(lines)

    def recommended_strategy(self) -> str:
        """
        Based on stability, recommend a feature selection strategy.
        Returns 'intersection', 'majority', or 'union'.
        """
        if self.mean_jaccard >= 0.60:
            return "intersection"
        elif self.mean_jaccard >= 0.35:
            return "majority"
        else:
            return "union"


class PCMCIStabilityAnalyzer:
    """
    Tests PCMCI feature stability across multiple rolling sub-windows
    of the training data.

    Recommended use: call run() after splitting into training data only.
    Never pass test data — that would contaminate the causal discovery
    process with future information.
    """

    JACCARD_STABLE_THRESHOLD   = 0.60
    JACCARD_MODERATE_THRESHOLD = 0.35

    def __init__(self, config_path: Optional[str] = None, n_windows: int = 3):
        cfg = _load_config(config_path)
        self.n_windows  = n_windows
        self.target_col = cfg["model"]["target"]
        self.pc_alpha   = cfg["causal"]["pcmci"]["pc_alpha"]
        self.alpha_level = cfg["causal"]["pcmci"]["alpha_level"]

    def run(
        self,
        df_train: pd.DataFrame,
        target: Optional[str] = None,
        ticker: str = "UNKNOWN",
    ) -> StabilityReport:
        """
        Run PCMCI on n_windows rolling sub-windows and measure feature stability.

        Args:
            df_train: Training data ONLY — never pass test data.
            target:   Target column name.
            ticker:   Ticker for reporting.

        Returns:
            StabilityReport with full analysis.
        """
        target = target or self.target_col
        logger.info(
            f"[stability] Running PCMCI stability analysis for {ticker} "
            f"({self.n_windows} windows, {len(df_train)} training rows) ..."
        )

        window_size = len(df_train) // self.n_windows
        if window_size < 200:
            logger.warning(
                f"[stability] Window size {window_size} < 200 rows. "
                f"Results may be unreliable. Consider reducing n_windows."
            )

        feature_sets    = []
        date_ranges     = []

        for i in range(self.n_windows):
            start_idx = i * window_size
            # Last window takes all remaining rows
            end_idx   = (i + 1) * window_size if i < self.n_windows - 1 else len(df_train)
            window_df = df_train.iloc[start_idx:end_idx]

            start_date = str(window_df.index.min().date())
            end_date   = str(window_df.index.max().date())
            date_ranges.append((start_date, end_date))

            logger.info(
                f"[stability] Window {i+1}/{self.n_windows}: "
                f"{len(window_df)} rows ({start_date} → {end_date})"
            )

            features = self._run_pcmci_window(window_df, target, window_num=i+1)
            feature_sets.append(set(features))
            logger.info(
                f"[stability] Window {i+1} features ({len(features)}): {features}"
            )

        # Compute pairwise Jaccard similarity
        pairwise_jaccard = []
        for a, b in combinations(range(self.n_windows), 2):
            j = self._jaccard(feature_sets[a], feature_sets[b])
            pairwise_jaccard.append(j)

        mean_j = float(np.mean(pairwise_jaccard)) if pairwise_jaccard else 0.0
        std_j  = float(np.std(pairwise_jaccard))  if pairwise_jaccard else 0.0

        # Core feature sets
        stable_core   = set.intersection(*feature_sets) if feature_sets else set()
        any_window    = set.union(*feature_sets)        if feature_sets else set()
        n_threshold   = self.n_windows // 2 + 1        # majority = more than half
        majority_core = {
            f for f in any_window
            if sum(1 for fs in feature_sets if f in fs) >= n_threshold
        }

        # Verdict
        if mean_j >= self.JACCARD_STABLE_THRESHOLD:
            verdict = "STABLE"
        elif mean_j >= self.JACCARD_MODERATE_THRESHOLD:
            verdict = "MODERATE"
        else:
            verdict = "UNSTABLE"

        report = StabilityReport(
            ticker=ticker,
            target=target,
            n_windows=self.n_windows,
            window_feature_sets=feature_sets,
            window_date_ranges=date_ranges,
            pairwise_jaccard=pairwise_jaccard,
            mean_jaccard=mean_j,
            std_jaccard=std_j,
            stable_core=stable_core,
            majority_core=majority_core,
            any_window=any_window,
            verdict=verdict,
        )

        logger.info(
            f"[stability] Done. Mean Jaccard={mean_j:.3f} → {verdict}. "
            f"Stable core: {sorted(stable_core)}"
        )
        print(report.summary())
        return report

    def _run_pcmci_window(
        self, window_df: pd.DataFrame, target: str, window_num: int
    ) -> list[str]:
        """Run PCMCI on a single window and return discovered causal features."""
        from ml.src.causal.pcmci import PCMCIDiscovery
        try:
            pcmci   = PCMCIDiscovery()
            results = pcmci.run(window_df, target=target)
            return pcmci.get_causal_features(results)
        except Exception as e:
            logger.warning(f"[stability] Window {window_num} PCMCI failed: {e}")
            return []

    @staticmethod
    def _jaccard(set_a: set, set_b: set) -> float:
        """Jaccard similarity: |A ∩ B| / |A ∪ B|. Returns 1.0 if both empty."""
        if not set_a and not set_b:
            return 1.0
        union = set_a | set_b
        if not union:
            return 0.0
        return len(set_a & set_b) / len(union)

    def select_features_by_stability(
        self, report: StabilityReport
    ) -> tuple[list[str], str]:
        """
        Given a stability report, return the recommended feature list
        and the strategy used.

        Returns:
            (feature_list, strategy_name)
        """
        strategy = report.recommended_strategy()
        if strategy == "intersection":
            features = sorted(report.stable_core)
            logger.info(
                f"[stability] Using stable core ({len(features)} features) "
                f"— Jaccard={report.mean_jaccard:.3f} ≥ 0.60"
            )
        elif strategy == "majority":
            features = sorted(report.majority_core)
            logger.info(
                f"[stability] Using majority core ({len(features)} features) "
                f"— Jaccard={report.mean_jaccard:.3f} ∈ [0.35, 0.60)"
            )
        else:
            features = sorted(report.any_window)
            logger.warning(
                f"[stability] Using full union ({len(features)} features) "
                f"— Jaccard={report.mean_jaccard:.3f} < 0.35. "
                f"Feature instability should be disclosed in the paper."
            )
        return features, strategy


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="Run PCMCI stability analysis for a ticker"
    )
    parser.add_argument("--ticker",    type=str, required=True)
    parser.add_argument("--market",    type=str, default="us",
                        choices=["us", "india"])
    parser.add_argument("--n-windows", type=int, default=3)
    args = parser.parse_args()

    ticker = args.ticker.upper()
    cfg    = _load_config()

    # Load feature matrix
    if args.market == "india" or ticker in ("NIFTY", "^NSEI"):
        from ml.src.data.nifty_loader import NiftyLoader
        feat_path = NiftyLoader().out_dir / "NIFTY_features.csv"
        target    = "log_return_5d"
    else:
        from ml.src.features.pipeline import FeaturePipeline
        feat_path = FeaturePipeline().features_dir / f"{ticker}_features.csv"
        target    = cfg["model"]["target"]

    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)

    # Use only training split — no leakage
    train_end = int(len(df) * cfg["model"]["train_ratio"])
    df_train  = df.iloc[:train_end]

    analyzer = PCMCIStabilityAnalyzer(n_windows=args.n_windows)
    report   = analyzer.run(df_train, target=target, ticker=ticker)

    features, strategy = analyzer.select_features_by_stability(report)
    print(f"\nRecommended strategy: {strategy}")
    print(f"Recommended features ({len(features)}): {features}")