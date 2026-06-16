"""
granger.py
----------
Granger causality baseline for causal feature selection.

Granger causality tests whether past values of variable X help predict
variable Y beyond Y's own past values. It is NOT true causality — it is
predictive causality. We use it as a baseline to compare against PCMCI.

For the paper:
    - Granger is Table 1 baseline
    - PCMCI is the proposed method
    - We show PCMCI finds a stricter, more robust causal graph

Output:
    Dict mapping feature_name → {
        "causal":    bool,     # True if Granger-causal at significance level
        "min_pval":  float,    # minimum p-value across all tested lags
        "best_lag":  int,      # lag at which min p-value occurs
    }

Usage:
    from ml.src.causal.granger import GrangerCausality
    granger = GrangerCausality()
    results = granger.run(df, target="log_return_5d")
    causal_features = granger.get_causal_features(results)
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np
try:
    from statsmodels.tsa.stattools import grangercausalitytests
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    
    # Custom implementation of Benjamini-Hochberg FDR correction
    def multipletests(pvals, alpha=0.05, method="fdr_bh"):
        pvals = np.array(pvals)
        n = len(pvals)
        sort_idx = np.argsort(pvals)
        sorted_pvals = pvals[sort_idx]
        
        adjusted = np.zeros(n)
        prev_adj = 1.0
        for i in range(n - 1, -1, -1):
            adj = sorted_pvals[i] * n / (i + 1)
            adj = min(adj, prev_adj)
            adjusted[sort_idx[i]] = adj
            prev_adj = adj
            
        reject = adjusted < alpha
        return reject, adjusted, None, None

from ml.src.data.loader import _load_config

logger = logging.getLogger(__name__)


class GrangerCausality:
    """
    Tests Granger causality between each candidate feature and the target.

    For each feature X, tests H0: X does NOT Granger-cause target Y.
    Rejects H0 (marks as causal) if p-value < significance threshold
    at any tested lag.
    """

    def __init__(self, config_path: Optional[str] = None):
        cfg     = _load_config(config_path)
        granger = cfg["causal"]["granger"]

        self.max_lag     = granger["max_lag"]
        self.significance = granger["significance"]
        self.test        = granger["test"]
        self.target_col  = cfg["model"]["target"]

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        verbose: bool = False,
    ) -> dict[str, dict]:
        """
        Run Granger causality test for every feature column against target.

        Args:
            df:      Feature matrix with DatetimeIndex.
                     Must contain the target column.
            target:  Target column name. Defaults to config value.
            verbose: If True, log results for every feature.

        Returns:
            Dict: feature_name → {causal, min_pval, best_lag}
        """
        target = target or self.target_col
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")

        # Features = all columns except target
        feature_cols = [c for c in df.columns if c != target]
        logger.info(
            f"[granger] Testing {len(feature_cols)} features against '{target}' "
            f"(max_lag={self.max_lag}, significance={self.significance}) ..."
        )

        results = {}
        for col in feature_cols:
            result = self._test_feature(df[[target, col]].dropna(), target, col, verbose)
            results[col] = result

        # Multiple testing correction (Benjamini-Hochberg FDR)
        try:
            pvals = [results[col]["min_pval"] for col in feature_cols]
            reject, pvals_corrected, _, _ = multipletests(pvals, alpha=self.significance, method="fdr_bh")
            for col, rej, p_corr in zip(feature_cols, reject, pvals_corrected):
                results[col]["min_pval_corrected"] = float(p_corr)
                # Update causal flag to reflect FDR-corrected decision
                results[col]["causal"] = bool(rej)

            n_causal = sum(1 for r in results.values() if r["causal"])
            logger.info(
                f"[granger] Done. {n_causal}/{len(feature_cols)} features "
                f"Granger-causal after FDR-BH correction (alpha={self.significance})."
            )
        except Exception:
            # If correction fails for any reason, fall back to uncorrected counts
            n_causal = sum(1 for r in results.values() if r["causal"])
            logger.info(
                f"[granger] Done. {n_causal}/{len(feature_cols)} features "
                f"Granger-causal at p < {self.significance} (no FDR correction applied)."
            )

        return results

    def get_causal_features(self, results: dict[str, dict]) -> list[str]:
        """
        Extract list of feature names marked as causal.

        Args:
            results: Output of run()

        Returns:
            List of feature names where causal=True, sorted by min_pval.
        """
        causal = [
            (name, r["min_pval"])
            for name, r in results.items()
            if r["causal"]
        ]
        causal.sort(key=lambda x: x[1])   # sort by p-value ascending
        return [name for name, _ in causal]

    def summary_table(self, results: dict[str, dict]) -> pd.DataFrame:
        """
        Return a readable summary DataFrame of all Granger test results.
        Useful for notebooks and the paper.
        """
        rows = []
        for name, r in results.items():
            rows.append({
                "feature":  name,
                "causal":   r["causal"],
                "min_pval": round(r["min_pval"], 4),
                "best_lag": r["best_lag"],
            })
        df = pd.DataFrame(rows).sort_values("min_pval")
        return df

    # -----------------------------------------------------------------------
    # Private
    # -----------------------------------------------------------------------

    def _test_feature(
        self,
        df: pd.DataFrame,
        target: str,
        feature: str,
        verbose: bool,
    ) -> dict:
        """
        Run Granger test for a single feature.
        statsmodels grangercausalitytests expects [target, feature] column order.
        Returns dict with causal, min_pval, best_lag.
        """
        # Need at least 3 * max_lag rows for reliable test
        min_rows = 3 * self.max_lag

        # Work on contiguous non-missing blocks to avoid mixing disjoint
        # time segments (dropna() would create discontinuities which break
        # the lag interpretation). Find the longest contiguous block of
        # non-missing values and run the Granger test there.
        full = df[[target, feature]].copy()
        mask = full.notnull().all(axis=1).to_numpy()
        idxs = np.where(mask)[0]
        if len(idxs) < min_rows:
            logger.warning(
                f"[granger] {feature}: too few non-missing rows ({len(idxs)} < {min_rows}), skipping."
            )
            return {"causal": False, "min_pval": 1.0, "best_lag": -1}

        # Split into contiguous segments of indices
        splits = np.split(idxs, np.where(np.diff(idxs) != 1)[0] + 1)
        # Choose the longest segment
        best_seg = max(splits, key=lambda s: len(s))
        if len(best_seg) < min_rows:
            logger.warning(
                f"[granger] {feature}: longest contiguous block too short ({len(best_seg)} < {min_rows}), skipping."
            )
            return {"causal": False, "min_pval": 1.0, "best_lag": -1}

        data_seg = full.iloc[best_seg[0] : best_seg[-1] + 1]

        try:
            if not STATSMODELS_AVAILABLE:
                from sklearn.linear_model import LinearRegression
                from scipy import stats
                
                pvals = []
                y = data_seg[target].to_numpy()
                x = data_seg[feature].to_numpy()
                n = len(y)
                
                for lag in range(1, self.max_lag + 1):
                    Y_target = y[lag:]
                    
                    # Restricted model: lagged target only
                    X_restricted = np.column_stack([y[lag - i: n - i] for i in range(1, lag + 1)])
                    
                    # Unrestricted model: lagged target + lagged feature
                    X_unrestricted = np.column_stack([
                        y[lag - i: n - i] for i in range(1, lag + 1)
                    ] + [
                        x[lag - i: n - i] for i in range(1, lag + 1)
                    ])
                    
                    # restricted fit
                    lr_r = LinearRegression().fit(X_restricted, Y_target)
                    ssr_r = np.sum((Y_target - lr_r.predict(X_restricted)) ** 2)
                    
                    # unrestricted fit
                    lr_u = LinearRegression().fit(X_unrestricted, Y_target)
                    ssr_u = np.sum((Y_target - lr_u.predict(X_unrestricted)) ** 2)
                    
                    df1 = lag
                    df2 = len(Y_target) - 2 * lag - 1
                    
                    if ssr_u == 0 or df2 <= 0:
                        pval = 1.0
                    else:
                        F = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
                        pval = float(stats.f.sf(F, df1, df2))
                    
                    pvals.append((lag, pval))
            else:
                test_results = grangercausalitytests(
                    data_seg,
                    maxlag=self.max_lag,
                    verbose=False,
                )

                pvals = []
                for lag, result in test_results.items():
                    pval = result[0][self.test][1]
                    pvals.append((lag, pval))

            best_lag, min_pval = min(pvals, key=lambda x: x[1])
            is_causal = bool(min_pval < self.significance)

            if verbose:
                status = "CAUSAL" if is_causal else "not causal"
                logger.info(
                    f"[granger] {feature:40s} → {status:12s} "
                    f"(p={min_pval:.4f}, lag={best_lag})"
                )

            return {
                "causal":   is_causal,
                "min_pval": min_pval,
                "best_lag": best_lag,
            }

        except Exception as e:
            logger.warning(f"[granger] {feature}: test failed — {e}")
            return {"causal": False, "min_pval": 1.0, "best_lag": -1}