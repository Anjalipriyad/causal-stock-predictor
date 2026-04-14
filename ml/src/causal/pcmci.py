"""
pcmci.py
--------
PCMCI causal discovery using the tigramite library.

PCMCI (Peter-Clark Momentary Conditional Independence) finds lagged
causal links between time series while controlling for confounders.
Unlike Granger, it tests conditional independence — X causes Y only if
the link holds after conditioning on all other variables.

This is the core novelty of the paper.

For the paper:
    - PCMCI produces a causal graph: which features causally drive returns
      and at what lag (1-5 days)
    - We compare this against Granger and show PCMCI is stricter
    - Models trained on PCMCI features are more robust across regimes

Output:
    {
        "causal_links": {
            "feature_name": {
                "causal":    bool,
                "best_lag":  int,
                "pval":      float,
                "val":       float,    # test statistic value
            }
        },
        "causal_graph":  np.ndarray,   # full adjacency matrix
        "p_matrix":      np.ndarray,   # p-value matrix
        "val_matrix":    np.ndarray,   # test statistic matrix
        "var_names":     list[str],    # variable names in matrix order
    }

Usage:
    from ml.src.causal.pcmci import PCMCIDiscovery
    pcmci = PCMCIDiscovery()
    results = pcmci.run(df, target="log_return_5d")
    causal_features = pcmci.get_causal_features(results)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ml.src.data.loader import _load_config

logger = logging.getLogger(__name__)


class PCMCIDiscovery:
    """
    Wraps tigramite's PCMCI algorithm for causal discovery on
    the feature matrix.

    Finds which features have statistically significant lagged
    causal links to the target return variable.
    """

    def __init__(self, config_path: Optional[str] = None):
        cfg   = _load_config(config_path)
        pcmci = cfg["causal"]["pcmci"]

        self.tau_min      = pcmci["tau_min"]
        self.tau_max      = pcmci["tau_max"]
        self.pc_alpha     = pcmci["pc_alpha"]
        self.cond_ind_test = pcmci["cond_ind_test"]
        self.alpha_level  = pcmci["alpha_level"]
        self.target_col   = cfg["model"]["target"]

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        exclude_target: bool = False,
    ) -> dict:
        """
        Run PCMCI on the full feature matrix.

        Args:
            df:     Feature matrix with DatetimeIndex.
                    All columns are treated as candidate causal variables.
                    Target column must be included.
            target: Target column name. Defaults to config value.

        Returns:
            Results dict — see module docstring for structure.
        """
        target = target or self.target_col
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not in DataFrame columns.")

        # tigramite import here — lazy import so rest of codebase
        # doesn't break if tigramite isn't installed
        try:
            from tigramite import data_processing as pp
            from tigramite.pcmci import PCMCI
            from tigramite.independence_tests.parcorr import ParCorr
        except ImportError:
            raise ImportError(
                "tigramite is required for PCMCI. "
                "Install with: pip install tigramite"
            )

        df     = df.copy().sort_index()
        # Note: do NOT dropna() here — tigramite expects aligned series and
        # users may prefer to dropna earlier. We'll drop rows only when building
        # the tigramite DataFrame below (safe conversion to float).

        # Optionally run PCMCI on features-only to avoid including the
        # forward-looking target column directly inside the PCMCI variable set.
        if exclude_target:
            cols = [c for c in df.columns if c != target]
            target_idx = None
        else:
            cols = list(df.columns)
            target_idx = cols.index(target)

        logger.info(
            f"[pcmci] Running PCMCI on {len(cols)} variables "
            f"(tau_min={self.tau_min}, tau_max={self.tau_max}, "
            f"cond_ind_test={self.cond_ind_test}) ..."
        )

        # Build tigramite dataframe
        # Build numeric array for tigramite; if we excluded the target,
        # use df[cols], otherwise use the full df. Keep the time index
        # aligned to the rows actually passed to tigramite (after dropna).
        data_frame_for_pcmci = df[cols].dropna()
        dropped = len(df) - len(data_frame_for_pcmci)
        if dropped > 0:
            logger.info(f"[pcmci] Dropped {dropped} rows with NaNs before PCMCI (remaining={len(data_frame_for_pcmci)})")
        data = data_frame_for_pcmci.values.astype(float)
        # datatime must match the number of rows in `data_frame_for_pcmci`.
        dataframe = pp.DataFrame(
            data,
            var_names=cols,
            datatime=np.arange(len(data_frame_for_pcmci)),
        )

        # Conditional independence test
        cit = self._get_cit()

        # Run PCMCI
        pcmci_obj = PCMCI(
            dataframe=dataframe,
            cond_ind_test=cit,
            verbosity=0,
        )

        results = pcmci_obj.run_pcmci(
            tau_min=self.tau_min,
            tau_max=self.tau_max,
            pc_alpha=self.pc_alpha,
        )

        # Extract causal links TO the target variable
        # If we excluded the target from the PCMCI run, construct an
        # alternative "causal_links" mapping based on outgoing links from
        # each feature to the rest of the feature set (minimal p-value
        # across outgoing edges). This avoids inserting the forward-looking
        # target into the variable set while still producing a usable
        # pcmci_results structure for the selector.
        causal_links = self._extract_links(
            results, cols, target_idx
        )

        logger.info(
            f"[pcmci] Done. {sum(1 for v in causal_links.values() if v['causal'])} "
            f"causal links found to '{target}'."
        )

        return {
            "causal_links": causal_links,
            "p_matrix":     results["p_matrix"],
            "val_matrix":   results["val_matrix"],
            "var_names":    cols,
            "target":       target,
            "target_idx":   target_idx,
        }

    def get_causal_features(self, results: dict) -> list[str]:
        """
        Extract list of feature names with significant causal links to target.
        Sorted by p-value ascending (most significant first).

        Args:
            results: Output of run()

        Returns:
            List of feature names (excludes target itself).
        """
        target = results.get("target", self.target_col)
        causal = [
            (name, info["pval"])
            for name, info in results["causal_links"].items()
            if info["causal"] and name != target
        ]
        causal.sort(key=lambda x: x[1])
        return [name for name, _ in causal]

    def summary_table(self, results: dict) -> pd.DataFrame:
        """
        Return a readable summary DataFrame of PCMCI results.
        Useful for notebooks and the paper (Table 1).
        """
        rows = []
        for name, info in results["causal_links"].items():
            rows.append({
                "feature":  name,
                "causal":   info["causal"],
                "pval":     round(info["pval"], 4),
                "val":      round(info["val"], 4),
                "best_lag": info["best_lag"],
            })
        df = pd.DataFrame(rows).sort_values("pval")
        return df

    def causal_graph_matrix(self, results: dict) -> pd.DataFrame:
        """
        Return the full p-value matrix as a labelled DataFrame.
        Rows = source variables, cols = target variables.
        Useful for visualising the full causal graph in notebooks.
        """
        var_names = results["var_names"]
        # p_matrix shape: (n_vars, n_vars, tau_max+1)
        # Take minimum p-value across all lags for each pair
        p_min = results["p_matrix"].min(axis=2)
        return pd.DataFrame(p_min, index=var_names, columns=var_names)

    # -----------------------------------------------------------------------
    # Private
    # -----------------------------------------------------------------------

    def _get_cit(self):
        """Instantiate the conditional independence test from config."""
        test = self.cond_ind_test
        if test == "ParCorr":
            from tigramite.independence_tests.parcorr import ParCorr
            return ParCorr(significance="analytic")
        elif test == "GPDC":
            from tigramite.independence_tests.gpdc import GPDC
            return GPDC(significance="analytic")
        elif test == "CMIknn":
            from tigramite.independence_tests.cmiknn import CMIknn
            return CMIknn(significance="shuffle_test")
        else:
            raise ValueError(
                f"Unknown cond_ind_test: '{test}'. "
                "Choose from: ParCorr | GPDC | CMIknn"
            )

    def _extract_links(
        self,
        results: dict,
        var_names: list[str],
        target_idx: Optional[int],
    ) -> dict[str, dict]:
        """
        Extract causal-link summaries.

        If `target_idx` is provided, extract links pointing TO that target
        variable (original behaviour). If `target_idx` is None (we ran
        PCMCI on a feature-only set to avoid forward-looking target leakage),
        return for each feature the strongest outgoing link to any other
        variable (used as a conservative proxy for feature importance).
        """
        p_matrix   = results["p_matrix"]
        val_matrix = results["val_matrix"]
        n_vars     = len(var_names)

        causal_links = {}

        if target_idx is not None:
            # Original behaviour: extract links TO the target variable only
            for i, name in enumerate(var_names):
                if i == target_idx:
                    continue   # skip self-link

                # p-values for this variable → target across all lags
                pvals = p_matrix[i, target_idx, :]
                vals  = val_matrix[i, target_idx, :]

                # Best (minimum) p-value across lags
                best_tau_idx = int(np.nanargmin(pvals))
                best_pval    = float(pvals[best_tau_idx])
                best_val     = float(vals[best_tau_idx])
                best_lag     = best_tau_idx + self.tau_min

                causal_links[name] = {
                    "causal":   best_pval < self.alpha_level,
                    "pval":     best_pval,
                    "val":      best_val,
                    "best_lag": best_lag,
                }
        else:
            # Excluded target: produce a feature-centric summary based on the
            # strongest outgoing link from each feature to any other variable.
            # This provides a conservative proxy for a feature's causal
            # importance when the explicit target is intentionally omitted
            # from the PCMCI variable set to avoid forward-looking leakage.
            tau_range = p_matrix.shape[2]
            for i, name in enumerate(var_names):
                # Exclude self-links by setting them to nan
                pvals_i = np.array(p_matrix[i, :, :], copy=True)
                pvals_i[i, :] = np.nan
                # Flatten and find best (treat non-finite as non-significant)
                pvals_flat = np.nan_to_num(pvals_i, nan=1.0, posinf=1.0, neginf=1.0)
                best_flat_idx = int(np.argmin(pvals_flat))
                j_idx, tau_idx = np.unravel_index(best_flat_idx, (n_vars, tau_range))
                raw_pval = pvals_i[j_idx, tau_idx]
                best_pval = float(raw_pval) if np.isfinite(raw_pval) else 1.0
                raw_val = val_matrix[i, j_idx, tau_idx]
                best_val = float(raw_val) if np.isfinite(raw_val) else 0.0
                best_lag = int(tau_idx + self.tau_min)

                causal_links[name] = {
                    "causal":   best_pval < self.alpha_level,
                    "pval":     best_pval,
                    "val":      best_val,
                    "best_lag": best_lag,
                }

        return causal_links