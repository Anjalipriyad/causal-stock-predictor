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

        df     = df.copy().sort_index().dropna()
        cols   = list(df.columns)
        target_idx = cols.index(target)

        logger.info(
            f"[pcmci] Running PCMCI on {len(cols)} variables "
            f"(tau_min={self.tau_min}, tau_max={self.tau_max}, "
            f"cond_ind_test={self.cond_ind_test}) ..."
        )

        # Build tigramite dataframe
        data     = df.values.astype(float)
        dataframe = pp.DataFrame(
            data,
            var_names=cols,
            datatime=np.arange(len(df)),
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
        target_idx: int,
    ) -> dict[str, dict]:
        """
        Extract all causal links pointing TO the target variable.

        p_matrix shape:   (n_vars, n_vars, tau_max - tau_min + 1)
        val_matrix shape: same

        p_matrix[i, j, tau] = p-value for link X_i(t-tau) → X_j(t)
        We want all i where j = target_idx.
        """
        p_matrix   = results["p_matrix"]
        val_matrix = results["val_matrix"]
        n_vars     = len(var_names)

        causal_links = {}
        for i, name in enumerate(var_names):
            if i == target_idx:
                continue   # skip self-link

            # p-values for this variable → target across all lags
            pvals = p_matrix[i, target_idx, :]
            vals  = val_matrix[i, target_idx, :]

            # Best (minimum) p-value across lags
            best_tau_idx = int(np.argmin(pvals))
            best_pval    = float(pvals[best_tau_idx])
            best_val     = float(vals[best_tau_idx])
            best_lag     = best_tau_idx + self.tau_min

            causal_links[name] = {
                "causal":   best_pval < self.alpha_level,
                "pval":     best_pval,
                "val":      best_val,
                "best_lag": best_lag,
            }

        return causal_links