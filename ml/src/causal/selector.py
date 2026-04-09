"""
selector.py
-----------
Combines Granger and PCMCI results into the final causal feature list.
This is the output that gets saved to saved_models/causal_features_{ticker}.json
and loaded by the model training + inference pipeline.

Strategy (from config):
    "intersection" — feature must appear causal in BOTH Granger AND PCMCI
                     Stricter. Better for the paper — fewer false positives.
    "union"        — feature appears causal in EITHER Granger OR PCMCI
                     More permissive. Keeps more features.

Also enforces:
    min_causal_features — abort if too few survive (likely data problem)
    max_causal_features — cap to avoid overfitting

Saves:
    saved_models/causal_features_{ticker}.json
    {
        "ticker":         "AAPL",
        "strategy":       "intersection",
        "n_features":     8,
        "features": [
            {
                "name":          "vix_change_1d",
                "granger_causal": true,
                "granger_pval":   0.003,
                "pcmci_causal":   true,
                "pcmci_pval":     0.008,
                "pcmci_lag":      1
            },
            ...
        ]
    }

Usage:
    from ml.src.causal.selector import CausalSelector
    selector = CausalSelector()
    features = selector.select(
        ticker="AAPL",
        granger_results=granger.run(df),
        pcmci_results=pcmci.run(df),
    )
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ml.src.data.loader import _load_config

logger = logging.getLogger(__name__)


class CausalSelector:
    """
    Combines Granger + PCMCI results and saves the final causal feature list.
    """

    def __init__(self, config_path: Optional[str] = None, cfg: Optional[dict] = None):
        # Allow passing an already-loaded config dict to keep runtime overrides
        cfg = cfg if cfg is not None else _load_config(config_path)
        self.cfg = cfg
        selector = cfg["causal"]["selector"]

        self.strategy           = selector["strategy"]
        self.min_causal_features = selector["min_causal_features"]
        self.max_causal_features = selector["max_causal_features"]
        self.target_col          = cfg["model"]["target"]

        root = Path(__file__).resolve().parents[3]
        self.models_dir = root / cfg["saved_models"]["dir"]
        self.models_dir.mkdir(parents=True, exist_ok=True)

        fname_template = cfg["saved_models"]["causal_features_filename"]
        self._fname_template = fname_template   # "causal_features_{ticker}.json"

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def select(
        self,
        ticker: str,
        granger_results: dict[str, dict],
        pcmci_results: dict,
        save: bool = True,
        force_adaptive: bool = False,
    ) -> list[str]:
        """
        Combine Granger + PCMCI results and return the final causal feature list.

        Args:
            ticker:          e.g. "AAPL"
            granger_results: output of GrangerCausality.run()
            pcmci_results:   output of PCMCIDiscovery.run()
            save:            if True, save to saved_models/

        Returns:
            List of feature names to use for model training.
        """
        ticker = ticker.upper()
        logger.info(
            f"[selector] Selecting causal features for {ticker} "
            f"(strategy='{self.strategy}') ..."
        )

        # Build combined feature table
        table = self._build_table(granger_results, pcmci_results)

        # Helper: return sorted feature list from a selection DataFrame
        def _features_from_df(df_sel):
            return df_sel.sort_values("pcmci_pval")["feature"].tolist()

        feature_names = []
        selection_info = {"method": self.strategy, "note": "initial"}

        if self.strategy == "intersection":
            # First try strict intersection using boolean causal flags
            sel = table[table["granger_causal"] & table["pcmci_causal"]]
            if len(sel) >= self.min_causal_features and not force_adaptive:
                feature_names = _features_from_df(sel)
                selection_info.update({"stage": "intersection_bool", "n": len(feature_names)})
            else:
                # Adaptive relaxation: try relaxed p-value thresholds
                granger_sig = float(self.cfg.get("causal", {}).get("granger", {}).get("significance", 0.05))
                pcmci_alpha = float(self.cfg.get("causal", {}).get("pcmci", {}).get("alpha_level", 0.01))

                granger_thresholds = [granger_sig, min(0.1, max(granger_sig, 0.1))]
                pcmci_thresholds = [pcmci_alpha, 0.02, 0.05]

                found = False
                stage = 0
                for gth in granger_thresholds:
                    for pth in pcmci_thresholds:
                        stage += 1
                        sel = table[(table["granger_pval"] <= gth) & (table["pcmci_pval"] <= pth)]
                        if len(sel) >= self.min_causal_features:
                            feature_names = _features_from_df(sel)
                            selection_info.update({
                                "method": "adaptive_intersection",
                                "stage": stage,
                                "granger_pval": gth,
                                "pcmci_pval": pth,
                                "n": len(feature_names),
                            })
                            found = True
                            break
                    if found:
                        break

                if not found:
                    # Fall back to union (more permissive)
                    sel_union = table[(table["granger_pval"] <= granger_sig) | (table["pcmci_pval"] <= pcmci_alpha)]
                    if len(sel_union) >= self.min_causal_features:
                        feature_names = _features_from_df(sel_union)
                        selection_info.update({"method": "fallback_union", "n": len(feature_names)})
                    else:
                        # As a last resort, pick top features by PCMCI p-value
                        fallback = table.sort_values("pcmci_pval")["feature"].tolist()
                        # Ensure we have at least min_causal_features by taking top N
                        if len(fallback) >= self.min_causal_features:
                            feature_names = fallback[: self.min_causal_features]
                            selection_info.update({"method": "fallback_top_pcmci", "n": len(feature_names)})
                        else:
                            raise ValueError(
                                f"[selector] Only {len(fallback)} candidate features available for {ticker} (min={self.min_causal_features})."
                            )

        elif self.strategy == "union":
            sel = table[table["granger_causal"] | table["pcmci_causal"]]
            feature_names = _features_from_df(sel)
            selection_info.update({"stage": "union_bool", "n": len(feature_names)})

        else:
            raise ValueError(
                f"Unknown strategy '{self.strategy}'. Use 'intersection' or 'union'."
            )

        # Enforce limits (cap at max, ensure min satisfied)
        # If feature_names is longer than max, _enforce_limits will cap it.
        feature_names = self._enforce_limits(feature_names, ticker)

        # Build output record and attach selection metadata
        record = self._build_record(ticker, table, feature_names)
        record["selection_info"] = selection_info
        record["strategy_used"] = self.strategy

        if save:
            self._save(ticker, record)

        # Expose last selection metadata for callers (useful for ablation scripts)
        try:
            self._last_selection_info = selection_info
            self._last_selected_features = feature_names
        except Exception:
            pass

        logger.info(
            f"[selector] Final causal features for {ticker} "
            f"({len(feature_names)}): {feature_names}  | info: {selection_info}"
        )
        return feature_names

    def load(self, ticker: str) -> list[str]:
        """
        Load previously saved causal feature list from disk.

        Args:
            ticker: e.g. "AAPL"

        Returns:
            List of feature names.
        """
        path = self._path(ticker)
        if not path.exists():
            raise FileNotFoundError(
                f"No causal features saved for {ticker}. "
                f"Run selector.select('{ticker}', ...) first."
            )
        with open(path, "r") as f:
            record = json.load(f)
        features = [item["name"] for item in record["features"]]
        logger.info(
            f"[selector] Loaded {len(features)} causal features for {ticker}."
        )
        return features

    def load_record(self, ticker: str) -> dict:
        """Load full JSON record including p-values and lag info."""
        path = self._path(ticker)
        if not path.exists():
            raise FileNotFoundError(
                f"No causal features saved for {ticker}."
            )
        with open(path, "r") as f:
            return json.load(f)

    def comparison_table(
        self,
        granger_results: dict[str, dict],
        pcmci_results: dict,
    ) -> pd.DataFrame:
        """
        Return a full comparison table of Granger vs PCMCI results.
        Used for notebook 03 and paper Table 1.
        """
        return self._build_table(granger_results, pcmci_results)

    # -----------------------------------------------------------------------
    # Private
    # -----------------------------------------------------------------------

    def _build_table(
        self,
        granger_results: dict[str, dict],
        pcmci_results: dict,
    ) -> pd.DataFrame:
        """
        Build a unified comparison table from both result dicts.
        One row per feature.
        """
        pcmci_links = pcmci_results.get("causal_links", {})
        all_features = set(granger_results.keys()) | set(pcmci_links.keys())
        # Exclude target variable itself
        all_features.discard(self.target_col)

        rows = []
        for feat in sorted(all_features):
            g = granger_results.get(feat, {})
            p = pcmci_links.get(feat, {})

            rows.append({
                "feature":        feat,
                "granger_causal": bool(g.get("causal", False)),
                "granger_pval":   round(float(g.get("min_pval", 1.0)), 4),
                "granger_lag":    int(g.get("best_lag", -1)),
                "pcmci_causal":   bool(p.get("causal", False)),
                "pcmci_pval":     round(float(p.get("pval", 1.0)), 4),
                "pcmci_val":      round(float(p.get("val", 0.0)), 4),
                "pcmci_lag":      int(p.get("best_lag", -1)),
            })

        return pd.DataFrame(rows).sort_values("pcmci_pval")

    def _enforce_limits(
        self, feature_names: list[str], ticker: str
    ) -> list[str]:
        """Apply min/max feature count limits."""
        if len(feature_names) < self.min_causal_features:
            raise ValueError(
                f"[selector] Only {len(feature_names)} causal features found for {ticker} "
                f"(min={self.min_causal_features}). "
                "Try 'union' strategy or check data quality."
            )
        if len(feature_names) > self.max_causal_features:
            logger.warning(
                f"[selector] {len(feature_names)} causal features — "
                f"capping at {self.max_causal_features}."
            )
            feature_names = feature_names[: self.max_causal_features]
        return feature_names

    def _build_record(
        self,
        ticker: str,
        table: pd.DataFrame,
        feature_names: list[str],
    ) -> dict:
        """Build the JSON record to save."""
        selected_rows = table[table["feature"].isin(feature_names)]
        features_list = []
        for _, row in selected_rows.iterrows():
            features_list.append({
                "name":           row["feature"],
                "granger_causal": row["granger_causal"],
                "granger_pval":   row["granger_pval"],
                "pcmci_causal":   row["pcmci_causal"],
                "pcmci_pval":     row["pcmci_pval"],
                "pcmci_lag":      row["pcmci_lag"],
            })
        return {
            "ticker":     ticker,
            "strategy":   self.strategy,
            "n_features": len(features_list),
            "features":   features_list,
        }

    def _save(self, ticker: str, record: dict) -> None:
        """Save causal features JSON to saved_models/."""
        path = self._path(ticker)
        with open(path, "w") as f:
            json.dump(record, f, indent=2)
        logger.info(f"[selector] Saved causal features → {path.name}")

    def _path(self, ticker: str) -> Path:
        """Resolve path for causal features JSON file."""
        fname = self._fname_template.replace("{ticker}", ticker.upper())
        return self.models_dir / fname


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import pandas as pd
    from ml.src.causal.granger import GrangerCausality
    from ml.src.causal.pcmci import PCMCIDiscovery
    from ml.src.data.loader import _load_config

    parser = argparse.ArgumentParser(description="Run causal discovery for a ticker")
    parser.add_argument("--ticker",   type=str, required=True, help="e.g. AAPL")
    parser.add_argument("--strategy", type=str, default="intersection",
                        help="intersection | union")
    args = parser.parse_args()

    cfg     = _load_config()
    ticker  = args.ticker.upper()
    root    = Path(__file__).resolve().parents[3]
    feat_path = root / cfg["data"]["processed_dir"] / "features" / f"{ticker}_features.csv"

    if not feat_path.exists():
        print(f"ERROR: No feature matrix found at {feat_path}")
        print(f"Run first: python -m ml.src.features.pipeline --ticker {ticker}")
        exit(1)

    df     = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    target = cfg["model"]["target"]

    print(f"\n[1/3] Running Granger causality on {len(df)} rows...")
    granger         = GrangerCausality()
    granger_results = granger.run(df, target=target, verbose=True)

    print(f"\n[2/3] Running PCMCI...")
    pcmci         = PCMCIDiscovery()
    pcmci_results = pcmci.run(df, target=target)

    print(f"\n[3/3] Selecting causal features (strategy={args.strategy})...")
    selector          = CausalSelector()
    selector.strategy = args.strategy
    features          = selector.select(ticker, granger_results, pcmci_results, save=True)

    print(f"\n=== CAUSAL FEATURES FOR {ticker} ({len(features)}) ===")
    for i, f in enumerate(features, 1):
        print(f"  {i:2d}. {f}")
    print(f"\nSaved to: saved_models/causal_features_{ticker}.json")