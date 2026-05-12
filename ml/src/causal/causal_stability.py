"""
causal_stability.py
-------------------
Per-regime causal discovery and stability analysis.

This is the CENTRAL NOVEL CONTRIBUTION of the paper: running causal
discovery (Granger + PCMCI) separately on each Indian market regime and
analysing how the causal graph changes across structurally distinct periods.

Key finding: causal features are NOT stable across regimes. Features that
drive returns during a rate-hike cycle (VIX, yield signals) differ from
those during a stimulus melt-up (PE momentum, sentiment). This instability
is WHY a globally-trained causal model degrades under regime shifts.

Paper section: Section 4 — "Per-Regime Causal Discovery"
Paper tables: Table 3 — Causal Feature Stability Matrix

Usage:
    from ml.src.causal.causal_stability import CausalStabilityAnalyser
    analyser = CausalStabilityAnalyser(market='india')
    regime_features = analyser.run(df_train, ticker='NIFTY')
    stability = analyser.analyse_stability(regime_features)
    analyser.print_stability_table(stability)
    analyser.save_stability_report(stability, 'NIFTY', models_dir)
"""

import json
import logging
import time
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.src.data.loader import _load_config
from ml.src.causal.stability import jaccard_similarity

logger = logging.getLogger(__name__)


class CausalStabilityAnalyser:
    """
    Runs causal discovery (Granger + optionally PCMCI) on each market regime
    independently, then analyses how the causal feature set changes across
    structurally distinct periods.
    """

    MIN_REGIME_ROWS = 200

    def __init__(
        self,
        config_path: Optional[str] = None,
        market: str = "india",
        skip_pcmci: bool = False,
    ):
        self.cfg = _load_config(config_path)
        self.config_path = config_path
        self.market = market
        self.skip_pcmci = skip_pcmci

        from ml.src.evaluation.regime_splitter import RegimeSplitter
        self.splitter = RegimeSplitter(config_path, market=market)

    # -----------------------------------------------------------------------
    # Public — run per-regime causal discovery
    # -----------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        ticker: str,
        market: str = "india",
    ) -> dict[str, list[str]]:
        """
        Run causal discovery on each regime and the full training set.

        For each regime with >= MIN_REGIME_ROWS rows:
          1. Run GrangerCausality.run() on that regime's data
          2. Run PCMCIDiscovery.run() (unless skip_pcmci=True)
          3. Run CausalSelector.select() with strategy='union', save=False
          4. Store the resulting feature set

        Also runs on the FULL training set (global causal features).

        Args:
            df:     Feature matrix (typically training split only).
            ticker: Stock ticker (e.g. 'NIFTY').
            market: Market identifier ('india' or 'us').

        Returns:
            Dict: {'global': [...], 'regime_name': [...], ...}
        """
        from ml.src.causal.granger import GrangerCausality
        from ml.src.causal.pcmci import PCMCIDiscovery
        from ml.src.causal.selector import CausalSelector

        ticker = ticker.upper()
        logger.info(
            f"[causal_stability] Starting per-regime causal discovery for {ticker} "
            f"(market={market}, skip_pcmci={self.skip_pcmci})"
        )

        # Determine target column
        target = self._resolve_target(df, ticker)

        # Identify columns to exclude from causal discovery
        # (target columns contain future prices via shift(-5))
        target_cols = [
            c for c in df.columns
            if (c.startswith("log_return_") and c != "log_return_1d")
            or c.startswith("excess_return_")
        ]

        regime_splits = self.splitter.split_all(df)
        result = {}

        # ── Global (full training set) ──────────────────────────────────
        logger.info("[causal_stability] Running on FULL training set (global) ...")
        global_features = self._run_causal_on_subset(
            df, target, target_cols, ticker, "global"
        )
        result["global"] = global_features

        # ── Per-regime ──────────────────────────────────────────────────
        total_regimes = len(regime_splits)
        completed = 0

        for regime_name, regime_df in regime_splits.items():
            completed += 1
            n_rows = len(regime_df)

            if n_rows < self.MIN_REGIME_ROWS:
                logger.warning(
                    f"[causal_stability] Skipping {regime_name} — "
                    f"only {n_rows} rows (need >= {self.MIN_REGIME_ROWS})"
                )
                continue

            logger.info(
                f"[causal_stability] Regime {completed}/{total_regimes}: "
                f"{regime_name} ({n_rows} rows) ..."
            )
            start_t = time.time()

            regime_features = self._run_causal_on_subset(
                regime_df, target, target_cols, ticker, regime_name
            )
            result[regime_name] = regime_features

            elapsed = time.time() - start_t
            elapsed_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{elapsed/60:.1f}min"
            logger.info(
                f"[causal_stability] Regime {completed}/{total_regimes}: "
                f"{regime_name} — {len(regime_features)} causal features ({elapsed_str})"
            )

        logger.info(
            f"[causal_stability] Per-regime discovery complete. "
            f"{len(result)} sets (including global)."
        )
        return result

    # -----------------------------------------------------------------------
    # Public — stability analysis
    # -----------------------------------------------------------------------

    def analyse_stability(self, regime_feature_sets: dict[str, list[str]]) -> dict:
        """
        Analyse how the causal graph changes across regimes.

        Computes:
          - features_by_regime: which features are present per regime
          - feature_frequency: how many regimes each feature appears in
          - jaccard_pairwise: Jaccard similarity between each pair of regime sets
          - stable_features: features in >= 75% of regimes
          - regime_specific_features: features appearing in only 1 regime

        Args:
            regime_feature_sets: Output of run().

        Returns:
            Structured dict with all stability metrics.
        """
        logger.info("[causal_stability] Analysing causal feature stability ...")

        # Separate global from per-regime
        global_features = set(regime_feature_sets.get("global", []))
        regime_sets = {
            k: set(v) for k, v in regime_feature_sets.items()
            if k != "global"
        }
        n_regimes = len(regime_sets)

        if n_regimes == 0:
            logger.warning("[causal_stability] No regimes with causal features found.")
            return {"error": "no_regimes"}

        # All features seen across any regime (excluding global)
        all_features = set()
        for s in regime_sets.values():
            all_features |= s
        all_features |= global_features

        # Feature frequency: how many regimes each feature appears in
        feature_frequency = {}
        for feat in sorted(all_features):
            count = sum(1 for s in regime_sets.values() if feat in s)
            feature_frequency[feat] = count

        # Jaccard similarity between each pair of regime feature sets
        jaccard_pairwise = {}
        regime_names = sorted(regime_sets.keys())
        for r1, r2 in combinations(regime_names, 2):
            j = jaccard_similarity(regime_sets[r1], regime_sets[r2])
            jaccard_pairwise[f"{r1}_vs_{r2}"] = round(j, 4)

        # Jaccard of each regime vs global
        jaccard_vs_global = {}
        for name, fset in regime_sets.items():
            j = jaccard_similarity(fset, global_features)
            jaccard_vs_global[name] = round(j, 4)

        # Stable features: in >= 75% of regimes
        threshold = max(1, int(np.ceil(n_regimes * 0.75)))
        stable_features = [
            f for f, count in feature_frequency.items()
            if count >= threshold
        ]

        # Regime-specific features: appear in exactly 1 regime
        regime_specific_features = [
            f for f, count in feature_frequency.items()
            if count == 1
        ]

        result = {
            "n_regimes": n_regimes,
            "global_features": sorted(global_features),
            "features_by_regime": {k: sorted(v) for k, v in regime_sets.items()},
            "feature_frequency": feature_frequency,
            "jaccard_pairwise": jaccard_pairwise,
            "jaccard_vs_global": jaccard_vs_global,
            "stable_features": sorted(stable_features),
            "regime_specific_features": sorted(regime_specific_features),
            "stability_threshold": f">= {threshold}/{n_regimes} regimes (75%)",
        }

        logger.info(
            f"[causal_stability] Stability analysis complete: "
            f"{len(stable_features)} stable features, "
            f"{len(regime_specific_features)} regime-specific features"
        )
        return result

    # -----------------------------------------------------------------------
    # Public — print Table 3
    # -----------------------------------------------------------------------

    def print_stability_table(self, stability_results: dict) -> None:
        """
        Print a clean table showing which features are causal in which regimes.

        Rows = features
        Columns = regimes + global
        Cell = ✓ if feature is causal in that regime, blank if not
        Last column = frequency count
        Bottom rows = Jaccard similarity to global set

        This becomes Table 3 in the paper.
        """
        if "error" in stability_results:
            print("[causal_stability] No stability results to display.")
            return

        global_features = set(stability_results.get("global_features", []))
        regime_features = stability_results.get("features_by_regime", {})
        feature_freq = stability_results.get("feature_frequency", {})
        jaccard_global = stability_results.get("jaccard_vs_global", {})
        n_regimes = stability_results.get("n_regimes", 0)

        # Column order: Global, then regimes in chronological order
        regime_names = sorted(regime_features.keys())
        all_cols = ["global"] + regime_names

        # All features sorted by frequency (descending), then alphabetically
        all_features = sorted(
            feature_freq.keys(),
            key=lambda f: (-feature_freq.get(f, 0), f),
        )

        # Column widths
        feat_width = max(22, max((len(f) for f in all_features), default=10) + 2)
        col_width = max(10, max((len(r) for r in all_cols), default=6) + 2)

        # Header
        W = feat_width + (col_width * len(all_cols)) + col_width + 4
        print(f"\n{'='*W}")
        print(f"  TABLE 3 — CAUSAL FEATURE STABILITY ACROSS REGIMES")
        print(f"{'='*W}")

        header = f"{'Feature':<{feat_width}}"
        for col in all_cols:
            label = col[:col_width-1]
            header += f" {label:^{col_width}}"
        header += f" {'Freq':^{col_width}}"
        print(header)
        print("-" * W)

        # Feature rows
        for feat in all_features:
            row = f"{feat:<{feat_width}}"
            for col in all_cols:
                if col == "global":
                    present = feat in global_features
                else:
                    present = feat in set(regime_features.get(col, []))
                cell = "  ✓" if present else ""
                row += f" {cell:^{col_width}}"
            freq = feature_freq.get(feat, 0)
            row += f" {freq:^{col_width}}/{n_regimes}"
            print(row)

        # Jaccard vs global row
        print("-" * W)
        jrow = f"{'Jaccard vs global':<{feat_width}}"
        for col in all_cols:
            if col == "global":
                jrow += f" {'—':^{col_width}}"
            else:
                j = jaccard_global.get(col, 0.0)
                jrow += f" {j:^{col_width}.3f}"
        jrow += f" {'':^{col_width}}"
        print(jrow)
        print(f"{'='*W}")

        # Summary
        stable = stability_results.get("stable_features", [])
        specific = stability_results.get("regime_specific_features", [])
        print(f"\n  Stable features (≥75% of regimes): {stable if stable else 'None'}")
        print(f"  Regime-specific features (1 regime only): {specific if specific else 'None'}")
        print()

    # -----------------------------------------------------------------------
    # Public — save report
    # -----------------------------------------------------------------------

    def save_stability_report(
        self,
        stability_results: dict,
        ticker: str,
        output_dir: Path,
    ) -> Path:
        """
        Save full stability results to JSON.

        Args:
            stability_results: Output of analyse_stability().
            ticker:            Stock ticker.
            output_dir:        Directory to save the report.

        Returns:
            Path to saved JSON file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"causal_stability_{ticker.upper()}.json"
        csv_path = output_dir / "table3_causal_stability.csv"

        # 1. Save raw JSON
        with open(json_path, "w") as f:
            json.dump(stability_results, f, indent=2, default=str)

        # 2. Save Table 3 CSV
        if "error" not in stability_results:
            global_features = set(stability_results.get("global_features", []))
            regime_features = stability_results.get("features_by_regime", {})
            feature_freq = stability_results.get("feature_frequency", {})
            jaccard_global = stability_results.get("jaccard_vs_global", {})
            n_regimes = stability_results.get("n_regimes", 0)

            regime_names = sorted(regime_features.keys())
            all_cols = ["global"] + regime_names
            all_features = sorted(
                feature_freq.keys(),
                key=lambda f: (-feature_freq.get(f, 0), f),
            )

            rows = []
            for feat in all_features:
                row = {"Feature": feat}
                for col in all_cols:
                    if col == "global":
                        row[col] = "✓" if feat in global_features else ""
                    else:
                        row[col] = "✓" if feat in set(regime_features.get(col, [])) else ""
                row["Frequency"] = f"{feature_freq.get(feat, 0)}/{n_regimes}"
                rows.append(row)

            # Jaccard row at bottom
            jrow = {"Feature": "Jaccard vs global", "global": "—"}
            for col in regime_names:
                jrow[col] = f"{jaccard_global.get(col, 0.0):.3f}"
            jrow["Frequency"] = ""
            rows.append(jrow)

            pd.DataFrame(rows).to_csv(csv_path, index=False)
            logger.info(f"[causal_stability] Saved Table 3 CSV → {csv_path}")

        logger.info(f"[causal_stability] Saved stability JSON → {json_path}")
        return json_path

    # -----------------------------------------------------------------------
    # Private — helpers
    # -----------------------------------------------------------------------

    def _resolve_target(self, df: pd.DataFrame, ticker: str) -> str:
        """Determine the target column for causal discovery."""
        if ticker.upper() in ("NIFTY", "^NSEI", "NIFTY50"):
            if "excess_return_5d" in df.columns and not (df["excess_return_5d"] == 0).all():
                logger.info("[causal_stability] Using excess_return_5d as target")
                return "excess_return_5d"
            logger.info("[causal_stability] Using log_return_5d as target")
            return "log_return_5d"
        return self.cfg["model"]["target"]

    def _run_causal_on_subset(
        self,
        subset_df: pd.DataFrame,
        target: str,
        target_cols: list[str],
        ticker: str,
        label: str,
    ) -> list[str]:
        """
        Run Granger (+ optionally PCMCI) on a data subset and return
        the selected causal features.
        """
        from ml.src.causal.granger import GrangerCausality
        from ml.src.causal.pcmci import PCMCIDiscovery
        from ml.src.causal.selector import CausalSelector

        # Strip target columns from the feature set
        cols_to_drop = [c for c in target_cols if c in subset_df.columns]
        df_causal = subset_df.drop(columns=cols_to_drop, errors="ignore")

        # Re-add only the target for Granger (it needs y in the df)
        if target in subset_df.columns:
            df_granger = df_causal.copy()
            df_granger[target] = subset_df[target]
        else:
            logger.warning(
                f"[causal_stability] Target '{target}' not in {label} data — skipping"
            )
            return []

        # ── Granger always runs — it's the baseline ─────────────────────
        granger_results = None
        try:
            granger = GrangerCausality(self.config_path)
            granger_results = granger.run(df_granger, target=target, verbose=False)
        except Exception as e:
            logger.error(f"[causal_stability] Granger failed for {label}: {e}")

        # ── PCMCI runs independently — failure doesn't affect Granger ───
        pcmci_results = None
        if not self.skip_pcmci:
            try:
                # Use last 50% for PCMCI (consistent with main pipeline)
                n_pcmci = max(200, len(df_causal) // 2)
                df_pcmci = df_causal.iloc[-n_pcmci:]
                if target in subset_df.columns:
                    df_pcmci = df_pcmci.copy()
                    df_pcmci[target] = subset_df.loc[df_pcmci.index, target]

                pcmci = PCMCIDiscovery(self.config_path)
                pcmci_results = pcmci.run(df_pcmci, target=target, exclude_target=True)
            except Exception as e:
                logger.warning(
                    f"[causal_stability] PCMCI failed for {label}: {e} — "
                    f"using Granger only"
                )
        else:
            logger.info(f"[causal_stability] PCMCI skipped for {label} (--skip-pcmci)")

        # ── Select features ─────────────────────────────────────────────
        if granger_results is None:
            return []

        # If PCMCI didn't run, create a minimal empty result for the selector
        if pcmci_results is None:
            pcmci_results = {
                "causal_links": {},
                "p_matrix": np.array([]),
                "val_matrix": np.array([]),
                "var_names": [],
                "target": target,
                "target_idx": None,
            }

        try:
            selector = CausalSelector(self.config_path, cfg=self.cfg)
            # Use union strategy for per-regime (more permissive since less data)
            selector.strategy = "union"
            features = selector.select(
                ticker, granger_results, pcmci_results, save=False
            )
            return features
        except Exception as e:
            logger.warning(f"[causal_stability] Feature selection failed for {label}: {e}")
            # Fallback: return Granger-only features
            try:
                granger_obj = GrangerCausality(self.config_path)
                return granger_obj.get_causal_features(granger_results)
            except Exception:
                return []


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Per-regime causal discovery and stability analysis"
    )
    parser.add_argument("--ticker", type=str, required=True, help="e.g. NIFTY")
    parser.add_argument("--market", type=str, default="india", choices=["us", "india"])
    parser.add_argument(
        "--skip-pcmci", action="store_true",
        help="Use only Granger for per-regime analysis (much faster)"
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()
    cfg = _load_config()

    # Load feature matrix
    if ticker in ("NIFTY", "^NSEI", "NIFTY50") or args.market == "india":
        from ml.src.data.nifty_loader import NiftyLoader
        feat_path = NiftyLoader().out_dir / "NIFTY_features.csv"
    else:
        from ml.src.features.pipeline import FeaturePipeline
        feat_path = FeaturePipeline().features_dir / f"{ticker}_features.csv"

    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)

    # Use only training split — no leakage
    train_end = int(len(df) * cfg["model"]["train_ratio"])
    df_train = df.iloc[:train_end]

    analyser = CausalStabilityAnalyser(
        market=args.market, skip_pcmci=args.skip_pcmci
    )
    regime_features = analyser.run(df_train, ticker, market=args.market)
    stability = analyser.analyse_stability(regime_features)
    analyser.print_stability_table(stability)

    # Save
    root = Path(__file__).resolve().parents[3]
    output_dir = root / "paper_output" / ticker
    analyser.save_stability_report(stability, ticker, output_dir)
    print(f"\nSaved to: {output_dir}")
