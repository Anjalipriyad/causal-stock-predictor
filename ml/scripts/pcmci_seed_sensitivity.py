import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Setup paths
root = Path(__file__).resolve().parents[2]
sys.path.append(str(root))

from ml.src.data.loader import _load_config
from ml.src.causal.pcmci import PCMCIDiscovery
from ml.src.causal.leakage_guard import make_causal_discovery_frame

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def jaccard_similarity(list1, list2):
    set1, set2 = set(list1), set(list2)
    if not set1 and not set2:
        return 1.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def main():
    ticker = "NIFTY"
    cfg = _load_config()
    
    # 1. Load the NIFTY feature matrix
    feat_path = root / "ml" / "data" / "processed" / "features" / f"{ticker}_features.csv"
    if not feat_path.exists():
        logger.error(f"Feature matrix not found: {feat_path}")
        return
        
    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    
    # 2. Take only the training split (first 70% of rows by date)
    n = len(df)
    train_end = int(n * 0.70)
    df_train = df.iloc[:train_end].copy()
    
    # 3. Determine target column
    target_col = "log_return_5d"
    if "excess_return_5d" in df_train.columns:
        if df_train["excess_return_5d"].abs().sum() > 0:
            target_col = "excess_return_5d"
            
    logger.info(f"Using target column: {target_col}")
    
    # 4. Strip auxiliary forward-return labels, keeping only the active target.
    df_train = make_causal_discovery_frame(df_train, target_col)
    
    # 5. Run PCMCIDiscovery 10 times with seeds 0 through 9
    results_by_seed = {}
    all_features = set()
    feature_counts = {}
    
    for seed in range(10):
        logger.info(f"Running PCMCI with seed {seed}...")
        np.random.seed(seed)
        
        # Override config seed so pcmci.py uses it
        cfg["project"]["random_seed"] = seed
        pcmci = PCMCIDiscovery(cfg=cfg)
        
        try:
            res = pcmci.run(df_train, target=target_col, exclude_target=False)
            selected = pcmci.get_causal_features(res)
        except Exception as e:
            logger.error(f"PCMCI run failed: {e}")
            selected = []
            
        results_by_seed[seed] = selected
        for f in selected:
            all_features.add(f)
            feature_counts[f] = feature_counts.get(f, 0) + 1
            
    # 7. Compute pairwise Jaccard similarities
    jaccards = []
    for i in range(10):
        for j in range(i + 1, 10):
            jaccards.append(jaccard_similarity(results_by_seed[i], results_by_seed[j]))
            
    mean_jaccard = np.mean(jaccards) if jaccards else 0.0
    std_jaccard = np.std(jaccards) if jaccards else 0.0
    min_jaccard = np.min(jaccards) if jaccards else 0.0
    max_jaccard = np.max(jaccards) if jaccards else 0.0
    
    # 8. Report
    core_features = [f for f, c in feature_counts.items() if c == 10]
    mostly_stable = [f for f, c in feature_counts.items() if 7 <= c <= 9]
    unstable = [f for f, c in feature_counts.items() if 1 <= c <= 3]
    pure_artifacts = [f for f, c in feature_counts.items() if c == 1]
    
    report = {
        "jaccard": {
            "mean": mean_jaccard,
            "std": std_jaccard,
            "min": min_jaccard,
            "max": max_jaccard
        },
        "feature_stability": {
            "core_all_10": core_features,
            "mostly_stable_7_to_9": mostly_stable,
            "unstable_1_to_3": unstable,
            "pure_artifacts_1": pure_artifacts
        },
        "runs": results_by_seed
    }
    
    # 9. Save to json
    out_dir = root / "paper_output" / "NIFTY"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pcmci_seed_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
        
    # 10. Print summary
    print("\n" + "="*60)
    print("PCMCI SEED SENSITIVITY SUMMARY")
    print("="*60)
    print(f"Mean Jaccard: {mean_jaccard:.4f}")
    print(f"Std Jaccard:  {std_jaccard:.4f}")
    print(f"Min Jaccard:  {min_jaccard:.4f}")
    print(f"Max Jaccard:  {max_jaccard:.4f}")
    print("-" * 60)
    print(f"Core Features (10/10 runs): {len(core_features)}")
    for f in core_features: print(f"  - {f}")
    print(f"\nMostly Stable (7-9 runs): {len(mostly_stable)}")
    for f in mostly_stable: print(f"  - {f}")
    print(f"\nUnstable (1-3 runs): {len(unstable)}")
    for f in unstable: print(f"  - {f}")
    print(f"\nPure Artifacts (1 run): {len(pure_artifacts)}")
    for f in pure_artifacts: print(f"  - {f}")
    print("="*60)

if __name__ == "__main__":
    main()
