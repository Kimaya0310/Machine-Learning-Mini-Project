"""Terminal Demonstration: Instacart Customer Segmentation & Reorder Prediction System.

Runs model inference, prints executive summaries, evaluates segment personas,
and generates next-order repeat purchase predictions for held-out customers.
"""
import os
import sys
import json
from pathlib import Path

# Silence loky core detection warning
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import joblib
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

def main():
    print("=" * 80)
    print("  INSTACART CUSTOMER SEGMENTATION & REORDER PREDICTION SYSTEM")
    print("  Machine Learning Lab Mini-Project | 24CS01TP0501")
    print("  Ramdeobaba University, Nagpur - Academic Year 2026-27")
    print("=" * 80)
    
    meta_path = REPORTS / "meta.json"
    if not meta_path.exists():
        print("Error: reports/meta.json not found.")
        sys.exit(1)
        
    meta = json.loads(meta_path.read_text())
    h = meta["headline"]
    
    print("\n[1] EXECUTIVE TRANSACTIONAL OVERVIEW:")
    print(f"  • Active Customers Analyzed:  {h['users']:,}")
    print(f"  • Total Customer Orders:      {h['orders']:,}")
    print(f"  • Prior Line-Items Purchased: {h['prior_items']:,}")
    print(f"  • Unique Product Catalog:     {h['products']:,}")
    print(f"  • Catalog Reorder Rate:       {h['reorder_rate']:.1%}")
    print(f"  • Average Basket Size:        {h['avg_basket_size']:.2f} items/order")
    print(f"  • Median Inter-Order Gap:     {h['median_days_between_orders']:.1f} days")
    
    print("\n[2] UNSUPERVISED CUSTOMER SEGMENTATION (K-Means & Hierarchical):")
    print(f"  • Optimal Clusters (k):       {meta['k']}")
    print(f"  • K-Means Silhouette Score:   {meta['silhouette_kmeans']:.4f}")
    hier = meta["hierarchical"]
    print(f"  • Hierarchical Silhouette:    {hier['silhouette']:.4f}")
    print(f"  • Adjusted Rand Index (ARI):  {hier['adjusted_rand_vs_kmeans']:.4f} (K-Means vs Ward Hierarchical)")
    
    profiles = pd.read_csv(REPORTS / "segment_profiles.csv")
    print("\n  Segment Personas Identified:")
    for _, row in profiles.iterrows():
        print(f"    - Segment {int(row['segment'])}: {row['segment_name']}")
        print(f"      Size: {int(row['users']):,} shoppers ({row['share_pct']}%) | Orders: {row['n_orders']:.1f} | "
              f"Basket: {row['basket_size']:.1f} | Reorder Ratio: {row['reorder_ratio']:.1%} | "
              f"Organic Share: {row['organic_share']:.1%}")

    print("\n[3] SUPERVISED CLASSIFICATION BENCHMARK (Held-Out Test Set: 521,348 rows):")
    comp = pd.read_csv(REPORTS / "model_comparison.csv")
    display_cols = ["model", "precision", "recall", "f1", "accuracy", "roc_auc", "avg_precision", "threshold", "train_time_s"]
    print(comp[display_cols].to_string(index=False))
    
    print(f"\n  ★ Top-Performing Model: {meta['best_model']} (F1 = {meta['best_f1']:.4f}, ROC-AUC = {meta['best_auc']:.4f})")
    
    print("\n[4] NEXT-ORDER BASKET PREDICTION FOR HELD-OUT SHOPPER:")
    cand = pd.read_csv(REPORTS / "app_candidates.csv.gz")
    users = pd.read_csv(REPORTS / "app_users.csv.gz", index_col=0)
    
    sample_uid = cand["user_id"].unique()[0]
    u = users.loc[sample_uid]
    seg_name = meta["segment_names"][str(int(u["segment"]))]
    
    print(f"  • Selected Customer ID:       {sample_uid}")
    print(f"  • Customer Behavioral Persona:{seg_name} (Segment {int(u['segment'])})")
    print(f"  • Historical Orders Placed:   {int(u['n_orders'])}")
    print(f"  • Average Basket Size:        {u['basket_size']:.1f} items")
    print(f"  • Historical Reorder Ratio:   {u['reorder_ratio']:.1%}")
    
    # Run prediction using Gradient Boosting
    gb_art = joblib.load(MODELS / "gradient_boosting.joblib")
    sub = cand[cand["user_id"] == sample_uid].copy()
    sub["reorder_prob"] = gb_art["model"].predict_proba(sub[gb_art["features"]])[:, 1]
    sub = sub.sort_values("reorder_prob", ascending=False)
    sub["predicted_reorder"] = np.where(sub["reorder_prob"] >= gb_art["threshold"], "YES", "NO")
    sub["actually_reordered"] = np.where(sub["label"] == 1, "YES (Purchased)", "No")
    
    print("\n  Top 10 Products Ranked by Repeat Purchase Likelihood:")
    top10 = sub.head(10)[["product_name", "reorder_prob", "predicted_reorder", "actually_reordered"]]
    for idx, (_, r) in enumerate(top10.iterrows(), 1):
        print(f"    {idx:2d}. {r['product_name'][:45]:<45} | Prob: {r['reorder_prob']:.1%} | Pred: {r['predicted_reorder']:<3} | Actual: {r['actually_reordered']}")
        
    pred_set = set(sub.loc[sub["predicted_reorder"] == "YES", "product_id"])
    true_set = set(sub.loc[sub["label"] == 1, "product_id"])
    hit = len(pred_set & true_set)
    p = hit / len(pred_set) if pred_set else 0
    r = hit / len(true_set) if true_set else 0
    
    print(f"\n  Evaluation for Shopper {sample_uid}:")
    print(f"    - Products Recommended: {len(pred_set)}")
    print(f"    - Precision (Hit Rate): {p:.1%}")
    print(f"    - Recall (Caught Rate): {r:.1%}")
    print("\n" + "=" * 80)
    print("  Demonstration completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
