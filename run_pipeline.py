"""End-to-end pipeline: load -> EDA -> segmentation -> reorder models -> save artifacts.

Examples
--------
  python run_pipeline.py --data data --users 20000          # real Kaggle data
  python run_pipeline.py --data data_synthetic --users 4000 # demo data
  python run_pipeline.py --data data --users 5000 --fast    # quick run
"""
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))  # silence Windows joblib warning

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import eda, features as F, models as M, segmentation as S
from src.config import CLUSTER_FEATURES, DATA_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR


def main(data_dir: Path, n_users: int, fast: bool):
    t0 = time.time()
    from src.data_loader import load_data

    print(f"[1/6] Loading data from {data_dir} (sampling {n_users} users) ...")
    data = load_data(data_dir, n_users=n_users, seed=RANDOM_STATE)
    synthetic = (Path(data_dir) / "SYNTHETIC_DATA.txt").exists()
    products_full = F.enrich_products(data)
    prior_tbl = F.build_prior_table(data, products_full)
    print(f"      users={data['orders'].user_id.nunique():,}  prior items={len(prior_tbl):,}")

    print("[2/6] User-level features + EDA ...")
    user_feats = F.build_user_features(data, prior_tbl)
    stats = eda.run_eda(data, prior_tbl, user_feats, products_full)

    print("[3/6] Splitting USERS 80/20 (before any learned statistic) ...")
    train_users, test_users = F.split_users(user_feats.index.values, 0.2, RANDOM_STATE)

    print("[4/6] Customer segmentation (k-means + hierarchical) ...")
    Xs_train, scaler = S.transform(user_feats.loc[train_users])
    sweep = S.sweep_kmeans(Xs_train, range(2, 11))
    sweep.to_csv(REPORTS_DIR / "kmeans_sweep.csv", index=False)
    k = S.choose_k(sweep)
    km = S.fit_kmeans(Xs_train, k)
    hier = S.hierarchical(Xs_train, k, km.labels_)
    Xs_all, _ = S.transform(user_feats, scaler)
    seg_all = pd.Series(km.predict(Xs_all), index=user_feats.index, name="segment")
    prof = S.profile_clusters(user_feats, seg_all.values)
    names = S.name_clusters(prof, user_feats)
    S.plot_segments(Xs_all, seg_all.values, names, prof, user_feats)
    prof_out = prof.copy(); prof_out.insert(0, "segment_name", [names[i] for i in prof_out.index])
    prof_out.to_csv(REPORTS_DIR / "segment_profiles.csv")
    sil_final = float(sweep.loc[sweep["k"] == k, "silhouette"].iloc[0])
    print(f"      chosen k={k}  silhouette={sil_final:.3f}  hierarchical ARI vs k-means={hier['adjusted_rand_vs_kmeans']:.2f}")
    for s, n in names.items():
        print(f"      segment {s}: {n}  ({int(prof.loc[s, 'users']):,} users)")

    print("[5/6] Building leakage-free candidate table ...")
    prod_feats = F.build_product_features(prior_tbl, train_users)   # train users only
    cand = F.build_candidates(data, prior_tbl, user_feats, prod_feats, products_full, seg_all)
    cand = F.one_hot_segments(cand, k)
    features = F.feature_columns(cand, one_hot_segment=True)
    cand_train = cand[cand["user_id"].isin(train_users)].reset_index(drop=True)
    cand_test = cand[cand["user_id"].isin(test_users)].reset_index(drop=True)
    assert set(cand_train.user_id).isdisjoint(cand_test.user_id), "user leakage!"
    print(f"      train rows={len(cand_train):,} (positive {cand_train.label.mean():.1%})   "
          f"test rows={len(cand_test):,} (positive {cand_test.label.mean():.1%})   features={len(features)}")

    print("[6/6] Training + tuning models ...")
    kw = dict(n_iter=3, cv=3, tune_rows=25_000, eval_rows=60_000) if fast else {}
    res, fitted, extras = M.train_and_compare(cand_train, cand_test, features, **kw)
    M.plot_comparison(res, extras)
    imp = M.feature_importance_plot(fitted, features)
    sweep_rf = M.rf_parameter_impact(extras, features, n_rows=30_000 if fast else 60_000)

    # ----------------------------------------------------------------- save artifacts
    res.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    imp.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    sweep_rf.to_csv(REPORTS_DIR / "rf_parameter_impact.csv", index=False)
    (REPORTS_DIR / "confusion_matrices.json").write_text(json.dumps(extras["confusion"], indent=2))

    for name, (est, thr) in fitted.items():
        joblib.dump({"model": est, "threshold": thr, "features": features},
                    MODELS_DIR / f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')}.joblib")
    joblib.dump({"scaler": scaler, "kmeans": km, "features": CLUSTER_FEATURES, "names": names},
                MODELS_DIR / "segmentation.joblib")

    # slice of held-out customers for the app's "Predict" tab
    pname = products_full.set_index("product_id")["product_name"]
    app_users = np.random.default_rng(RANDOM_STATE).choice(test_users, size=min(300, len(test_users)), replace=False)
    app_cand = cand_test[cand_test["user_id"].isin(app_users)].copy()
    app_cand["product_name"] = app_cand["product_id"].map(pname)
    app_cand.to_csv(REPORTS_DIR / "app_candidates.csv.gz", index=False)
    user_feats.loc[app_users].assign(segment=seg_all.loc[app_users].values).to_csv(
        REPORTS_DIR / "app_users.csv.gz")

    best = res.iloc[0]
    meta = {
        "synthetic_data": synthetic, "n_users": int(len(user_feats)), "k": int(k),
        "silhouette_kmeans": sil_final, "hierarchical": hier,
        "segment_names": {int(a): b for a, b in names.items()},
        "best_model": best["model"], "best_f1": float(best["f1"]), "best_auc": float(best["roc_auc"]),
        "train_rows": int(len(cand_train)), "test_rows": int(len(cand_test)),
        "positive_rate": float(cand_train.label.mean()), "features": features,
        "runtime_s": round(time.time() - t0, 1),
        "headline": stats["headline"],
    }
    (REPORTS_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone in {time.time() - t0:.0f}s.  Best model: {best['model']} (F1={best['f1']:.3f}, AUC={best['roc_auc']:.3f})")
    print("Next: streamlit run app.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--users", type=int, default=20000, help="users to sample (0 = all)")
    ap.add_argument("--fast", action="store_true", help="smaller tuning budget for quick runs")
    a = ap.parse_args()
    main(a.data, a.users or None, a.fast)