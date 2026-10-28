"""Exploratory Data Analysis: head/tail/describe/shape, missing values, duplicates,
outliers, plus univariate, bivariate and multivariate plots.  Figures go to reports/figures."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import CLUSTER_FEATURES, FIG_DIR, REPORTS_DIR

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = "viridis"
DOW_LABELS = ["Day 0", "Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6"]


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=130)
    plt.close(fig)


def basic_summary(data: dict) -> dict:
    """shape / missing / duplicates / dtype summary for every raw table."""
    out = {}
    for name in ("orders", "prior", "train", "products", "aisles", "departments"):
        df = data[name]
        out[name] = {
            "shape": list(df.shape),
            "missing": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
            "duplicate_rows": int(df.duplicated().sum()),
        }
    return out


def outlier_report(user_feats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in CLUSTER_FEATURES:
        s = user_feats[c]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append({"feature": c, "lower_fence": lo, "upper_fence": hi,
                     "n_outliers": int(((s < lo) | (s > hi)).sum()),
                     "pct_outliers": round(float(((s < lo) | (s > hi)).mean() * 100), 2)})
    return pd.DataFrame(rows)


def run_eda(data: dict, prior_tbl: pd.DataFrame, user_feats: pd.DataFrame, products_full: pd.DataFrame) -> dict:
    orders = data["orders"]
    po = orders[orders["eval_set"] == "prior"]
    stats = {"tables": basic_summary(data)}

    # --- tabular EDA outputs (head/tail/describe) -------------------------------------------
    orders.head(10).to_csv(REPORTS_DIR / "eda_orders_head.csv", index=False)
    orders.tail(10).to_csv(REPORTS_DIR / "eda_orders_tail.csv", index=False)
    user_feats.describe().T.to_csv(REPORTS_DIR / "eda_user_features_describe.csv")
    outlier_report(user_feats).to_csv(REPORTS_DIR / "eda_outliers.csv", index=False)

    n_orders_total = int(len(orders))
    stats["headline"] = {
        "users": int(orders["user_id"].nunique()),
        "orders": n_orders_total,
        "prior_items": int(len(data["prior"])),
        "products": int(len(data["products"])),
        "reorder_rate": round(float(data["prior"]["reordered"].mean()), 4),
        "avg_basket_size": round(float(user_feats["basket_size"].mean()), 2),
        "median_days_between_orders": round(float(po["days_since_prior_order"].median()), 1),
    }

    # --- univariate ----------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(x=po["order_dow"], ax=axes[0], color=sns.color_palette(PALETTE)[2])
    axes[0].set(title="Orders by day of week", xlabel="order_dow (0-6)", ylabel="orders")
    sns.countplot(x=po["order_hour_of_day"], ax=axes[1], color=sns.color_palette(PALETTE)[3])
    axes[1].set(title="Orders by hour of day", xlabel="hour", ylabel="orders")
    _save(fig, "01_orders_dow_hour")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(po["days_since_prior_order"].dropna(), bins=31, ax=axes[0], color=sns.color_palette(PALETTE)[1])
    axes[0].set(title="Days since prior order", xlabel="days")
    sizes = prior_tbl.groupby("order_id").size()
    sns.histplot(sizes.clip(upper=sizes.quantile(0.99)), bins=40, ax=axes[1], color=sns.color_palette(PALETTE)[3])
    axes[1].set(title="Basket size (items per order)", xlabel="items")
    sns.histplot(user_feats["n_orders"], bins=40, ax=axes[2], color=sns.color_palette(PALETTE)[4])
    axes[2].set(title="Prior orders per user", xlabel="orders")
    _save(fig, "02_distributions")

    top = (prior_tbl.merge(products_full[["product_id", "product_name"]], on="product_id")
           .groupby("product_name").size().nlargest(15).sort_values())
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top.index, top.values, color=sns.color_palette(PALETTE, 15))
    ax.set(title="Top 15 products by purchases", xlabel="purchases")
    _save(fig, "03_top_products")

    dep = prior_tbl.groupby("department").agg(items=("order_id", "size"), reorder=("reordered", "mean"))
    dep = dep.sort_values("items")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(dep.index, dep["items"], color=sns.color_palette(PALETTE, len(dep)))
    axes[0].set(title="Purchases by department", xlabel="items")
    dep2 = dep.sort_values("reorder")
    axes[1].barh(dep2.index, dep2["reorder"], color=sns.color_palette("mako", len(dep2)))
    axes[1].set(title="Reorder rate by department", xlabel="share of items that are reorders")
    _save(fig, "04_departments")

    # --- bivariate -----------------------------------------------------------------------------
    cart = prior_tbl.groupby("add_to_cart_order")["reordered"].mean()
    cart = cart[cart.index <= 30]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(cart.index, cart.values, marker="o", color=sns.color_palette(PALETTE)[2])
    axes[0].set(title="Reorder rate vs. position in cart", xlabel="add_to_cart_order", ylabel="reorder rate")
    gap = po.dropna(subset=["days_since_prior_order"]).groupby("days_since_prior_order").size().index
    sz = prior_tbl.groupby("order_id").size().rename("size").reset_index().merge(
        po[["order_id", "days_since_prior_order"]], on="order_id").dropna()
    sz = sz.groupby("days_since_prior_order")["size"].mean()
    axes[1].plot(sz.index, sz.values, marker="o", color=sns.color_palette(PALETTE)[4])
    axes[1].set(title="Average basket size vs. days since prior order", xlabel="days since prior order", ylabel="items")
    _save(fig, "05_bivariate")

    # --- multivariate --------------------------------------------------------------------------
    corr = user_feats[CLUSTER_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation of user-level features")
    _save(fig, "06_correlation")

    sample = user_feats[["n_orders", "basket_size", "reorder_ratio", "avg_days_between"]].sample(
        min(2500, len(user_feats)), random_state=42)
    pg = sns.pairplot(sample, corner=True, plot_kws={"s": 8, "alpha": 0.5}, diag_kind="kde")
    pg.figure.suptitle("Pairwise relationships between key user features", y=1.02)
    pg.figure.savefig(FIG_DIR / "07_pairplot.png", dpi=110, bbox_inches="tight")
    plt.close(pg.figure)

    melt = user_feats[["n_orders", "basket_size", "reorder_ratio", "avg_days_between", "distinct_products"]]
    z = (melt - melt.mean()) / melt.std()
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=z.clip(-4, 6), orient="h", ax=ax, palette=PALETTE, fliersize=2)
    ax.set(title="Outliers: standardised user features (box = IQR)", xlabel="z-score (clipped)")
    _save(fig, "08_outlier_boxplots")

    with open(REPORTS_DIR / "eda_summary.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats
