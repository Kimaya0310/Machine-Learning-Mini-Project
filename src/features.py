"""Feature engineering - leakage-free by construction.

Rules that keep the project leakage-free:
  * Every feature is computed ONLY from `prior` orders.
  * The label comes ONLY from the user's `train` order (their latest order).
  * Users are split into train/test BEFORE any learned statistic is computed
    (product-level statistics, scaler, k-means all use training users only),
    and the split is by USER, so the same customer never sits in both sets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CLUSTER_FEATURES


def enrich_products(data: dict) -> pd.DataFrame:
    p = data["products"].merge(data["aisles"], on="aisle_id", how="left").merge(
        data["departments"], on="department_id", how="left")
    p["is_organic"] = p["product_name"].str.contains("organic", case=False, na=False).astype("int8")
    return p


def build_prior_table(data: dict, products_full: pd.DataFrame) -> pd.DataFrame:
    """One row per (prior order, product) with user + product context attached."""
    o = data["orders"][["order_id", "user_id", "order_number", "order_dow",
                        "order_hour_of_day", "days_since_prior_order"]]
    t = data["prior"].merge(o, on="order_id", how="left").merge(
        products_full[["product_id", "aisle_id", "department_id", "department", "is_organic"]],
        on="product_id", how="left")
    return t


def split_users(user_ids, test_size: float = 0.2, seed: int = 42):
    train_u, test_u = train_test_split(np.asarray(user_ids), test_size=test_size, random_state=seed)
    return train_u, test_u


def build_user_features(data: dict, prior_tbl: pd.DataFrame) -> pd.DataFrame:
    """One row per user, summarising their prior behaviour (used for clustering + modelling)."""
    po = data["orders"].query("eval_set == 'prior'").copy()
    po["weekend"] = po["order_dow"].isin([0, 1]).astype("int8")  # dow 0/1 assumed to be the weekend
    g = po.groupby("user_id")
    uf = pd.DataFrame({
        "n_orders": g["order_number"].max(),
        "avg_days_between": g["days_since_prior_order"].mean(),
        "std_days_between": g["days_since_prior_order"].std(),
        "avg_order_hour": g["order_hour_of_day"].mean(),
        "weekend_share": g["weekend"].mean(),
    })
    basket = prior_tbl.groupby(["user_id", "order_id"]).size().groupby("user_id").mean()
    ug = prior_tbl.groupby("user_id")
    uf["basket_size"] = basket
    uf["reorder_ratio"] = ug["reordered"].mean()
    uf["distinct_products"] = ug["product_id"].nunique()
    uf["organic_share"] = ug["is_organic"].mean()

    dept = pd.crosstab(prior_tbl["user_id"], prior_tbl["department"], normalize="index")
    dept.columns = [f"dept_{c.replace(' ', '_')}" for c in dept.columns]
    uf = uf.join(dept)
    for col in ("dept_produce", "dept_snacks"):
        if col not in uf:
            uf[col] = 0.0
    uf["produce_share"] = uf["dept_produce"]
    uf["snacks_share"] = uf["dept_snacks"]

    uf["std_days_between"] = uf["std_days_between"].fillna(0)
    uf["avg_days_between"] = uf["avg_days_between"].fillna(uf["avg_days_between"].median())
    return uf.astype("float32")


def build_product_features(prior_tbl: pd.DataFrame, fit_users) -> pd.DataFrame:
    """Product popularity / reorder statistics from TRAINING users' prior orders only."""
    t = prior_tbl[prior_tbl["user_id"].isin(fit_users)]
    pf = t.groupby("product_id").agg(
        p_orders=("order_id", "size"),
        p_users=("user_id", "nunique"),
        p_reorder_rate=("reordered", "mean"),
        p_avg_cart_pos=("add_to_cart_order", "mean"),
    )
    # aisle / department level reorder rates (also from training users only)
    a_rate = t.groupby("aisle_id")["reordered"].mean().rename("a_reorder_rate")
    d_rate = t.groupby("department_id")["reordered"].mean().rename("d_reorder_rate")
    pmap = t[["product_id", "aisle_id", "department_id"]].drop_duplicates("product_id").set_index("product_id")
    pf = pf.join(pmap).join(a_rate, on="aisle_id").join(d_rate, on="department_id")
    # popularity: share of training users who ever bought the product
    pf["p_user_share"] = pf["p_users"] / max(t["user_id"].nunique(), 1)
    return pf.drop(columns=["aisle_id", "department_id"]).astype("float32")


def build_candidates(data: dict, prior_tbl: pd.DataFrame, user_feats: pd.DataFrame,
                     prod_feats: pd.DataFrame, products_full: pd.DataFrame,
                     segments: pd.Series | None = None) -> pd.DataFrame:
    """Every (user, previously-bought product) pair, with features and the 0/1 label
    'does this product appear in the user's next (train) order?'."""
    up = prior_tbl.groupby(["user_id", "product_id"]).agg(
        up_orders=("order_id", "size"),
        up_first=("order_number", "min"),
        up_last=("order_number", "max"),
        up_avg_cart_pos=("add_to_cart_order", "mean"),
    ).reset_index()

    up = up.merge(user_feats[["n_orders", "avg_days_between", "std_days_between", "basket_size",
                              "reorder_ratio", "distinct_products", "organic_share",
                              "avg_order_hour", "weekend_share"]],
                  left_on="user_id", right_index=True, how="left")
    up["up_order_rate"] = up["up_orders"] / up["n_orders"]
    up["up_orders_since_last"] = up["n_orders"] - up["up_last"]
    up["up_rate_since_first"] = up["up_orders"] / (up["n_orders"] - up["up_first"] + 1)

    # --- recency in real DAYS: cumulative days at every prior order of every user
    po = data["orders"].query("eval_set == 'prior'")[["user_id", "order_number", "days_since_prior_order"]]
    po = po.sort_values(["user_id", "order_number"])
    po["gap0"] = po["days_since_prior_order"].fillna(0)
    po["cum_days"] = po.groupby("user_id")["gap0"].cumsum()
    total_days = po.groupby("user_id")["cum_days"].max().rename("total_days")
    po = po.rename(columns={"order_number": "up_last", "cum_days": "last_cum_days"})[["user_id", "up_last", "last_cum_days"]]
    up = up.merge(po, on=["user_id", "up_last"], how="left").merge(total_days, left_on="user_id", right_index=True, how="left")
    up["up_days_since_last_prior"] = up["total_days"] - up["last_cum_days"]

    # --- purchase frequency inside the product's own life span
    span = (up["up_last"] - up["up_first"]).clip(lower=0)
    up["up_avg_gap_orders"] = np.where(up["up_orders"] > 1, span / (up["up_orders"] - 1).clip(lower=1), up["n_orders"])
    up["up_overdue_ratio"] = up["up_orders_since_last"] / up["up_avg_gap_orders"].clip(lower=1)
    up["up_bought_last_order"] = (up["up_orders_since_last"] == 0).astype("int8")
    up["up_first_order_share"] = up["up_first"] / up["n_orders"]

    # --- recent behaviour: purchases in the user's last 3 / 5 prior orders
    n_o = prior_tbl["user_id"].map(user_feats["n_orders"])
    for window in (3, 5):
        recent = prior_tbl.loc[prior_tbl["order_number"] > (n_o - window), ["user_id", "product_id"]]
        cnt = recent.groupby(["user_id", "product_id"]).size().rename(f"up_last{window}")
        up = up.join(cnt, on=["user_id", "product_id"])
        up[f"up_last{window}"] = up[f"up_last{window}"].fillna(0)
    del n_o
    up = up.drop(columns=["up_first", "up_last", "last_cum_days", "total_days"])

    up = up.merge(prod_feats, left_on="product_id", right_index=True, how="left")
    for c in prod_feats.columns:
        up[c] = up[c].fillna(0)
    up = up.merge(products_full[["product_id", "is_organic"]], on="product_id", how="left")

    # context of the order being predicted (known when the customer starts shopping)
    tr = data["orders"].query("eval_set == 'train'")[["user_id", "order_id", "days_since_prior_order",
                                                       "order_dow", "order_hour_of_day"]]
    tr = tr.rename(columns={"days_since_prior_order": "target_days_since_prior",
                            "order_dow": "target_dow", "order_hour_of_day": "target_hour"})
    up = up.merge(tr, on="user_id", how="inner")
    up["target_days_since_prior"] = up["target_days_since_prior"].fillna(up["avg_days_between"])
    # real days since the product was last bought, as of the moment of the next order
    up["up_days_since_last"] = up["up_days_since_last_prior"] + up["target_days_since_prior"]
    up["gap_vs_usual"] = up["target_days_since_prior"] / up["avg_days_between"].clip(lower=1)
    up = up.drop(columns=["up_days_since_last_prior"])

    if segments is not None:
        up["segment"] = up["user_id"].map(segments).astype("int8")

    # label from the user's train order only
    pos = data["train"].merge(tr[["user_id", "order_id"]], on="order_id")[["user_id", "product_id"]]
    pos["label"] = 1
    up = up.merge(pos, on=["user_id", "product_id"], how="left")
    up["label"] = up["label"].fillna(0).astype("int8")
    return up.drop(columns=["order_id"])


NON_FEATURE_COLS = ["user_id", "product_id", "label"]


def feature_columns(cand: pd.DataFrame, one_hot_segment: bool = True) -> list[str]:
    cols = [c for c in cand.columns if c not in NON_FEATURE_COLS + ["segment"]]
    if not one_hot_segment:
        cols = [c for c in cols if not c.startswith("seg_")]
    return cols


def one_hot_segments(cand: pd.DataFrame, n_segments: int) -> pd.DataFrame:
    cand = cand.copy()
    for k in range(n_segments):
        cand[f"seg_{k}"] = (cand["segment"] == k).astype("int8")
    return cand