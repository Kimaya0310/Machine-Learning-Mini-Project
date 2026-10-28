"""Load the Instacart Market Basket Analysis files (memory-safe).

The full `order_products__prior` table has ~32M rows, which can crash a free
Colab session.  We therefore:
  1. read the (small) orders table,
  2. sample N users that have a `train` order,
  3. stream the big item tables in chunks, keeping only rows of sampled orders,
  4. down-cast dtypes.

Files may be plain `.csv` or `.csv.zip` (exactly as downloaded from Kaggle).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ITEM_DTYPES = {
    "order_id": "int32",
    "product_id": "int32",
    "add_to_cart_order": "int16",
    "reordered": "int8",
}
ORDER_DTYPES = {
    "order_id": "int32",
    "user_id": "int32",
    "order_number": "int16",
    "order_dow": "int8",
    "order_hour_of_day": "int8",
}


def _find(data_dir: Path, name: str) -> Path:
    for suffix in (".csv", ".csv.zip", ".csv.gz"):
        p = Path(data_dir) / f"{name}{suffix}"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {name}.csv (or .csv.zip) in {data_dir}.\n"
        "Download the Instacart data from Kaggle and unzip it there "
        "(see README.md), or run `python make_synthetic_data.py` for a demo."
    )


def _read_items(path: Path, order_ids: set, chunksize: int = 2_000_000) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, dtype=ITEM_DTYPES, chunksize=chunksize):
        parts.append(chunk[chunk["order_id"].isin(order_ids)])
    return pd.concat(parts, ignore_index=True)


def load_data(data_dir: str | Path, n_users: int | None = 20000, seed: int = 42) -> dict:
    """Return a dict of DataFrames restricted to a random sample of users."""
    data_dir = Path(data_dir)
    orders = pd.read_csv(_find(data_dir, "orders"), dtype=ORDER_DTYPES)
    orders["days_since_prior_order"] = orders["days_since_prior_order"].astype("float32")

    # Only users who have a labelled `train` order can be used for supervised learning
    train_users = orders.loc[orders["eval_set"] == "train", "user_id"].unique()
    rng = np.random.default_rng(seed)
    if n_users and n_users < len(train_users):
        sampled = rng.choice(train_users, size=n_users, replace=False)
    else:
        sampled = train_users
    orders = orders[orders["user_id"].isin(sampled) & orders["eval_set"].isin(["prior", "train"])]

    prior_ids = set(orders.loc[orders["eval_set"] == "prior", "order_id"])
    train_ids = set(orders.loc[orders["eval_set"] == "train", "order_id"])

    prior = _read_items(_find(data_dir, "order_products__prior"), prior_ids)
    train = _read_items(_find(data_dir, "order_products__train"), train_ids)

    products = pd.read_csv(_find(data_dir, "products"))
    aisles = pd.read_csv(_find(data_dir, "aisles"))
    departments = pd.read_csv(_find(data_dir, "departments"))

    return {
        "orders": orders.reset_index(drop=True),
        "prior": prior,
        "train": train,
        "products": products,
        "aisles": aisles,
        "departments": departments,
    }
