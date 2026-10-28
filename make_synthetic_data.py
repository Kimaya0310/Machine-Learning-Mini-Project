"""Generate a SYNTHETIC dataset with the exact schema of the Instacart files.

Purpose: lets you run and demo the whole pipeline / Streamlit app instantly,
without a Kaggle account.  The numbers produced from this data are NOT real
Instacart results - for your report, download the real data (see README.md).

Usage:  python make_synthetic_data.py --users 4000
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEPARTMENTS = ["produce", "dairy eggs", "snacks", "beverages", "frozen", "pantry",
               "bakery", "canned goods", "deli", "dry goods pasta", "meat seafood",
               "breakfast", "household", "personal care"]

BASE_ITEMS = {
    "produce": ["Banana", "Strawberries", "Baby Spinach", "Avocado", "Lemon", "Cucumber", "Tomatoes", "Carrots"],
    "dairy eggs": ["Whole Milk", "Greek Yogurt", "Cheddar Cheese", "Large Eggs", "Butter", "Almond Milk"],
    "snacks": ["Potato Chips", "Dark Chocolate", "Granola Bars", "Popcorn", "Pretzels", "Trail Mix"],
    "beverages": ["Sparkling Water", "Orange Juice", "Cold Brew Coffee", "Green Tea", "Cola", "Coconut Water"],
    "frozen": ["Frozen Pizza", "Ice Cream", "Frozen Berries", "Frozen Peas", "Veggie Burgers"],
    "pantry": ["Olive Oil", "Peanut Butter", "Honey", "Tomato Sauce", "Brown Rice", "Black Beans"],
    "bakery": ["Whole Wheat Bread", "Bagels", "Tortillas", "Croissants"],
    "canned goods": ["Chickpeas", "Coconut Milk", "Diced Tomatoes", "Tuna"],
    "deli": ["Hummus", "Turkey Slices", "Guacamole", "Fresh Salsa"],
    "dry goods pasta": ["Spaghetti", "Penne", "Quinoa", "Oats"],
    "meat seafood": ["Chicken Breast", "Ground Beef", "Salmon Fillet", "Bacon"],
    "breakfast": ["Cereal", "Pancake Mix", "Maple Syrup", "Instant Oatmeal"],
    "household": ["Paper Towels", "Dish Soap", "Laundry Detergent", "Trash Bags"],
    "personal care": ["Shampoo", "Toothpaste", "Hand Soap", "Deodorant"],
}

# segment: (mean #orders, mean basket, mean days between, loyalty, organic bias, weekend bias, dept preference)
SEGMENTS = {
    "bulk_weekly":   dict(w=0.22, orders=(28, 8), basket=(22, 5), gap=(7, 1.5),  loyal=0.55, organic=0.15, dept=["pantry", "household", "meat seafood", "frozen"]),
    "top_up":        dict(w=0.28, orders=(35, 12), basket=(6, 2), gap=(4, 1.5),  loyal=0.70, organic=0.20, dept=["produce", "dairy eggs", "bakery"]),
    "occasional":    dict(w=0.25, orders=(9, 4), basket=(10, 4), gap=(18, 6),    loyal=0.25, organic=0.10, dept=["snacks", "beverages", "frozen"]),
    "organic_fresh": dict(w=0.25, orders=(24, 9), basket=(14, 4), gap=(9, 3),    loyal=0.60, organic=0.75, dept=["produce", "dairy eggs", "deli", "canned goods"]),
}


def build_catalog(rng):
    departments = pd.DataFrame({"department_id": range(1, len(DEPARTMENTS) + 1), "department": DEPARTMENTS})
    aisles, products = [], []
    aisle_id, pid = 0, 0
    for d_id, dept in enumerate(DEPARTMENTS, start=1):
        for base in BASE_ITEMS[dept]:
            aisle_id += 1
            aisles.append({"aisle_id": aisle_id, "aisle": f"{base.lower()} aisle"})
            for variant in ("", "Organic ", "Large ", "Family Pack "):
                pid += 1
                name = f"{variant}{base}".strip()
                if variant == "Organic " and dept in ("household", "personal care"):
                    name = base + " Eco"
                products.append({"product_id": pid, "product_name": name, "aisle_id": aisle_id, "department_id": d_id})
    return pd.DataFrame(products), pd.DataFrame(aisles), departments


def main(n_users: int, out: Path, seed: int):
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    products, aisles, departments = build_catalog(rng)
    dept_of = dict(zip(products.product_id, products.department_id))
    dept_name = dict(zip(departments.department_id, departments.department))
    organic = products.product_name.str.contains("Organic").to_numpy()
    pid_arr = products.product_id.to_numpy()
    pdept = products.department_id.map(dept_name).to_numpy()
    popularity = rng.pareto(1.2, len(products)) + 1.0
    popularity /= popularity.sum()

    seg_names = list(SEGMENTS)
    seg_w = np.array([SEGMENTS[s]["w"] for s in seg_names])
    orders_rows, prior_rows, train_rows = [], [], []
    order_id = 0

    for user_id in range(1, n_users + 1):
        s = SEGMENTS[seg_names[rng.choice(len(seg_names), p=seg_w / seg_w.sum())]]
        n_orders = int(np.clip(rng.normal(*s["orders"]), 4, 80))   # incl. the final `train` order
        fav_dow, fav_hour = rng.integers(0, 7), int(np.clip(rng.normal(14, 3.5), 6, 22))

        # user-specific product preference weights
        w = popularity.copy()
        pref_mask = np.isin(pdept, s["dept"])
        w[pref_mask] *= 6.0
        w[organic] *= (0.2 + 4.0 * s["organic"])
        w /= w.sum()
        pool = rng.choice(pid_arr, size=min(len(pid_arr), int(rng.integers(25, 70))), replace=False, p=w)
        pool_w = rng.dirichlet(np.ones(len(pool)) * 0.5)
        favourites = set(rng.choice(pool, size=max(4, len(pool) // 6), replace=False, p=pool_w))
        seen = set()

        for n in range(1, n_orders + 1):
            order_id += 1
            gap = np.nan if n == 1 else float(np.clip(round(rng.normal(*s["gap"])), 0, 30))
            dow = int(np.clip(fav_dow + rng.integers(-1, 2), 0, 6)) % 7
            hour = int(np.clip(fav_hour + rng.integers(-2, 3), 0, 23))
            eval_set = "train" if n == n_orders else "prior"
            orders_rows.append((order_id, user_id, eval_set, n, dow, hour, gap))

            size = max(1, int(rng.normal(*s["basket"])))
            chosen = []
            for _ in range(size):
                if seen and rng.random() < s["loyal"]:
                    # re-buy: favourites are far more likely than other seen items
                    cand = [p for p in seen]
                    cw = np.array([5.0 if p in favourites else 1.0 for p in cand])
                    p = rng.choice(cand, p=cw / cw.sum())
                else:
                    p = rng.choice(pid_arr, p=w)
                if p not in chosen:
                    chosen.append(int(p))
            rng.shuffle(chosen)
            rows = [(order_id, p, k + 1, int(p in seen)) for k, p in enumerate(chosen)]
            (train_rows if eval_set == "train" else prior_rows).extend(rows)
            seen.update(chosen)

    cols = ["order_id", "product_id", "add_to_cart_order", "reordered"]
    pd.DataFrame(orders_rows, columns=["order_id", "user_id", "eval_set", "order_number", "order_dow",
                                       "order_hour_of_day", "days_since_prior_order"]).to_csv(out / "orders.csv", index=False)
    pd.DataFrame(prior_rows, columns=cols).to_csv(out / "order_products__prior.csv", index=False)
    pd.DataFrame(train_rows, columns=cols).to_csv(out / "order_products__train.csv", index=False)
    products.to_csv(out / "products.csv", index=False)
    aisles.to_csv(out / "aisles.csv", index=False)
    departments.to_csv(out / "departments.csv", index=False)
    (out / "SYNTHETIC_DATA.txt").write_text(
        "This folder holds SYNTHETIC data in the Instacart schema. Replace it with the real Kaggle files for final results.\n")
    print(f"Wrote synthetic Instacart-style data for {n_users} users ({order_id} orders) to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=4000)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "data_synthetic")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.users, a.out, a.seed)
