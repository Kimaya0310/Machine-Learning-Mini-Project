"""Central configuration: paths and constants shared by every module."""
import os
from pathlib import Path

# Silence joblib's harmless "could not find physical cores" warning on Windows (no `wmic`)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42

# Columns used for customer segmentation (Part 2)
CLUSTER_FEATURES = [
    "n_orders",
    "avg_days_between",
    "basket_size",
    "reorder_ratio",
    "distinct_products",
    "organic_share",
    "produce_share",
    "snacks_share",
    "avg_order_hour",
    "weekend_share",
]
# Heavily right-skewed columns get a log1p transform before scaling
LOG_FEATURES = ["n_orders", "basket_size", "distinct_products"]

for _p in (DATA_DIR, MODELS_DIR, REPORTS_DIR, FIG_DIR):
    _p.mkdir(parents=True, exist_ok=True)