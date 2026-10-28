"""Streamlit app: Instacart Customer Segmentation & Reorder Prediction System.

Run the pipeline first (python run_pipeline.py ...), then:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
REPORTS, FIGS, MODELS = ROOT / "reports", ROOT / "reports" / "figures", ROOT / "models"

st.set_page_config(page_title="Instacart Customer Intelligence", page_icon="🛒", layout="wide")


def need_artifacts():
    if not (REPORTS / "meta.json").exists():
        st.error("No results found. Run `python run_pipeline.py` first (see README.md).")
        st.stop()


@st.cache_data
def load_json(name):
    return json.loads((REPORTS / name).read_text())


@st.cache_data
def load_csv(name, **kw):
    return pd.read_csv(REPORTS / name, **kw)


@st.cache_resource
def load_model(fname):
    return joblib.load(MODELS / fname)


def fig(name, caption=None):
    p = FIGS / f"{name}.png"
    if p.exists():
        try:  # newer Streamlit
            st.image(str(p), caption=caption, width="stretch")
        except Exception:  # older Streamlit (e.g. 1.45)
            st.image(str(p), caption=caption, use_container_width=True)


def model_file(name):
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "") + ".joblib"


need_artifacts()
meta = load_json("meta.json")
eda_stats = load_json("eda_summary.json")
comp = load_csv("model_comparison.csv")
profiles = load_csv("segment_profiles.csv", index_col="segment")
seg_names = {int(k): v for k, v in meta["segment_names"].items()}

st.title("🛒 Instacart Customer Segmentation & Reorder Prediction")
st.caption("Machine Learning Lab mini-project · EDA · K-Means & Hierarchical clustering · "
           "Decision Tree, Random Forest, k-NN, Logistic Regression, AdaBoost")
if meta["synthetic_data"]:
    st.warning("These results come from SYNTHETIC demo data in the Instacart schema. "
               "Re-run the pipeline on the real Kaggle files for genuine results.")

tab_over, tab_eda, tab_seg, tab_models, tab_pred = st.tabs(
    ["📌 Overview", "🔎 EDA", "👥 Segments", "🤖 Models", "🎯 Predict"])

# ----------------------------------------------------------------------------- Overview
with tab_over:
    h = meta["headline"]
    c = st.columns(5)
    c[0].metric("Customers analysed", f"{h['users']:,}")
    c[1].metric("Orders", f"{h['orders']:,}")
    c[2].metric("Items purchased (prior)", f"{h['prior_items']:,}")
    c[3].metric("Reorder rate", f"{h['reorder_rate']:.0%}")
    c[4].metric("Avg basket", f"{h['avg_basket_size']:.1f} items")

    st.subheader("What this project does")
    st.markdown(
        """
1. **Explore** grocery-shopping behaviour (EDA: univariate, bivariate, multivariate, outliers).
2. **Segment** customers with **K-Means** (k chosen via elbow + silhouette) and cross-check with
   **Agglomerative Hierarchical** clustering.
3. **Predict** whether a customer will **reorder a product they have bought before** in their next order,
   comparing five tuned classifiers.
4. **Serve** everything through this interactive app.

**Leakage-free design:** users are split 80/20 *before* anything is learned; features use prior orders only;
the label comes from the customer's latest order; the scaler, k-means and product statistics are fitted on
training users only; hyper-parameters are tuned with user-grouped cross-validation; and the decision
threshold is chosen on a validation split, never on the test set.
        """)
    st.subheader("Headline results")
    c = st.columns(4)
    c[0].metric("Segments found (k)", meta["k"], f"silhouette {meta['silhouette_kmeans']:.2f}")
    c[1].metric("Best model", meta["best_model"])
    c[2].metric("Best F1", f"{meta['best_f1']:.3f}")
    c[3].metric("Best ROC-AUC", f"{meta['best_auc']:.3f}")

# ----------------------------------------------------------------------------- EDA
with tab_eda:
    st.subheader("Dataset health check")
    t = eda_stats["tables"]
    rows = [{"table": k, "rows": v["shape"][0], "columns": v["shape"][1],
             "columns with missing values": ", ".join(v["missing"]) or "-", "duplicate rows": v["duplicate_rows"]}
            for k, v in t.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("`days_since_prior_order` is empty for each customer's first order by design.")

    c1, c2 = st.columns(2)
    c1.markdown("**`orders.head()`**"); c1.dataframe(load_csv("eda_orders_head.csv"), hide_index=True)
    c2.markdown("**`orders.tail()`**"); c2.dataframe(load_csv("eda_orders_tail.csv"), hide_index=True)
    st.markdown("**`describe()` of user-level features**")
    st.dataframe(load_csv("eda_user_features_describe.csv", index_col=0).round(3))
    st.markdown("**Outliers (IQR rule) in user-level features**")
    st.dataframe(load_csv("eda_outliers.csv").round(2), hide_index=True)

    st.subheader("Univariate")
    fig("01_orders_dow_hour"); fig("02_distributions"); fig("03_top_products"); fig("04_departments")
    st.subheader("Bivariate"); fig("05_bivariate")
    st.subheader("Multivariate"); fig("06_correlation"); fig("07_pairplot"); fig("08_outlier_boxplots")

# ----------------------------------------------------------------------------- Segments
with tab_seg:
    st.subheader("How many segments?")
    fig("10_kmeans_k_selection")
    hier = meta["hierarchical"]
    c = st.columns(3)
    c[0].metric("K-Means silhouette", f"{meta['silhouette_kmeans']:.3f}")
    c[1].metric("Hierarchical silhouette", f"{hier['silhouette']:.3f}")
    c[2].metric("Agreement (Adjusted Rand)", f"{hier['adjusted_rand_vs_kmeans']:.2f}")
    fig("11_dendrogram")

    st.subheader("Who are the segments?")
    show = profiles.copy()
    st.dataframe(show.round(2))
    c1, c2 = st.columns([3, 2])
    with c1: fig("13_segment_profile_heatmap")
    with c2: fig("14_segment_sizes")
    fig("12_segments_pca")

    st.subheader("🧭 Which segment is this customer?")
    st.caption("Move the sliders to describe a shopper; the trained K-Means model assigns a segment.")
    seg_art = load_model("segmentation.joblib")
    med = profiles.drop(columns=["segment_name", "users", "share_pct"]).mean()
    c = st.columns(3)
    vals = {
        "n_orders": c[0].slider("Number of past orders", 3, 100, 20),
        "avg_days_between": c[1].slider("Avg days between orders", 1, 30, 10),
        "basket_size": c[2].slider("Avg items per order", 1, 50, 10),
        "reorder_ratio": c[0].slider("Share of items that are re-buys", 0.0, 1.0, float(round(med["reorder_ratio"], 2))),
        "distinct_products": c[1].slider("Distinct products bought", 5, 600, 80),
        "organic_share": c[2].slider("Organic share", 0.0, 1.0, float(round(med["organic_share"], 2))),
        "produce_share": c[0].slider("Produce share", 0.0, 1.0, float(round(med["produce_share"], 2))),
        "snacks_share": c[1].slider("Snacks share", 0.0, 1.0, float(round(med["snacks_share"], 2))),
        "avg_order_hour": c[2].slider("Typical order hour", 0, 23, 14),
        "weekend_share": c[0].slider("Share of weekend orders", 0.0, 1.0, float(round(med["weekend_share"], 2))),
    }
    from src.segmentation import transform
    x = pd.DataFrame([vals])
    xs, _ = transform(x, seg_art["scaler"])
    seg = int(seg_art["kmeans"].predict(xs)[0])
    st.success(f"**Segment {seg}: {seg_names[seg]}**")
    st.dataframe(profiles.loc[[seg]].round(2))

# ----------------------------------------------------------------------------- Models
with tab_models:
    st.subheader("Model comparison (held-out customers)")
    show = comp[["model", "precision", "recall", "f1", "accuracy", "roc_auc", "avg_precision", "threshold", "train_time_s"]]
    st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns if c not in ("model", "train_time_s")})
                 .background_gradient(subset=["f1", "roc_auc"], cmap="viridis"), hide_index=True)
    st.caption("Metrics use a decision threshold picked on a validation split of training users. "
               "Accuracy is shown but misleading here because most (user, product) pairs are NOT reordered - "
               "focus on precision, recall, F1 and ROC-AUC.")
    fig("20_model_comparison")
    c1, c2 = st.columns([1, 1.6])
    with c1: fig("21_precision_recall")
    with c2: fig("22_confusion_matrices")

    st.subheader("Tuned hyper-parameters")
    st.dataframe(comp[["model", "best_params"]], hide_index=True)

    st.subheader("Random Forest: what matters, and what ensemble parameters do")
    c1, c2 = st.columns(2)
    with c1: fig("23_feature_importance")
    with c2: fig("24_rf_parameter_impact")

    st.subheader("Confusion matrix explorer")
    cms = load_json("confusion_matrices.json")
    pick = st.selectbox("Model", [m for m in cms if not m.startswith("Baseline")])
    which = st.radio("Threshold", ["tuned_threshold", "default_0.5"], horizontal=True)
    cm = np.array(cms[pick][which])
    st.dataframe(pd.DataFrame(cm, index=["Actual: not reordered", "Actual: reordered"],
                              columns=["Pred: not reordered", "Pred: reordered"]))

# ----------------------------------------------------------------------------- Predict
with tab_pred:
    st.subheader("What will this customer reorder next?")
    st.caption("Pick a held-out customer (never seen in training) and a model. The app ranks the products "
               "they have bought before by predicted probability of appearing in their next order, "
               "and shows what they actually bought.")
    cand = load_csv("app_candidates.csv.gz")
    users = load_csv("app_users.csv.gz", index_col=0)
    c = st.columns([1, 1, 1])
    uid = c[0].selectbox("Customer ID", sorted(cand["user_id"].unique()))
    mname = c[1].selectbox("Model", [m for m in comp["model"] if not m.startswith("Baseline")],
                           index=0)
    topn = c[2].slider("Show top N products", 5, 30, 10)

    art = load_model(model_file(mname))
    sub = cand[cand["user_id"] == uid].copy()
    sub["reorder_probability"] = art["model"].predict_proba(sub[art["features"]])[:, 1]
    sub = sub.sort_values("reorder_probability", ascending=False)
    sub["predicted_reorder"] = np.where(sub["reorder_probability"] >= art["threshold"], "Yes", "No")
    sub["actually_reordered"] = np.where(sub["label"] == 1, "✅ Yes", "—")

    u = users.loc[uid]
    st.info(f"**Segment {int(u['segment'])}: {seg_names[int(u['segment'])]}** · {int(u['n_orders'])} past orders · "
            f"{u['basket_size']:.1f} items/order · re-buy ratio {u['reorder_ratio']:.0%}")
    top = sub.head(topn)[["product_name", "reorder_probability", "predicted_reorder", "actually_reordered"]]
    st.dataframe(top.style.format({"reorder_probability": "{:.1%}"}).background_gradient(
        subset=["reorder_probability"], cmap="viridis"), hide_index=True)

    pred_set = set(sub.loc[sub["predicted_reorder"] == "Yes", "product_id"])
    true_set = set(sub.loc[sub["label"] == 1, "product_id"])
    hit = len(pred_set & true_set)
    p = hit / len(pred_set) if pred_set else 0
    r = hit / len(true_set) if true_set else 0
    c = st.columns(3)
    c[0].metric("Products recommended", len(pred_set))
    c[1].metric("Correct (precision)", f"{p:.0%}")
    c[2].metric("Caught (recall)", f"{r:.0%}")

st.divider()
st.caption("Data: Instacart Market Basket Analysis (Kaggle). Built with pandas, scikit-learn, matplotlib, seaborn and Streamlit.")