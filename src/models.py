"""Supervised models for 'will the customer reorder this product in their next order?'

Models compared: Logistic Regression, k-NN, Decision Tree, Random Forest, AdaBoost
(+ a majority-class baseline).  Each is tuned with RandomizedSearchCV using GroupKFold
(grouped by user so no customer leaks across folds).  The decision threshold is chosen on a
VALIDATION split of training users - never on the test set.
"""
from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import FIG_DIR, RANDOM_STATE

MODEL_COLORS = {"Logistic Regression": "#440154", "k-NN": "#3b528b", "Decision Tree": "#21918c",
                "Random Forest": "#5ec962", "AdaBoost": "#fde725", "Gradient Boosting": "#f58231",
                "Baseline (majority)": "#999999"}


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=130)
    plt.close(fig)


def model_zoo(seed: int = RANDOM_STATE) -> dict:
    """name -> (estimator, param_distributions, max_training_rows)"""
    return {
        "Logistic Regression": (
            Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=300, class_weight="balanced"))]),
            {"m__C": [0.01, 0.1, 1, 10]}, 400_000),
        "k-NN": (
            Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier(n_jobs=-1))]),
            {"m__n_neighbors": [15, 31, 51, 101], "m__weights": ["uniform", "distance"]}, 60_000),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=seed),
            {"max_depth": [4, 6, 8, 12, 16], "min_samples_leaf": [10, 50, 200], "criterion": ["gini", "entropy"]}, 400_000),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
            {"n_estimators": [60, 100, 150], "max_depth": [8, 12, 16], "min_samples_leaf": [5, 20, 50],
             "max_features": ["sqrt", 0.5]}, 500_000),
        "AdaBoost": (
            AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3, class_weight="balanced"), random_state=seed),
            {"n_estimators": [50, 100, 150], "learning_rate": [0.1, 0.5, 1.0]}, 150_000),
        "Gradient Boosting": (
            HistGradientBoostingClassifier(class_weight="balanced", early_stopping=True,
                                           validation_fraction=0.1, random_state=seed),
            {"learning_rate": [0.05, 0.1, 0.2], "max_iter": [200, 400], "max_leaf_nodes": [31, 63, 127],
             "min_samples_leaf": [20, 100], "l2_regularization": [0.0, 1.0]}, 1_000_000),
    }


def user_split(df: pd.DataFrame, frac: float = 0.15, seed: int = RANDOM_STATE):
    """Return (fit_idx, val_idx) positional indices, split by user."""
    gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    return next(gss.split(df, groups=df["user_id"]))


def best_threshold(y_true, proba) -> float:
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-9, None)
    return float(t[int(np.argmax(f1))])


def metrics(y_true, proba, thr: float) -> dict:
    pred = (proba >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "avg_precision": average_precision_score(y_true, proba),
    }


def train_and_compare(cand_train: pd.DataFrame, cand_test: pd.DataFrame, features: list[str],
                      n_iter: int = 8, cv: int = 3, tune_rows: int = 80_000, eval_rows: int = 250_000,
                      seed: int = RANDOM_STATE, verbose: bool = True):
    """Tune + fit every model; evaluate on held-out USERS.  Returns (results_df, fitted, extras)."""
    rng = np.random.default_rng(seed)
    fit_idx, val_idx = user_split(cand_train, 0.15, seed)
    fit_df, val_df = cand_train.iloc[fit_idx], cand_train.iloc[val_idx]
    if len(cand_test) > eval_rows:
        cand_test = cand_test.iloc[rng.choice(len(cand_test), eval_rows, replace=False)]
    Xte, yte = cand_test[features], cand_test["label"].to_numpy()
    Xval, yval = val_df[features], val_df["label"].to_numpy()

    fitted, rows, curves, cms = {}, [], {}, {}
    zoo = {"Baseline (majority)": (DummyClassifier(strategy="prior"), None, 10_000)} | model_zoo(seed)

    for name, (est, params, max_rows) in zoo.items():
        t0 = time.time()
        sub = fit_df if len(fit_df) <= max_rows else fit_df.iloc[rng.choice(len(fit_df), max_rows, replace=False)]
        best_params = {}
        if params:
            tune = sub if len(sub) <= tune_rows else sub.iloc[rng.choice(len(sub), tune_rows, replace=False)]
            search = RandomizedSearchCV(
                est, params, n_iter=min(n_iter, int(np.prod([len(v) for v in params.values()]))),
                scoring="average_precision", cv=GroupKFold(n_splits=cv), n_jobs=1,
                random_state=seed, refit=False)
            search.fit(tune[features], tune["label"], groups=tune["user_id"])
            best_params = search.best_params_
            est.set_params(**best_params)
        est.fit(sub[features], sub["label"])
        train_time = time.time() - t0

        # threshold from validation users, applied unchanged to the test users
        thr = best_threshold(yval, est.predict_proba(Xval)[:, 1]) if params else 0.5
        proba = est.predict_proba(Xte)[:, 1]
        m_def = metrics(yte, proba, 0.5)
        m_tuned = metrics(yte, proba, thr)
        rows.append({
            "model": name, "train_rows": len(sub), "train_time_s": round(train_time, 1),
            "accuracy@0.5": m_def["accuracy"], "precision@0.5": m_def["precision"],
            "recall@0.5": m_def["recall"], "f1@0.5": m_def["f1"],
            "threshold": thr, "accuracy": m_tuned["accuracy"], "precision": m_tuned["precision"],
            "recall": m_tuned["recall"], "f1": m_tuned["f1"],
            "roc_auc": m_tuned["roc_auc"], "avg_precision": m_tuned["avg_precision"],
            "best_params": str(best_params),
        })
        cms[name] = {"default_0.5": confusion_matrix(yte, (proba >= 0.5).astype(int)).tolist(),
                     "tuned_threshold": confusion_matrix(yte, (proba >= thr).astype(int)).tolist()}
        curves[name] = {"proba": proba}
        fitted[name] = (est, thr)
        if verbose:
            print(f"  {name:22s} F1={m_tuned['f1']:.3f}  AUC={m_tuned['roc_auc']:.3f}  "
                  f"P={m_tuned['precision']:.3f}  R={m_tuned['recall']:.3f}  ({train_time:.0f}s)")

    res = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    return res, fitted, {"y_test": yte, "curves": curves, "confusion": cms, "test_df": cand_test,
                         "fit_df": fit_df, "val_df": val_df}


# --------------------------------------------------------------------------------------- plots
def plot_comparison(res: pd.DataFrame, extras: dict) -> None:
    r = res[res["model"] != "Baseline (majority)"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    metric_cols = ["precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(r)); w = 0.2
    for i, mc in enumerate(metric_cols):
        axes[0].bar(x + i * w - 1.5 * w, r[mc], w, label=mc.upper().replace("_", "-"),
                    color=sns.color_palette("viridis", 4)[i])
    axes[0].set_xticks(x); axes[0].set_xticklabels(r["model"], rotation=20, ha="right")
    axes[0].set(title="Model comparison on held-out customers", ylim=(0, 1))
    axes[0].legend(ncol=4, fontsize=8, loc="upper center")

    for name, c in extras["curves"].items():
        if name.startswith("Baseline"):
            continue
        fpr, tpr, _ = roc_curve(extras["y_test"], c["proba"])
        auc = res.loc[res["model"] == name, "roc_auc"].iloc[0]
        axes[1].plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})", color=MODEL_COLORS[name])
    axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[1].set(title="ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].legend(fontsize=8)
    _save(fig, "20_model_comparison")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, c in extras["curves"].items():
        if name.startswith("Baseline"):
            continue
        p, rc, _ = precision_recall_curve(extras["y_test"], c["proba"])
        ax.plot(rc, p, label=name, color=MODEL_COLORS[name])
    ax.axhline(extras["y_test"].mean(), color="k", ls="--", lw=0.8, label="no-skill")
    ax.set(title="Precision-Recall curves", xlabel="Recall", ylabel="Precision")
    ax.legend(fontsize=8)
    _save(fig, "21_precision_recall")

    names = [n for n in extras["confusion"] if not n.startswith("Baseline")]
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.6))
    for ax, n in zip(np.atleast_1d(axes), names):
        cm = np.array(extras["confusion"][n]["tuned_threshold"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", cbar=False, ax=ax,
                    xticklabels=["Not reordered", "Reordered"], yticklabels=["Not reordered", "Reordered"])
        ax.set(title=n, xlabel="Predicted", ylabel="Actual")
    _save(fig, "22_confusion_matrices")


def feature_importance_plot(fitted: dict, features: list[str]) -> pd.DataFrame:
    rf = fitted["Random Forest"][0]
    imp = pd.DataFrame({"feature": features, "importance": rf.feature_importances_}).sort_values("importance")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh(imp["feature"], imp["importance"], color=sns.color_palette("viridis", len(imp)))
    ax.set(title="Random Forest feature importance", xlabel="importance")
    _save(fig, "23_feature_importance")
    return imp.sort_values("importance", ascending=False)


def rf_parameter_impact(extras: dict, features: list[str], seed: int = RANDOM_STATE, n_rows: int = 60_000) -> pd.DataFrame:
    """Show how ensemble parameters (n_estimators, max_depth) change performance."""
    rng = np.random.default_rng(seed)
    fit_df, test_df = extras["fit_df"], extras["test_df"]
    sub = fit_df.iloc[rng.choice(len(fit_df), min(n_rows, len(fit_df)), replace=False)]
    te = test_df.iloc[rng.choice(len(test_df), min(60_000, len(test_df)), replace=False)]
    rows = []
    for n_est in [5, 10, 25, 50, 100, 200]:
        rf = RandomForestClassifier(n_estimators=n_est, max_depth=12, min_samples_leaf=20,
                                    class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
        rf.fit(sub[features], sub["label"])
        p = rf.predict_proba(te[features])[:, 1]
        rows.append({"param": "n_estimators", "value": n_est, "roc_auc": roc_auc_score(te["label"], p),
                     "avg_precision": average_precision_score(te["label"], p)})
    for depth in [2, 4, 6, 8, 12, 16, 24]:
        rf = RandomForestClassifier(n_estimators=100, max_depth=depth, min_samples_leaf=20,
                                    class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
        rf.fit(sub[features], sub["label"])
        ptr = rf.predict_proba(sub[features])[:, 1]
        p = rf.predict_proba(te[features])[:, 1]
        rows.append({"param": "max_depth", "value": depth, "roc_auc": roc_auc_score(te["label"], p),
                     "avg_precision": average_precision_score(te["label"], p),
                     "train_roc_auc": roc_auc_score(sub["label"], ptr)})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    a = df[df["param"] == "n_estimators"]
    axes[0].plot(a["value"], a["roc_auc"], marker="o", color="#21918c", label="test ROC-AUC")
    axes[0].plot(a["value"], a["avg_precision"], marker="s", color="#5ec962", label="test avg. precision")
    axes[0].set(title="Effect of number of trees (max_depth=12)", xlabel="n_estimators"); axes[0].legend()
    b = df[df["param"] == "max_depth"]
    axes[1].plot(b["value"], b["roc_auc"], marker="o", color="#21918c", label="test ROC-AUC")
    axes[1].plot(b["value"], b["train_roc_auc"], marker="o", ls="--", color="#440154", label="train ROC-AUC")
    axes[1].set(title="Effect of tree depth: over-fitting gap (100 trees)", xlabel="max_depth"); axes[1].legend()
    _save(fig, "24_rf_parameter_impact")
    return df