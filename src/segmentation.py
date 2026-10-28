"""Customer segmentation: K-Means (with elbow + silhouette selection) and
Agglomerative Hierarchical clustering (Ward linkage + dendrogram)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import CLUSTER_FEATURES, FIG_DIR, LOG_FEATURES, RANDOM_STATE


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=130)
    plt.close(fig)


def transform(user_feats: pd.DataFrame, scaler: StandardScaler | None = None):
    """log1p the skewed columns, then standardise.  Scaler is fitted on TRAINING users only."""
    X = user_feats[CLUSTER_FEATURES].astype("float64").copy()
    for c in LOG_FEATURES:
        X[c] = np.log1p(X[c])
    if scaler is None:
        scaler = StandardScaler().fit(X)
    return scaler.transform(X), scaler


def sweep_kmeans(Xs: np.ndarray, k_range=range(2, 11), seed: int = RANDOM_STATE) -> pd.DataFrame:
    rows = []
    sil_idx = np.random.default_rng(seed).choice(len(Xs), size=min(8000, len(Xs)), replace=False)
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(Xs[sil_idx], km.labels_[sil_idx]),
                     "davies_bouldin": davies_bouldin_score(Xs, km.labels_)})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(df["k"], df["inertia"], marker="o", color="#3b528b")
    axes[0].set(title="Elbow method", xlabel="k", ylabel="inertia (WCSS)")
    axes[1].plot(df["k"], df["silhouette"], marker="o", color="#21918c")
    axes[1].set(title="Silhouette score (higher = better)", xlabel="k")
    axes[2].plot(df["k"], df["davies_bouldin"], marker="o", color="#5ec962")
    axes[2].set(title="Davies-Bouldin (lower = better)", xlabel="k")
    _save(fig, "10_kmeans_k_selection")
    return df


def choose_k(sweep: pd.DataFrame, k_min: int = 3, k_max: int = 6) -> int:
    """Best silhouette inside a business-sensible range (3-6 actionable segments)."""
    sub = sweep[(sweep["k"] >= k_min) & (sweep["k"] <= k_max)]
    return int(sub.loc[sub["silhouette"].idxmax(), "k"])


def fit_kmeans(Xs: np.ndarray, k: int, seed: int = RANDOM_STATE) -> KMeans:
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Xs)


def hierarchical(Xs: np.ndarray, k: int, kmeans_labels: np.ndarray, n_sample: int = 5000, seed: int = RANDOM_STATE) -> dict:
    """Agglomerative clustering is O(n^2) - run it on a random sample of users."""
    idx = np.random.default_rng(seed).choice(len(Xs), size=min(n_sample, len(Xs)), replace=False)
    Xsub = Xs[idx]
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(Xsub)
    Z = linkage(Xsub[:min(600, len(Xsub))], method="ward")  # dendrogram on a small subset stays readable

    fig, ax = plt.subplots(figsize=(11, 4.5))
    dendrogram(Z, truncate_mode="lastp", p=24, ax=ax, color_threshold=Z[-(k - 1), 2], no_labels=True)
    ax.set(title=f"Hierarchical clustering dendrogram (Ward, 600-user subset, cut at k={k})",
           ylabel="Ward distance")
    _save(fig, "11_dendrogram")

    return {
        "n_sample": int(len(idx)),
        "silhouette": float(silhouette_score(Xsub, agg.labels_)),
        "davies_bouldin": float(davies_bouldin_score(Xsub, agg.labels_)),
        "adjusted_rand_vs_kmeans": float(adjusted_rand_score(kmeans_labels[idx], agg.labels_)),
    }


def profile_clusters(user_feats: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    df = user_feats[CLUSTER_FEATURES].copy()
    df["segment"] = labels
    prof = df.groupby("segment").mean()
    prof.insert(0, "users", df.groupby("segment").size())
    prof.insert(1, "share_pct", (prof["users"] / prof["users"].sum() * 100).round(1))
    return prof


def name_clusters(prof: pd.DataFrame, all_feats: pd.DataFrame) -> dict[int, str]:
    """Rule-based, human-readable names from the two most distinctive traits of each segment."""
    mu, sd = all_feats[CLUSTER_FEATURES].mean(), all_feats[CLUSTER_FEATURES].std().replace(0, 1)
    z = (prof[CLUSTER_FEATURES] - mu) / sd
    words = {
        "n_orders": ("Frequent", "Occasional"),
        "basket_size": ("Big-basket", "Small-basket"),
        "reorder_ratio": ("Loyal", "Explorer"),
        "organic_share": ("Organic", None),
        "produce_share": ("Fresh-produce", None),
        "snacks_share": ("Snack", None),
        "avg_days_between": ("Infrequent", "Rapid-restock"),
        "distinct_products": ("Wide-range", "Narrow-range"),
    }
    names, used = {}, set()
    for seg, row in z.iterrows():
        cand = [(abs(row[f]), words[f][0] if row[f] > 0 else words[f][1]) for f in words]
        cand = sorted([c for c in cand if c[1]], reverse=True)
        name = " ".join(w for _, w in cand[:2]) + " shoppers"
        if name in used:
            name += f" (#{seg})"
        used.add(name)
        names[int(seg)] = name
    return names


def plot_segments(Xs: np.ndarray, labels: np.ndarray, names: dict[int, str], prof: pd.DataFrame,
                  all_feats: pd.DataFrame) -> None:
    n = min(6000, len(Xs))
    idx = np.random.default_rng(RANDOM_STATE).choice(len(Xs), size=n, replace=False)
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(Xs)
    P = pca.transform(Xs[idx])
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for s in sorted(names):
        m = labels[idx] == s
        ax.scatter(P[m, 0], P[m, 1], s=10, alpha=0.55, label=names[s])
    ax.set(title=f"Customer segments in PCA space (explained variance {pca.explained_variance_ratio_.sum():.0%})",
           xlabel="PC1", ylabel="PC2")
    ax.legend(markerscale=2, fontsize=9)
    _save(fig, "12_segments_pca")

    mu, sd = all_feats[CLUSTER_FEATURES].mean(), all_feats[CLUSTER_FEATURES].std().replace(0, 1)
    z = ((prof[CLUSTER_FEATURES] - mu) / sd)
    z.index = [names[i] for i in z.index]
    fig, ax = plt.subplots(figsize=(10, 3.2 + 0.5 * len(z)))
    sns.heatmap(z, annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Segment profiles (z-score vs. all customers: red = above average, blue = below)")
    _save(fig, "13_segment_profile_heatmap")

    sizes = prof["users"].rename(index=names)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=sizes.index, autopct="%1.0f%%", startangle=90, textprops={"fontsize": 9},
           colors=sns.color_palette("viridis", len(sizes)))
    ax.set_title("Share of customers per segment")
    _save(fig, "14_segment_sizes")
