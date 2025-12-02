from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _compute_metrics(features: pd.DataFrame, labels: pd.Series) -> Dict[str, float | None]:
    unique_labels = set(labels)

    valid_clusters = [label for label in unique_labels if label != -1]
    if len(valid_clusters) < 2:
        return {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}

    try:
        silhouette = silhouette_score(features, labels)
        calinski = calinski_harabasz_score(features, labels)
        davies = davies_bouldin_score(features, labels)
        return {
            "silhouette": silhouette,
            "calinski_harabasz": calinski,
            "davies_bouldin": davies,
        }
    except ValueError:

        return {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}


def run_clustering(
    df: pd.DataFrame,
    params: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, Figure, pd.DataFrame]:

    feature_cols = params.get("features", [])
    label_column = params.get("cluster_feature", {}).get("label_column", "cluster_label")

    if not feature_cols:
        raise ValueError("No feature columns provided for clustering")

    features = df[feature_cols].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans_params = params.get("clustering", {}).get("kmeans", {})
    kmeans = KMeans(**kmeans_params, n_init="auto")
    kmeans_labels = pd.Series(kmeans.fit_predict(scaled), index=features.index)

    dbscan_params = params.get("clustering", {}).get("dbscan", {})
    dbscan = DBSCAN(**dbscan_params)
    dbscan_labels = pd.Series(dbscan.fit_predict(scaled), index=features.index)

    gmm_params = params.get("clustering", {}).get("gmm", {})
    gmm = GaussianMixture(**gmm_params)
    gmm_labels = pd.Series(gmm.fit_predict(scaled), index=features.index)

    labels_df = pd.DataFrame(
        {
            "kmeans_label": kmeans_labels,
            "dbscan_label": dbscan_labels,
            "gmm_label": gmm_labels,
        }
    )
    labels_df[label_column] = labels_df["kmeans_label"]

    metrics = []
    for algo, algo_labels in (
        ("kmeans", kmeans_labels),
        ("dbscan", dbscan_labels),
        ("gmm", gmm_labels),
    ):
        metric_values = _compute_metrics(pd.DataFrame(scaled, index=features.index), algo_labels)
        metrics.append({"algorithm": algo, **metric_values})

    metrics_df = pd.DataFrame(metrics)

    fig = _plot_clusters(scaled, labels_df, params)

    cluster_feature_table = df.copy()
    cluster_feature_table[label_column] = None
    cluster_feature_table.loc[labels_df.index, label_column] = labels_df[label_column]

    return labels_df, metrics_df, fig, cluster_feature_table


def _plot_clusters(features: Any, labels_df: pd.DataFrame, params: Dict[str, Any]) -> Figure:
    pca = PCA(n_components=2, random_state=params.get("clustering", {}).get("kmeans", {}).get("random_state", 42))
    embedding = pca.fit_transform(features)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    algorithms = [
        ("K-Means", labels_df["kmeans_label"]),
        ("DBSCAN", labels_df["dbscan_label"]),
        ("GMM", labels_df["gmm_label"]),
    ]

    for ax, (title, labels) in zip(axes, algorithms):
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=20)
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        if len(pd.unique(labels)) <= 10:
            fig.colorbar(scatter, ax=ax)

    fig.tight_layout()
    return fig