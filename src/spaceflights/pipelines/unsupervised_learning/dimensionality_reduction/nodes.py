from __future__ import annotations

from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap


def perform_pca(df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.DataFrame, PCA, pd.DataFrame, pd.DataFrame]:
    feature_cols = params.get("features", [])
    pca_params = params.get("dimensionality_reduction", {}).get("pca", {})

    if not feature_cols:
        raise ValueError("No feature columns provided for PCA")

    X = df[feature_cols].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)

    model = PCA(**pca_params)
    components = model.fit_transform(scaled)
    component_names = [f"pca_{i+1}" for i in range(model.n_components_)]
    pca_components = pd.DataFrame(components, index=X.index, columns=component_names)

    variance = pd.DataFrame(
        {
            "component": component_names,
            "explained_variance_ratio": model.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(model.explained_variance_ratio_),
        }
    )

    loadings = pd.DataFrame(
        model.components_.T,
        index=feature_cols,
        columns=component_names,
    )

    return pca_components, model, variance, loadings


def plot_pca_variance(variance: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(variance["component"], variance["explained_variance_ratio"], label="Varianza")
    ax.plot(variance["component"], variance["cumulative_variance"], marker="o", color="red", label="Acumulada")
    ax.set_xlabel("Componentes")
    ax.set_ylabel("Varianza explicada")
    ax.legend()
    ax.set_title("PCA - Varianza explicada")
    fig.tight_layout()
    return fig


def plot_pca_biplot(components: pd.DataFrame, loadings: pd.DataFrame) -> Figure:
    if components.shape[1] < 2:
        raise ValueError("Se necesitan al menos 2 componentes para generar el biplot")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(components.iloc[:, 0], components.iloc[:, 1], alpha=0.6)

    for feature in loadings.index:
        ax.arrow(0, 0, loadings.loc[feature, components.columns[0]] * 3, loadings.loc[feature, components.columns[1]] * 3,
                 color='red', head_width=0.05)
        ax.text(loadings.loc[feature, components.columns[0]] * 3.2, loadings.loc[feature, components.columns[1]] * 3.2,
                feature, color='red')

    ax.set_xlabel(components.columns[0])
    ax.set_ylabel(components.columns[1])
    ax.set_title("Biplot PCA")
    fig.tight_layout()
    return fig


def run_umap(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    feature_cols = params.get("features", [])
    umap_params = params.get("dimensionality_reduction", {}).get("umap", {})

    if not feature_cols:
        raise ValueError("No feature columns provided for UMAP")

    X = df[feature_cols].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)

    reducer = umap.UMAP(**umap_params)
    embedding = reducer.fit_transform(scaled)

    return pd.DataFrame(embedding, index=X.index, columns=["umap_1", "umap_2"])


def plot_umap(embedding: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(embedding["umap_1"], embedding["umap_2"], s=10, alpha=0.7)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Embeddings UMAP")
    fig.tight_layout()
    return fig