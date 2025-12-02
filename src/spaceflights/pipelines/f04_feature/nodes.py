import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def add_cluster_feature(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    feature_cols = params.get("features", [])
    label_column = params.get("cluster_feature", {}).get("label_column", "cluster_label")
    kmeans_params = params.get("clustering", {}).get("kmeans", {})

    if not feature_cols:
        return df

    features = df[feature_cols].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    model = KMeans(**kmeans_params, n_init="auto")
    labels = pd.Series(model.fit_predict(scaled), index=features.index)

    enriched = df.copy()
    enriched[label_column] = None
    enriched.loc[labels.index, label_column] = labels
    return enriched


def pca_kmeans_analysis(df: pd.DataFrame, n_components: int = 2, n_clusters: int = 3):
    #columnas numéricas y quitamos filas con nulos :)
    X = df.select_dtypes(include=["number"]).dropna()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    comps = pca.fit_transform(Xs)
    pca_df = pd.DataFrame(comps, index=X.index, columns=[f"pca_{i+1}" for i in range(n_components)])

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = km.fit_predict(pca_df)
    pca_df["cluster"] = labels

    return pca_df, km, pca


def generate_pca_report(pca_df: pd.DataFrame) -> pd.DataFrame:
    return pca_df.groupby("cluster").mean(numeric_only=True).reset_index()
