from kedro.pipeline import Pipeline, node
from .nodes import pca_kmeans_analysis, generate_pca_report

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=pca_kmeans_analysis,
            inputs="model_input_table",
            outputs=["pca_features", "kmeans_model", "pca_model"],
            name="pca_kmeans_analysis_node",
        ),
        node(
            func=generate_pca_report,
            inputs="pca_features",
            outputs="pca_kmeans_report",
            name="generate_pca_report_node",
        ),
    ])
