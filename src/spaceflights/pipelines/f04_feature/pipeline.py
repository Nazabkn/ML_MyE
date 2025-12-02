from kedro.pipeline import Pipeline, node
from .nodes import add_cluster_feature, pca_kmeans_analysis, generate_pca_report

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=add_cluster_feature,
            inputs=["model_input_table", "params:unsupervised_learning"],
            outputs="model_input_with_clusters",
            name="add_cluster_feature_node",
        ),
        node(
            func=pca_kmeans_analysis,
            inputs=["model_input_with_clusters", "params:parameters_data_processing.multivar.pca_n_components", "params:parameters_data_processing.multivar.kmeans_k"],
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
