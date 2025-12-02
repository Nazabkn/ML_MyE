from kedro.pipeline import Pipeline, node

from .nodes import run_clustering


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=run_clustering,
            inputs=["model_input_table", "params:unsupervised_learning"],
            outputs=[
                "unsupervised_cluster_labels",
                "unsupervised_cluster_metrics",
                "unsupervised_cluster_plots",
                "unsupervised_model_input_with_clusters",
            ],
            name="run_clustering_algorithms",
        ),
    ])