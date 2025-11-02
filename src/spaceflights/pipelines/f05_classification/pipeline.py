from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    build_splits,
    model_selection,
    predict,
    export_leaderboard,
    plot_cv_results,
)


def create_pipeline() -> Pipeline:
    return pipeline([
        node(
            func=build_splits,
            inputs=dict(
                model_input_table="model_input_table",
                params="params:parameters_data_processing",
            ),
            outputs=["X_train", "X_test", "y_train", "y_test", "clf_cols"],
            name="build_splits_node",
        ),

        node(
            func=model_selection,
            inputs=[
                "X_train", "y_train", "X_test", "y_test", "clf_cols",
                "params:parameters_data_processing",
                "params:models_clf",
            ],
            outputs=["hazardous_clf", "clf_cv_results"],
            name="model_selection_node",
        ),

        node(
            func=predict,
            inputs=["hazardous_clf", "X_test"],
            outputs="y_pred",
            name="predict_node",
        ),

        node(
            func=export_leaderboard,
            inputs="clf_cv_results",
            outputs="clf_results_table",
            name="export_leaderboard_node",
        ),
        
         node(
            func=plot_cv_results,
            inputs=dict(
                clf_results_table="clf_results_table",
                scoring="params:models_clf.scoring",
            ),
            outputs="clf_results_plot",
            name="plot_classification_leaderboard_node",
        ),
    ])
