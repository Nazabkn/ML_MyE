from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    build_splits_reg,
    model_selection_reg,
    predict_reg,
    export_reg_leaderboard,
)


def create_pipeline() -> Pipeline:
    return pipeline([
        node(
            func=build_splits_reg,
            inputs=dict(
                model_input_table="model_input_table",
                params="params:parameters_data_processing",
            ),
            outputs=["X_train_reg", "X_test_reg", "y_train_reg", "y_test_reg", "reg_cols"],
            name="build_splits_reg07_node",
        ),
        node(
            func=model_selection_reg,
            inputs=[
                "X_train_reg", "y_train_reg",
                "X_test_reg", "y_test_reg",
                "reg_cols",
                "params:parameters_data_processing",
                "params:models_reg",
            ],
            outputs=["best_regressor", "reg_cv_results"],
            name="model_selection_reg07_node",
        ),
        node(
            func=predict_reg,
            inputs=["best_regressor", "X_test_reg"],
            outputs="y_pred_reg",
            name="predict_reg07_node",
        ),
        node(
            func=export_reg_leaderboard,
            inputs="reg_cv_results",
            outputs="reg_results_table",
            name="export_reg_leaderboard07_node",
        ),
    ])
